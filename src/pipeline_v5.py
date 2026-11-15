"""v5 360° room pipeline (PLAN_V5): outputs that look good to a person.

    [1] panorama source   SDXL 2048 x 1024 + 360° adapter (src/generate_pano_xl.py, milestone 1) or the v4 source;
                          v5 prompt: spacious room, photographed from near a wall, no close-ups (milestone 2)
    [2] candidates        k panoramas; DAP distance + SegFormer classes for each; score = boxiness + room area +
                          camera clearance, candidates with the camera inside / touching furniture rejected
                          (src/camera_v5.py, milestone 2)
    [3] room              v4 analysis and build (Mask2Former instances, room box, blocks, openings, open sides)
    [4] 4096 panorama     Real-ESRGAN per view + multi-band blend (src/pano_refine.py, as v4)
    [5] design shots      level, 85-90° FOV; hero = best VALID view (main piece 15-50 % of the frame, median distance
                          > 1.5 m, < 15 % of the frame nearer than 0.8 m), src/shots_v5.py (milestone 2); then each
                          shot RE-RENDERED: SD 1.5 + MMIS LoRA + depth ControlNet img2img at 768 x 576, Real-ESRGAN to
                          1536 x 1152 (src/shot_render.py, milestone 3); the raw crops are kept for the Analysis section
    [6] Inside view       panorama viewer page (src/pano_viewer.py, milestone 4); the v4 depth mesh stays as the
                          "Experimental: walkable mesh" with a translation limit
    [7] dollhouse         furniture removed from wall / floor textures, flat colour on unseen block faces
                          (src/dollhouse_v5.py, milestone 5)
v4 stays callable unchanged (src/pipeline_v4.py) for before / after comparisons on the same seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import boxiness as bx
from . import camera_v5 as cam
from . import design_shots as ds
from . import pano_depth as pdp
from . import pipeline_v3 as p3v
from . import room_box as rb
from . import shots_v5 as sv

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
# milestone 2 prompt wording (camera position and room size); appended to the room description
SPACIOUS = ("spacious {room}, wide angle, photographed from a corner near the wall at eye height, furniture placed "
            "against the walls, open floor in the middle of the room")
CLOSEUP_NEG = ", close-up, macro, cluttered, cramped room, furniture in front of the camera, tabletop view"
MESH_MAX_TRANSLATION = 0.25   # m; overwritten by results/tables/v5_viewer_selection.json (milestone 4)


def decision(name: str, default=None):
    p = TAB / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def source() -> str:
    """Generator adopted in milestone 1 ('xl_redmond' / 'xl_pgg' / 'sd21')."""
    return (decision("v5_pano_decision.json") or {}).get("selected", "xl_redmond")


def prompts(room: str, kind: str, spacious: bool = True) -> tuple[str, str]:
    """(prompt, negative) for the generator `kind` and a room description."""
    from . import generate_pano as gp
    desc = SPACIOUS.format(room=room) if spacious else room
    if kind == "sd21":
        return gp.ONE_ROOM_TEMPLATE.format(room=desc), gp.ONE_ROOM_NEGATIVE + (CLOSEUP_NEG if spacious else "")
    from . import generate_pano_xl as gx
    v = kind.split("_", 1)[1]
    return gx.xl_prompt(desc, v), gx.XL_NEGATIVE + (CLOSEUP_NEG if spacious else "")


def generate(pipe, kind: str, room: str, seed: int, spacious: bool = True) -> np.ndarray:
    """One 2048 x 1024 panorama (sd21: 1024 x 512, upscale it with Real-ESRGAN afterwards)."""
    p, n = prompts(room, kind, spacious)
    if kind == "sd21":
        from . import generate_pano as gp
        return gp.generate_pano(pipe, room, seed, prompt=p, negative=n)
    from . import generate_pano_xl as gx
    return gx.generate_pano_xl(pipe, room, seed, prompt=p, negative=n)


def score_candidate(pano: np.ndarray, dap, seg, id2label: dict, device) -> dict:
    """DAP distance + SegFormer class map (1024 x 512) and the milestone-2 measures of one candidate panorama."""
    small = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
    dist = pdp.dap_depth_wrap(dap, small, device)
    cm = pdp.segment_pano(*seg, small, device)
    box = rb.fit_box(dist, cm, id2label)
    sc = bx.score(dist, cm, id2label, box)
    sc["d_near"] = cam.nearest_furniture(dist, cm, id2label, box)
    return {"dist": dist, "class_map": cm, "score": sc}


def selection_weights() -> dict:
    return (decision("v5_camera_selection.json") or {}).get("weights", {"w_area": 0.5, "w_near": 0.5})


def build(dist, texture, hires, ana, id2label, prompt, furniture_params=None) -> dict:
    """[3] + [5a]: the v4 room (box, blocks, openings, open sides) and the v5 shot selection (raw crops)."""
    res = p3v.build(dist, texture, ana, id2label, furniture_params, open_sides=True)
    res["hires"] = hires
    res["shots"] = sv.select_shots(hires, res["box"], ana["class_map"], ana["inst_map"], ana["instances"], id2label,
                                   dist, room=ds.room_type_of(prompt))
    for s in res["shots"]["shots"]:
        s["raw"] = s["image"]
    return res


def rerender(res: dict, dist, pipe, esrgan, prompt: str, strength: float, seed: int) -> None:
    """[5b] re-render every shot (in place: s['image'] = 1536 x 1152 re-render, s['raw'] = the crop)."""
    from . import shot_render as sr
    imgs = sr.rerender_shots(pipe, res["shots"]["shots"], dist, res["box"], prompt, strength, seed)
    for s, im in zip(res["shots"]["shots"], imgs):
        s["image"] = sr.upscale(esrgan, im) if esrgan is not None else im


def hero_yaw(res: dict) -> float:
    return float(res["shots"]["shots"][0]["yaw"])


def save(res: dict, out_dir, title: str, files_extra: dict | None = None) -> dict:
    """Plan with view cones, shots (re-rendered + raw), viewer page, dollhouse GLB, summary JSON."""
    from . import pano_viewer as pv
    from . import textured_mesh as tm
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shots = res["shots"]["shots"]
    files = p3v.save(res, out, title, tag="v5", views=ds.plan_views(shots), export_kw={"mipmaps": True, "jpeg_quality": 92})
    sd = out / "shots"
    sd.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in shots:
        p, pr = sd / f"{s['letter']}.jpg", sd / f"{s['letter']}_raw.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        cv2.imwrite(str(pr), cv2.cvtColor(s["raw"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        files[f"shot_{s['letter']}"], files[f"raw_{s['letter']}"] = str(p), str(pr)
        meta.append({"letter": s["letter"], "kind": s["kind"], "yaw_B_deg": s["yaw"], "hfov_deg": s["fov"],
                     "valid": s.get("valid"), "caption": sv.caption(s), "file": p.name, "raw": pr.name,
                     "measures": {k: v for k, v in s["measures"].items() if k not in ("inst_ids", "v_fail")},
                     "fail": s["measures"].get("v_fail", [])})
    (out / "shots.json").write_text(json.dumps({"shots": meta, "camera_height_m": res["box"]["camera_height"],
                                                "n_valid_yaws": res["shots"].get("n_valid_yaws")}, indent=1, default=float))
    files["inside_html"] = pv.save_viewer(out / "inside_v5.html", res["hires"], res["box"], hero_yaw(res),
                                          shots[0]["fov"], title)
    if res.get("doll_v5") is not None:
        files["dollhouse_glb"] = str(out / "dollhouse_v5.glb")
        tm.export_glb(res["doll_v5"], files["dollhouse_glb"], double_sided=False, mipmaps=True, jpeg_quality=92)
    files.update(files_extra or {})
    return files
