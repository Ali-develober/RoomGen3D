"""v4 360° room pipeline (PLAN_V4): the v3 room plus readable design shots (and, in later milestones, a
sharper panorama, a sharper 3D view and real furniture shapes).

    [1]-[5]  as v3 (src/pipeline_v3.py): candidates, instances, room box, solid furniture, openings
    [5b]     sharper panorama (src/pano_refine.py, milestone 2): 26 perspective views of the texture, optional
             img2img detail pass, Real-ESRGAN x2 per view, multi-band blend to 4096 x 2048 (`hires`)
    [5c]     sharp 3D (src/room_v4.py, milestone 3; when the 4096 panorama exists): inside mesh with 5 mm shell
             texels, silhouette band, stride-1 furniture surfaces with 3D Taubin smoothing; dollhouse with the
             same textures; packed atlas; mipmapped GLB export
    [6]      design shots (src/design_shots.py): hero view(s) + four wall views, perspective, level,
             1024 x 768, cut from the sharpest panorama available (hires if given, else the texture)
    [7]      outputs: v3 files with suffix _v4, the floor plan with the shots' camera and view cones
             (letters A-D, H), shots/<letter>.png, shots.json; the panorama as an inward-facing sphere
             (pano_sphere.glb) for the interactive 360° viewer

v3 stays callable unchanged (pipeline_v3.build / save) for before / after comparisons on the same seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import design_shots as ds
from . import pipeline_v3 as p3v
from . import viewers as vw

generate_candidates = p3v.generate_candidates
analyse = p3v.analyse


def sharpen(texture: np.ndarray, esrgan, i2i=None, prompt: str = "", strength: float = 0.0, seed: int = 0):
    """[5b] 4096 x 2048 panorama from the 2048 x 1024 texture (see src/pano_refine.py). Returns (pano, info)."""
    from . import pano_refine as pr
    pano, info = pr.refine(texture, esrgan, i2i, prompt, strength, seed)
    info.pop("views", None)
    return pano, info


def build(dist: np.ndarray, texture: np.ndarray, ana: dict, id2label: dict, furniture_params: dict | None = None,
          shot_weights: dict | None = None, hires: np.ndarray | None = None, mesh_opts: dict | None = None) -> dict:
    res = p3v.build(dist, texture, ana, id2label, furniture_params)
    res["hires"] = hires
    if hires is not None:
        from . import room_v4 as r4
        m = r4.build_meshes(dist, hires, res, ana, id2label, mesh_opts)
        res["mesh"], res["dollhouse"], res["mesh_opts"] = m["inside"], m["dollhouse"], m["opts"]
    res["shots"] = ds.design_shots(texture if hires is None else hires, res["box"], ana["class_map"], ana["inst_map"], ana["instances"],
                                   id2label, shot_weights, dist=dist)
    return res


def save(res: dict, out_dir, title: str | None = None, pano: np.ndarray | None = None) -> dict:
    out = Path(out_dir)
    shots = res["shots"]["shots"]
    o = res.get("mesh_opts")
    files = p3v.save(res, out, title, tag="v4", views=ds.plan_views(shots),
                     export_kw={"mipmaps": o["mipmaps"], "jpeg_quality": o["jpeg_q"]} if o else None)
    sd = out / "shots"
    sd.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in shots:
        p = sd / f"{s['letter']}.png"
        cv2.imwrite(str(p), cv2.cvtColor(s["image"], cv2.COLOR_RGB2BGR))
        files[f"shot_{s['letter']}"] = str(p)
        meta.append({"letter": s["letter"], "kind": s["kind"], "yaw_B_deg": s["yaw"], "hfov_deg": s["fov"],
                     "caption": ds.caption(s), "file": p.name,
                     "measures": {k: v for k, v in s["measures"].items() if k != "inst_ids"}})
    (out / "shots.json").write_text(json.dumps({"shots": meta, "seconds": res["shots"]["seconds"],
                                                "camera_height_m": res["box"]["camera_height"]}, indent=1,
                                               default=float))
    if res.get("hires") is not None:
        files["pano_hires"] = str(out / "pano_4096.jpg")
        cv2.imwrite(files["pano_hires"], cv2.cvtColor(res["hires"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if pano is not None:
        files["pano_sphere_glb"] = str(out / "pano_sphere.glb")
        vw.pano_sphere_glb(pano, files["pano_sphere_glb"])
    return files


def hero_yaw_A(res: dict) -> float:
    """Longitude (deg) of the hero direction in the mesh frame A (for the inside viewer's opening camera)."""
    y = np.radians(res["shots"]["shots"][0]["yaw"])
    d = res["box"]["Q"] @ np.array([np.sin(y), 0.0, np.cos(y)])
    return float(np.degrees(np.arctan2(d[0], d[2])))


def gallery(res: dict) -> list[tuple[np.ndarray, str]]:
    """(image, caption) pairs, hero first, for the app's gallery."""
    return [(s["image"], ds.caption(s)) for s in res["shots"]["shots"]]
