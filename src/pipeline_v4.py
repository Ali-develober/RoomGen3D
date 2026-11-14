"""v4 360° room pipeline (PLAN_V4): the v3 room plus readable design shots (and, in later milestones, a
sharper panorama, a sharper 3D view and real furniture shapes).

    [1]-[5]  as v3 (src/pipeline_v3.py): candidates, instances, room box, solid furniture, openings
    [5b]     sharper panorama (src/pano_refine.py, milestone 2): 26 perspective views of the texture, optional
             img2img detail pass, Real-ESRGAN x2 per view, multi-band blend to 4096 x 2048 (`hires`)
    [5c]     sharp 3D (src/room_v4.py, milestone 3; when the 4096 panorama exists): inside mesh with 5 mm shell
             texels, silhouette band, stride-1 furniture surfaces with 3D Taubin smoothing; dollhouse with the
             same textures; packed atlas; mipmapped GLB export
    [5d]     furniture objects (src/objects3d.py, milestone 4; optional, `triposr` model + max_objects > 0):
             TripoSR per instance, largest first, render-and-compare yaw; below the development IoU threshold
             or for block classes the v3 block stays; uncertain (rejected) instances become RELIEF in the
             dollhouse; the objects are a vertex-coloured primitive of dollhouse_v4.glb
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
          shot_weights: dict | None = None, hires: np.ndarray | None = None, mesh_opts: dict | None = None,
          triposr=None, device=None, max_objects: int = 0, cache_dir=None) -> dict:
    res = p3v.build(dist, texture, ana, id2label, furniture_params)
    res["hires"] = hires
    if hires is not None:
        from . import room_v4 as r4
        blocks, relief = None, relief_mask(res, dist, ana, id2label)       # relief: always (milestone 4)
        res["n_relief"] = relief_count(res, ana, id2label, relief)
        if triposr is not None and max_objects > 0:
            res["objects"] = add_objects(res, dist, hires, ana, id2label, triposr, device, max_objects, cache_dir)
            blocks = res["objects"]["blocks"]
        m = r4.build_meshes(dist, hires, res, ana, id2label, mesh_opts, block_solids=blocks, relief=relief)
        res["mesh"], res["dollhouse"], res["mesh_opts"] = m["inside"], m["dollhouse"], m["opts"]
    res["shots"] = ds.design_shots(texture if hires is None else hires, res["box"], ana["class_map"], ana["inst_map"], ana["instances"],
                                   id2label, shot_weights, dist=dist)
    return res


def selected_objects_params() -> dict:
    p = Path(__file__).resolve().parents[1] / "results/tables/objects_selection.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"iou_min": 0.5, "block_classes": []}


def add_objects(res: dict, dist, hires, ana, id2label, model, device, max_objects: int, cache_dir=None) -> dict:
    """[5d] generated objects for the largest solids, blocks for the rest, relief mask for uncertain instances."""
    from . import furniture as fu
    from . import objects3d as o3
    inst_map, _ = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)
    sel = selected_objects_params()
    out = o3.build_objects(model, device, hires, dist, inst_map, res["solids"], res["box"], max_objects=max_objects,
                           iou_min=sel["iou_min"], block_classes=sel["block_classes"],
                           cache_dir=None if cache_dir is None else Path(cache_dir))
    relief = relief_mask(res, dist, ana, id2label)
    Q = res["box"]["Q"]
    extra = None
    if out["objects"]:
        V = [o["VB"] @ Q.T for o in out["objects"]]
        off = np.cumsum([0] + [len(v) for v in V])[:-1]
        extra = {"vertices": np.concatenate(V), "faces": np.concatenate([o["faces"] + k for o, k in zip(out["objects"], off)]),
                 "colors": np.concatenate([o["colors"] for o in out["objects"]])}
    out.update({"relief": relief, "extra": extra, "n_relief": relief_count(res, ana, id2label, relief)})
    return out


def _inst_map(ana, id2label):
    from . import furniture as fu
    return fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)[0]


def relief_mask(res: dict, dist, ana, id2label, min_px: int = 150) -> np.ndarray:
    """Furniture instances that became no solid (rejected / uncertain in v3: recall 0.38): shown as RELIEF -
    their observed surface - in the dollhouse instead of disappearing."""
    inst_map = _inst_map(ana, id2label)
    used = {int(s.get("instance", -1)) for s in res["solids"]}
    relief = np.zeros(inst_map.shape, bool)
    ids, cnt = np.unique(inst_map[inst_map >= 0], return_counts=True)
    for i, n in zip(ids, cnt):
        if int(i) not in used and n >= min_px:
            relief |= inst_map == i
    return relief & p3v.in_front_of_shell(dist, res["box"])


def relief_count(res, ana, id2label, relief) -> int:
    return int(len(np.unique(_inst_map(ana, id2label)[relief]))) if relief.any() else 0


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
    if res.get("objects") and res["objects"]["extra"] is not None:                 # dollhouse with the objects
        from . import textured_mesh as tm
        tm.export_glb(res["dollhouse"], files["dollhouse_glb"], double_sided=False, mipmaps=o["mipmaps"],
                      jpeg_quality=o["jpeg_q"], extra=res["objects"]["extra"])
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
