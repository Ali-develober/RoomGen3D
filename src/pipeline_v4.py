"""v4 360° room pipeline (PLAN_V4): the v3 room plus readable design shots (and, in later milestones, a
sharper panorama, a sharper 3D view and real furniture shapes).

    [1]-[5]  as v3 (src/pipeline_v3.py): candidates, instances, room box, solid furniture, openings
    [6]      design shots (src/design_shots.py): hero view(s) + four wall views, perspective, level,
             1024 x 768, cut from the super-resolved panorama texture
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


def build(dist: np.ndarray, texture: np.ndarray, ana: dict, id2label: dict, furniture_params: dict | None = None,
          shot_weights: dict | None = None) -> dict:
    res = p3v.build(dist, texture, ana, id2label, furniture_params)
    res["shots"] = ds.design_shots(texture, res["box"], ana["class_map"], ana["inst_map"], ana["instances"],
                                   id2label, shot_weights, dist=dist)
    return res


def save(res: dict, out_dir, title: str | None = None, pano: np.ndarray | None = None) -> dict:
    out = Path(out_dir)
    shots = res["shots"]["shots"]
    files = p3v.save(res, out, title, tag="v4", views=ds.plan_views(shots))
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
    if pano is not None:
        files["pano_sphere_glb"] = str(out / "pano_sphere.glb")
        vw.pano_sphere_glb(pano, files["pano_sphere_glb"])
    return files


def gallery(res: dict) -> list[tuple[np.ndarray, str]]:
    """(image, caption) pairs, hero first, for the app's gallery."""
    return [(s["image"], ds.caption(s)) for s in res["shots"]["shots"]]
