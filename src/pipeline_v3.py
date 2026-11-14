"""v3 360 degree room pipeline (PLAN_V3): panorama -> clean box room with solid furniture and a readable plan.

    [1] panorama, super-resolution, DAP distance, SegFormer classes   as in v2 (src/pipeline_360.py)
    [2] instances      Mask2Former panoptic on 13 views, merged across views; hybrid class map (src/instances.py)
    [3] room box       dominant directions, per-side wall fit, textured exact shell (src/room_box.py)
    [4] furniture      cleaned, prior-checked, wall-aligned solids (src/furniture.py)
    [5] openings       doors / windows on the box walls (src/plan_v3.openings_on_box)
    [6] outputs        inside mesh (shell + observed furniture surfaces), dollhouse mesh (shell without ceiling +
                       furniture solids), floor plan PNG / SVG / DXF, top view

The v2 pipeline stays callable (src/pipeline_360.py) for before / after comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import boxiness as bx
from . import furniture as fu
from . import generate_pano as gp
from . import instances as ins
from . import pano_depth as pdp
from . import plan_v3 as pv
from . import room_box as rb
from . import textured_mesh as tm
from . import viewers as vw


ONE_ROOM = True          # one-room prompt template (milestone 5; see results/tables/oneroom_decision.json)


def generate_candidates(pipe, dap, seg, prompt: str, seed: int, k: int, device, one_room: bool = ONE_ROOM,
                        steps: int = gp.PANO_STEPS, guidance: float = gp.PANO_GUIDANCE) -> dict:
    """[1] k panoramas (seeds seed .. seed + k - 1), each with DAP distance, SegFormer classes and its
    boxiness score; returns the best one and all scores."""
    tpl, neg = (gp.ONE_ROOM_TEMPLATE, gp.ONE_ROOM_NEGATIVE) if one_room else (gp.ALT_TEMPLATE, gp.PANO_NEGATIVE)
    id2label = {int(i): v for i, v in seg[0].config.id2label.items()}
    best, cands = None, []
    for i in range(max(1, int(k))):
        pano = gp.generate_pano(pipe, prompt, seed + i, steps=steps, guidance=guidance,
                                prompt=tpl.format(room=prompt), negative=neg)
        dist = pdp.dap_depth_wrap(dap, pano, device)
        cm = pdp.segment_pano(*seg, pano, device)
        sc = bx.score(dist, cm, id2label)
        cands.append({"seed": seed + i, **sc})
        if best is None or sc["score"] > best["score"]["score"]:
            best = {"pano": pano, "dist": dist, "class_map": cm, "score": sc, "seed": seed + i}
    best["candidates"] = cands
    return best


def analyse(panoptic, pano: np.ndarray, seg_class_map: np.ndarray, id2label: dict, device) -> dict:
    """[2] instances + hybrid class map at the grid of the distance map (seg_class_map's size)."""
    h, w = seg_class_map.shape
    p = cv2.resize(pano, (w, h), interpolation=cv2.INTER_AREA) if pano.shape[:2] != (h, w) else pano
    v3 = ins.merge_views(ins.run_views(*panoptic, p, device), id2label, w, h)
    return {"class_map": ins.hybrid_class_map(seg_class_map, v3), "inst_map": v3["inst_map"], "instances": v3["instances"]}


def in_front_of_shell(dist: np.ndarray, box: dict, rel: float = 0.04, abs_: float = 0.05) -> np.ndarray:
    """Pixels whose observed surface lies clearly in front of the box shell (furniture, but also a real wall
    that stands in front of the fitted one, alcoves, curtains): their observed surface is kept in the inside
    mesh so that the view from the centre reproduces the panorama."""
    from . import equirect as eq
    H, W = dist.shape
    dirs = (eq.pixel_dirs(W, H) @ box["R"].T) @ box["Q"]
    return dist < bx.box_exit(dirs, box) * (1 - rel) - abs_


def build(dist: np.ndarray, texture: np.ndarray, ana: dict, id2label: dict, furniture_params: dict | None = None) -> dict:
    """[3]-[5] box, solids, openings, meshes."""
    cm = ana["class_map"]
    box = rb.fit_box(dist, cm, id2label)
    solids, counts = fu.build_solids(dist, ana["inst_map"], ana["instances"], box, id2label, furniture_params, cm)
    inst_map = ana["inst_map"]
    if (furniture_params or fu.DEFAULT_PARAMS).get("leftovers", fu.DEFAULT_PARAMS["leftovers"]):
        inst_map, _ = fu.add_leftovers(ana["inst_map"], ana["instances"], cm, id2label)
    shell = rb.build_shell(box, texture, dist)
    furn = fu.solids_part(solids, box, dist, inst_map, texture)
    # inside view: shell + the OBSERVED surfaces in front of it (from the centre it reproduces the panorama; the
    # closed shell behind them leaves no gaps); dollhouse: shell without ceiling + furniture solids
    seen = rb.observed_part(dist, texture, box["R"], in_front_of_shell(dist, box))
    mesh = rb.combine(shell["parts"] + ([seen] if len(seen["faces"]) else []))
    mesh.update({"box": box, "camera_height": box["camera_height"]})
    parts = shell["parts"] + ([furn] if len(furn["faces"]) else [])
    doll = rb.combine([p for p in parts if p["kind"] != "ceiling"])
    doll.update({"box": box, "camera_height": box["camera_height"]})
    openings = pv.openings_on_box(dist, cm, id2label, box)
    return {"box": box, "solids": solids, "counts": counts, "openings": openings, "mesh": mesh, "dollhouse": doll,
            "shell_stats": shell["stats"]}


def save(res: dict, out_dir, title: str | None = None, tag: str = "v3", views: list[dict] | None = None,
         export_kw: dict | None = None) -> dict:
    """[6] room_v3.glb, dollhouse_v3.glb (single-sided), floorplan_v3.png / .svg / .dxf, topdown_v3.png,
    summary_v3.json. tag / views: used by pipeline_v4 (file suffix; design-shot cones on the plan)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"room_glb": out / f"room_{tag}.glb", "dollhouse_glb": out / f"dollhouse_{tag}.glb",
             "plan_png": out / f"floorplan_{tag}.png", "plan_svg": out / f"floorplan_{tag}.svg",
             "plan_dxf": out / f"floorplan_{tag}.dxf"}
    tm.export_glb(res["mesh"], files["room_glb"], double_sided=False, **(export_kw or {}))
    tm.export_glb(res["dollhouse"], files["dollhouse_glb"], double_sided=False, **(export_kw or {}))
    files["topdown"] = out / f"topdown_{tag}.png"
    res["topdown"] = vw.topdown(res["dollhouse"], res["box"])
    cv2.imwrite(str(files["topdown"]), cv2.cvtColor(res["topdown"], cv2.COLOR_RGB2BGR))
    res["bg_color"] = vw.mean_color(res["mesh"])
    img, st = pv.draw_plan(res["box"], res["solids"], res["openings"], files["plan_png"], files["plan_svg"],
                           files["plan_dxf"], title=title, views=views)
    res["plan_image"], res["plan_stats"] = img, st
    b = res["box"]
    summary = {"width_m": b["width"], "depth_m": b["depth"], "area_m2": b["area"], "ceiling_m": b["ceiling_height"],
               "sides_detected": b["n_detected"], "furniture": [s["label"] for s in res["solids"]],
               "openings": [{k: o[k] for k in ("kind", "wall", "width")} for o in res["openings"]],
               "counts": res["counts"], "plan": st, "background": res["bg_color"]}
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    return {k: str(v) for k, v in files.items()}
