"""360 degree room mode: prompt -> panorama -> distance map -> full-room mesh (+ floor plan).

    [1] panorama        SD 2.1-base panorama model, circular padding, 1024 x 512 (generate_pano)
    [2] super-res       Real-ESRGAN x2 with wrap padding -> 2048 x 1024 texture (generate_pano)
    [3] depth           DAP + roll-and-blend wrap fix -> metric distance map (pano_depth)
    [4] segmentation    SegFormer on 14 perspective views, labels projected back (pano_depth)
    [5] mesh            gravity alignment, room box, wrap-around textured mesh, dollhouse (room_mesh)
    [6] floor plan      360 degree plan of the same point cloud (floorplan, via room_mesh.plan_360)

Choices: README section 8, milestones 2-4. One GPU model at a time in the scripts; the models are
small enough (panorama SD 2.6 GB, DAP 2.3 GB fp32, SegFormer 0.3 GB, Real-ESRGAN 0.1 GB) that the
demo keeps them loaded together (measured in scripts/run_room360.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import floorplan as fp
from . import generate_pano as gp
from . import pano_depth as pdp
from . import room_mesh as rm
from . import textured_mesh as tm

PANO_SEED_OFFSET = 0


def generate(pipe, prompt: str, seed: int, steps: int = gp.PANO_STEPS, guidance: float = gp.PANO_GUIDANCE) -> np.ndarray:
    """[1] 1024 x 512 panorama with the selected generator."""
    return gp.generate_pano(pipe, prompt, seed, circular=True, steps=steps, guidance=guidance,
                            prompt=gp.ALT_TEMPLATE.format(room=prompt))


def super_resolve(net, pano: np.ndarray) -> np.ndarray:
    """[2] 2048 x 1024 texture."""
    return gp.esrgan_upscale(net, pano)


def distance(dap, pano: np.ndarray, device) -> np.ndarray:
    """[3] metric distance map (m), same size as the panorama."""
    return pdp.dap_depth_wrap(dap, pano, device)


def segment(seg, pano: np.ndarray, device) -> np.ndarray:
    """[4] ADE20K class map of the panorama."""
    return pdp.segment_pano(*seg, pano, device)


def build(dist: np.ndarray, texture: np.ndarray, class_map: np.ndarray, id2label: dict) -> dict:
    """[5] + [6] full-room mesh, dollhouse variant and 360 degree floor plan."""
    mesh = rm.build_room_mesh(dist, texture, class_map, id2label)
    return {"mesh": mesh, "dollhouse": rm.dollhouse(mesh), "plan": mesh["plan"]}


def save(res: dict, out_dir: str | Path, pano: np.ndarray | None = None, texture: np.ndarray | None = None,
         dist: np.ndarray | None = None, class_map: np.ndarray | None = None, plan_title: str | None = None) -> dict:
    """Write room.glb, dollhouse.glb, floorplan.png/.svg (+ the inputs if given); returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"room_glb": out / "room.glb", "dollhouse_glb": out / "dollhouse.glb",
             "plan_png": out / "floorplan.png", "plan_svg": out / "floorplan.svg"}
    tm.export_glb(res["mesh"], files["room_glb"])
    tm.export_glb(res["dollhouse"], files["dollhouse_glb"])
    res["plan_image"] = fp.draw_floorplan(res["plan"], files["plan_png"], files["plan_svg"], title=plan_title)
    for name, arr in (("pano", pano), ("texture", texture)):
        if arr is not None:
            files[name] = out / f"{name}.png"
            cv2.imwrite(str(files[name]), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if dist is not None:
        files["distance"] = out / "distance.npy"
        np.save(files["distance"], dist.astype(np.float32))
    if class_map is not None:
        files["class_map"] = out / "class_map.png"
        cv2.imwrite(str(files["class_map"]), class_map.astype(np.uint8))
    return {k: str(v) for k, v in files.items()}
