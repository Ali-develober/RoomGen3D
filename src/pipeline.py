"""End-to-end pipeline: prompt -> 2D image -> preprocessing -> analysis -> depth -> 3D.

    [1] generate       SD v1.5 + LoRA (generate_2d)
    [2] preprocess     noise-gated denoising (preprocessing.preprocess_gated)
    [3] analyse        Canny edges + SegFormer room masks (analysis)
    [4] depth          metric depth in metres (depth) + depth refinement
    [5] point cloud    back-projection, flying-pixel + statistical outlier removal (reconstruct_3d)
    [6] mesh           normals, Poisson reconstruction, density trim, export

Each stage is a small function so the script (scripts/run_pipeline.py) and the demo (app/app.py)
run exactly the same code. `Stopwatch` records the time of every stage.
"""

import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from . import analysis as an
from . import depth as dp
from . import reconstruct_3d as r3
from .preprocessing import estimate_noise_sigma, preprocess_gated

DEPTH_MODEL_3D = "depth_anything_metric"    # depth model of the 3D stage (choice: README section 8, "3D")

# Fixed demo prompts (LoRA caption format) and seeds: chosen once, not picked after looking at results.
DEMO_PROMPTS = [
    ("a contemporary style living room", 11),
    ("a coastal style bedroom", 22),
    ("a rustic style kitchen", 33),
    ("a mid-century modern style dining room", 44),
    ("an art deco style bathroom", 55),
    ("a mediterranean style living room", 66),
]


def slug(prompt: str, seed: int) -> str:
    """File-system friendly name of a (prompt, seed) pair."""
    keep = "".join(c if c.isalnum() else "_" for c in prompt.lower()).strip("_")
    return f"{'_'.join(w for w in keep.split('_') if w)[:60]}_s{seed}"


class Stopwatch:
    """Collects wall-clock seconds per named stage (GPU work is synchronised before reading the clock)."""

    def __init__(self):
        self.seconds = {}

    @contextmanager
    def stage(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """[2] Noise-gated denoising. Returns (image for the next stages, {"noise_est", "denoised"})."""
    est = estimate_noise_sigma(img)
    out = preprocess_gated(img)
    return out, {"noise_est": est, "denoised": out is not img}


def analyse(img: np.ndarray, seg_model, seg_proc, device) -> dict:
    """[3] Canny edges + room segmentation. Returns edges, cleaned masks and the two overlays."""
    edges = an.canny_edges(img)
    masks = an.segment_room(seg_model, seg_proc, img, device, clean=True)
    return {"edges": edges, "masks": masks, "edge_overlay": an.overlay_edges(img, edges),
            "seg_overlay": an.overlay_masks(img, masks)}


def estimate_depth(img: np.ndarray, model, proc, device, model_name: str = DEPTH_MODEL_3D,
                   refine: bool = True) -> dict:
    """[4] Depth in metres (+ refinement). A relative model is mapped to pseudo-metres (see reconstruct_3d)."""
    if model_name in dp.METRIC_MODELS:
        raw = dp.predict_depth(model, proc, img, device)
        depth = dp.refine_metric_depth(raw, img) if refine else raw
    else:
        disp = dp.predict_disparity(model, proc, img, device)
        raw = r3.disparity_to_depth(disp)
        depth = r3.disparity_to_depth(dp.refine_depth(disp, img)) if refine else raw
    return {"depth_raw": raw, "depth": depth}


def depth_image(depth: np.ndarray) -> np.ndarray:
    """Colour-coded depth map for display (light = near, dark = far), RGB uint8."""
    lo, hi = np.percentile(depth, [1, 99])
    d = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def combined_overlay(img: np.ndarray, edges: np.ndarray, masks: dict) -> np.ndarray:
    """Segmentation colours with the Canny edges drawn on top (one picture for the demo)."""
    out = an.overlay_masks(img, masks)
    out[cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (255, 255, 255)
    return out


def image_to_3d(img: np.ndarray, seg, depth_model, device, watch: Stopwatch | None = None,
                out_dir: str | Path | None = None, mesh: bool = True) -> dict:
    """Stages [2]-[6] for one RGB uint8 image with the models already loaded.

    seg = (model, processor) of the segmenter, depth_model = (model, processor) of DEPTH_MODEL_3D.
    With out_dir the files image.png, depth.png, overlay.png, pointcloud.ply, mesh.glb and mesh.obj
    are written there and their paths returned under "files".
    """
    watch = watch or Stopwatch()
    with watch.stage("preprocess"):
        pre, info = preprocess(img)
    with watch.stage("analysis"):
        ana = analyse(pre, *seg, device)
    with watch.stage("depth"):
        dep = estimate_depth(pre, *depth_model, device, refine=False)
    with watch.stage("refine"):
        dep["depth"] = dp.refine_metric_depth(dep["depth_raw"], pre)
    with watch.stage("pointcloud"):
        rec = r3.reconstruct(dep["depth"], pre, mesh=False)
    if mesh:
        with watch.stage("mesh"):
            r3.estimate_normals(rec["pcd"])
            rec["mesh"] = r3.poisson_mesh(rec["pcd"])
            rec["stats"]["vertices"], rec["stats"]["triangles"] = len(rec["mesh"].vertices), len(rec["mesh"].triangles)
    out = {"image": img, "pre": pre, "info": info, **ana, **dep, **rec, "seconds": watch.seconds, "files": {}}
    if out_dir is not None:
        with watch.stage("export"):
            out["files"] = save_outputs(out, out_dir)
    return out


def save_outputs(res: dict, out_dir: str | Path) -> dict:
    """Write the pictures and 3D files of one result; returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"image": out_dir / "image.png", "depth": out_dir / "depth.png", "overlay": out_dir / "overlay.png",
             "pointcloud": out_dir / "pointcloud.ply"}
    cv2.imwrite(str(files["image"]), cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(files["depth"]), cv2.cvtColor(depth_image(res["depth"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(files["overlay"]), cv2.cvtColor(combined_overlay(res["pre"], res["edges"], res["masks"]),
                                                    cv2.COLOR_RGB2BGR))
    np.save(out_dir / "depth_m.npy", res["depth"].astype(np.float32))
    r3.export_pointcloud(res["pcd"], files["pointcloud"])
    if res.get("mesh") is not None:
        files["mesh_glb"], files["mesh_obj"] = out_dir / "mesh.glb", out_dir / "mesh.obj"
        r3.export_mesh(res["mesh"], files["mesh_glb"])
        r3.export_mesh(res["mesh"], files["mesh_obj"])
    return {k: str(v) for k, v in files.items()}
