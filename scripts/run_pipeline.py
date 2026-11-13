"""End-to-end run on the 6 demo prompts: prompt -> 2D image -> gated preprocessing -> Canny +
segmentation -> metric depth -> refinement -> point cloud -> textured mesh (see src/pipeline.py).

One model on the GPU at a time: all prompts go through SD + LoRA, then the segmenter, then the
depth model; the 3D stages run on the CPU. Per-prompt time is the sum of its stage times; model
loading is reported separately (it is paid once, not per prompt).

Outputs:
  outputs/pipeline/<prompt>_s<seed>/   image.png, depth.png, overlay.png, depth_m.npy,
                                       pointcloud.ply, mesh.glb, mesh.obj/.mtl/.png  (not in git)
  results/tables/pipeline_timing.csv   seconds per stage and prompt, peak VRAM per GPU stage
  results/figures/pipeline_3d_results.png        image | edges | segmentation | depth | point cloud | mesh
  results/figures/pointcloud_refinement.png      point cloud without / with refinement + cleaning
Usage: python scripts/run_pipeline.py [--limit 2]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402
from src import textured_mesh as tm  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "pipeline"
STAGES = ["generate", "preprocess", "analysis", "depth", "refine", "pointcloud", "mesh", "floorplan", "export"]


def gpu_stage(name: str, loader, load_s: dict, vram: dict):
    """Load one model, timing the load; returns it. Call end_gpu_stage afterwards."""
    ev.release_memory()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = loader()
    torch.cuda.synchronize()
    load_s[name] = time.perf_counter() - t0
    return model


def fig_results(results: list, prompts: list, path: Path) -> None:
    n = len(results)
    titles = ["Generated image", "Canny edges", "Segmentation", "Depth (light = near)",
              "Point cloud (side, orthographic)", "Textured mesh (camera turned 25°)"]
    fig, axes = plt.subplots(n, 6, figsize=(17.5, 2.95 * n), squeeze=False)
    for r, res in enumerate(results):
        panels = [res["image"], res["edge_overlay"], res["seg_overlay"], pl.depth_image(res["depth"]),
                  r3.render_pcd(res["pcd"], "side"),
                  tm.render_view(*tm.textured_samples(res["mesh"]), -25, pivot_z=res["mesh"]["target_z"])]
        for c, p in enumerate(panels):
            axes[r, c].imshow(p)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10.5)
        axes[r, 0].set_ylabel(f"{prompts[r][0]}\nseed {prompts[r][1]}".replace(" style ", " style\n"), fontsize=8.5)
        d = res["depth"]
        axes[r, 3].set_xlabel(f"{d.min():.1f} - {d.max():.1f} m", fontsize=8.5)
        axes[r, 4].set_xlabel(f"{res['stats']['points']:,} points", fontsize=8.5)
        axes[r, 5].set_xlabel(f"{res['stats']['triangles']:,} triangles", fontsize=8.5)
    handles = [Patch(color=np.array(c) / 255, label=g) for g, c in an.GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False, title="Segmentation groups")
    fig.suptitle("RoomGen3D end to end: prompt → 2D image → edges / segmentation → depth → point cloud → mesh "
                 "(white = empty)", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_refinement(results: list, prompts: list, path: Path, rows=(0, 1, 3)) -> None:
    """Point cloud straight from the raw depth (points that cleaning removes in red) vs. the pipeline output."""
    rows = [r for r in rows if r < len(results)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.75 * len(rows)), squeeze=False)
    titles = ["Raw depth, no cleaning (side)", "Refined + cleaned (side)", "Raw depth, no cleaning (top)",
              "Refined + cleaned (top)"]
    for i, r in enumerate(rows):
        res = results[r]
        raw, pre = res["depth_raw"], res["pre"]
        h, w = raw.shape
        k = r3.intrinsics(w, h)
        pts = r3.backproject(raw, *k).reshape(-1, 3)
        cols = pre.reshape(-1, 3).astype(np.float64) / 255
        fly = cv2.dilate(r3.flying_pixel_mask(raw, np.ones_like(raw, bool), *k).astype(np.uint8),
                         np.ones((3, 3), np.uint8)).ravel() > 0
        _, kept = r3.to_o3d(pts, cols).remove_statistical_outlier(r3.OUTLIER_NEIGHBORS, r3.OUTLIER_STD_RATIO)
        bad = fly.copy()
        bad[np.setdiff1d(np.arange(len(pts)), np.asarray(kept))] = True
        cols[bad] = (1.0, 0.0, 0.0)
        order = np.argsort(bad)                       # paint the red points last so they stay visible
        panels = [r3.render_points(pts[order], cols[order], "side", splat=2), r3.render_pcd(res["pcd"], "side"),
                  r3.render_points(pts[order], cols[order], "top", splat=2), r3.render_pcd(res["pcd"], "top")]
        for c, p in enumerate(panels):
            axes[i, c].imshow(p)
            axes[i, c].set_xticks([])
            axes[i, c].set_yticks([])
            if i == 0:
                axes[i, c].set_title(titles[c], fontsize=10.5)
        axes[i, 0].set_ylabel(prompts[r][0].replace(" style ", " style\n"), fontsize=9)
        s = res["stats"]
        axes[i, 0].set_xlabel(f"{len(pts):,} points, {int(bad.sum()):,} flagged (red)", fontsize=8.5)
        axes[i, 1].set_xlabel(f"{s['points']:,} points  (-{s['flying_removed']:,} flying, "
                              f"-{s['outliers_removed']:,} outliers)", fontsize=8.5)
    fig.suptitle("Point cloud without and with depth refinement + flying-pixel and statistical outlier removal "
                 "(red = points the cleaning removes)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (test; nothing written to results/)")
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    prompts = pl.DEMO_PROMPTS[: args.limit] if args.limit else pl.DEMO_PROMPTS
    watches = [pl.Stopwatch() for _ in prompts]
    results = [{} for _ in prompts]
    load_s, vram = {}, {}

    # ---- [1] SD + LoRA ------------------------------------------------------------
    pipe = gpu_stage("generate", lambda: g2.load_pipeline(g2.DEFAULT_LORA, device), load_s, vram)
    g2.generate(pipe, "warm-up", 0, steps=2)                 # first call compiles kernels; not a prompt's cost
    for (prompt, seed), w, res in zip(prompts, watches, results):
        with w.stage("generate"):
            res["image"] = g2.generate(pipe, prompt, seed)
    vram["generate"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del pipe

    # ---- [2] gated preprocessing + [3] Canny / segmentation ------------------------
    seg = gpu_stage("analysis", lambda: an.load_segmenter(device), load_s, vram)
    an.segment_classes(*seg, results[0]["image"], device)    # warm-up
    for w, res in zip(watches, results):
        with w.stage("preprocess"):
            res["pre"], res["info"] = pl.preprocess(res["image"])
        with w.stage("analysis"):
            res.update(pl.analyse(res["pre"], *seg, device))
    vram["analysis"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del seg

    # ---- [4] metric depth + refinement ---------------------------------------------
    dm = gpu_stage("depth", lambda: dp.load_depth_model(pl.DEPTH_MODEL_3D, device), load_s, vram)
    dp.predict_depth(*dm, results[0]["pre"], device)         # warm-up
    for w, res in zip(watches, results):
        with w.stage("depth"):
            res.update(pl.estimate_depth(res["pre"], *dm, device, refine=False))
        with w.stage("refine"):
            res["depth"] = dp.refine_metric_depth(res["depth_raw"], res["pre"])
    vram["depth"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del dm
    ev.release_memory()

    # ---- [5] point cloud + [6] mesh (CPU) ------------------------------------------
    for (prompt, seed), w, res in zip(prompts, watches, results):
        with w.stage("pointcloud"):
            res.update(r3.reconstruct(res["depth"], res["pre"], mesh=False))
        with w.stage("mesh"):
            res["mesh"] = pl.build_mesh(res["depth"], res["pre"], res)
            res["stats"].update(res["mesh"]["stats"])
        with w.stage("export"):
            res["files"] = pl.save_outputs(res, OUT / pl.slug(prompt, seed))

    # ---- timing table --------------------------------------------------------------
    rows = []
    for (prompt, seed), w, res in zip(prompts, watches, results):
        row = {"prompt": prompt, "seed": seed, **{s: w.seconds.get(s, 0.0) for s in STAGES}}
        row["total_s"] = sum(row[s] for s in STAGES)
        row.update({"noise_est": res["info"]["noise_est"], "denoised": res["info"]["denoised"], **res["stats"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    num = STAGES + ["total_s"]
    extra = pd.DataFrame([{"prompt": "MEAN", **df[num].mean().to_dict()},
                          {"prompt": "MODEL LOAD (once)", **load_s, "total_s": sum(load_s.values())},
                          {"prompt": "PEAK VRAM MB", **vram, "total_s": max(vram.values())}])
    table = pd.concat([df, extra], ignore_index=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table[["prompt"] + num].round(2).to_string(index=False))
    print(f"\nmean per prompt: {df['total_s'].mean():.1f} s (max {df['total_s'].max():.1f} s); "
          f"model loading once: {sum(load_s.values()):.1f} s; peak VRAM {max(vram.values()):.0f} MB")
    if args.limit:
        return
    table.to_csv(ROOT / "results/tables/pipeline_timing.csv", index=False, float_format="%.3f")
    fig_results(results, prompts, ROOT / "results/figures/pipeline_3d_results.png")
    fig_refinement(results, prompts, ROOT / "results/figures/pointcloud_refinement.png")
    print("saved table and figures; 3D files in", OUT)


if __name__ == "__main__":
    main()
