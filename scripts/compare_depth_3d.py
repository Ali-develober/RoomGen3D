"""Relative vs. metric Depth Anything for the 3D stage, on the 6 demo prompts (generated rooms).

A relative model gives inverse depth up to an unknown scale AND shift; to back-project it the
shift has to be guessed, which bends the geometry. The metric indoor model gives metres directly.
Generated rooms have no ground truth, so the two are compared (a) visually, side and top view of
the point cloud, and (b) with room-geometry checks from the segmentation masks: how flat the
floor and the main wall are and whether they meet at 90 degrees.

One model on the GPU at a time: SD (only if an image is missing) -> segmenter -> each depth model.
Outputs:
  results/figures/depth_relative_vs_metric_3d.png
  results/tables/depth_3d_model_choice.csv           per room and model, plus the mean rows
Usage: python scripts/compare_depth_3d.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "pipeline"
MODELS = ["depth_anything", "depth_anything_metric"]
SHORT = {"depth_anything": "Relative (Depth Anything)", "depth_anything_metric": "Metric (Depth Anything V2 indoor)"}


def demo_images(device) -> list[np.ndarray]:
    """The 6 demo images; generated with SD + LoRA and cached in outputs/pipeline/<slug>/image.png."""
    paths = [OUT / pl.slug(p, s) / "image.png" for p, s in pl.DEMO_PROMPTS]
    if not all(p.exists() for p in paths):
        pipe = g2.load_pipeline(g2.DEFAULT_LORA, device)
        for (prompt, seed), path in zip(pl.DEMO_PROMPTS, paths):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(g2.generate(pipe, prompt, seed), cv2.COLOR_RGB2BGR))
        del pipe
        ev.release_memory()
    return [pp.load_rgb(p) for p in paths]


def main():
    set_seed(42)
    device = torch.device("cuda")
    images = [pl.preprocess(im)[0] for im in demo_images(device)]

    seg = an.load_segmenter(device)
    masks = [an.segment_room(*seg, im, device) for im in images]
    del seg
    ev.release_memory()

    depths, rows = {}, []
    for name in MODELS:
        model = dp.load_depth_model(name, device)
        depths[name] = [pl.estimate_depth(im, *model, device, model_name=name)["depth"] for im in images]
        del model
        ev.release_memory()
        for (prompt, seed), d, m in zip(pl.DEMO_PROMPTS, depths[name], masks):
            stats = r3.room_plane_stats(d, m)
            stats.pop("wall_rms_pct")       # the wall mask spans several walls, so one plane does not describe it
            rows.append({"prompt": prompt, "seed": seed, "model": name, **stats,
                         "depth_min_m": float(d.min()), "depth_max_m": float(d.max())})
    df = pd.DataFrame(rows)
    df["right_angle_error"] = (90 - df["floor_wall_angle"]).abs()
    cols = ["floor_rms_pct", "floor_wall_angle", "right_angle_error", "floor_tilt"]
    mean = df.groupby("model", sort=False)[cols].mean().reset_index()
    mean.insert(0, "prompt", "MEAN")
    table = pd.concat([df, mean], ignore_index=True)
    table.to_csv(ROOT / "results/tables/depth_3d_model_choice.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(table[["prompt", "model"] + cols].round(2).to_string(index=False))

    # figure: image | relative side, top | metric side, top
    n = len(images)
    fig, axes = plt.subplots(n, 5, figsize=(15.5, 3.05 * n))
    titles = ["Generated image"] + [f"{SHORT[m]}\n{v} view" for m in MODELS for v in ("side", "top")]
    for r in range(n):
        panels = [images[r]]
        for name in MODELS:
            rec = r3.reconstruct(depths[name][r], images[r], mesh=False)
            panels += [r3.render_pcd(rec["pcd"], "side"), r3.render_pcd(rec["pcd"], "top")]
        for c, p in enumerate(panels):
            axes[r, c].imshow(p)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(pl.DEMO_PROMPTS[r][0].replace(" style ", " style\n"), fontsize=9)
        for k, name in enumerate(MODELS):
            s = df[(df["model"] == name) & (df["prompt"] == pl.DEMO_PROMPTS[r][0])].iloc[0]
            note = ("no floor visible" if np.isnan(s["floor_rms_pct"]) else
                    f"floor-wall angle {s['floor_wall_angle']:.0f}°, floor RMS {s['floor_rms_pct']:.1f}%")
            axes[r, 2 + 2 * k].set_xlabel(note, fontsize=8.5)
    fig.suptitle("Point clouds from relative vs. metric depth (orthographic renders; top view without the ceiling). "
                 f"Assumed camera: pinhole, {r3.DEFAULT_FOV_DEG:.0f}° horizontal FOV", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(ROOT / "results/figures/depth_relative_vs_metric_3d.png", dpi=110)
    plt.close(fig)
    print("saved figure and table")


if __name__ == "__main__":
    main()
