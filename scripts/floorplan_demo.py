"""Floor plans of the six demo rooms (outputs of scripts/run_pipeline.py).

Reads outputs/pipeline/<room>/image.png and depth_m.npy (refined metric depth), segments the image,
builds the plan with the pipeline's assumed camera (60 deg FOV) and writes
  outputs/pipeline/<room>/floorplan.png / .svg
  results/figures/floorplan_demo.png      image | floor plan for every room
  results/tables/floorplan_demo.csv       walls, furniture, openings, Manhattan, areas, heights, time
Usage: python scripts/floorplan_demo.py
"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402


def main():
    device = torch.device("cuda")
    seg = an.load_segmenter(device)
    rows, figs = [], []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs" / "pipeline" / pl.slug(prompt, seed)
        img = cv2.cvtColor(cv2.imread(str(d / "image.png")), cv2.COLOR_BGR2RGB)
        depth = np.load(d / "depth_m.npy")
        ana = {"class_map": an.segment_classes(*seg, img, device), "id2label": seg[0].config.id2label}
        t0 = time.perf_counter()
        plan = pl.build_floorplan(depth, ana)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        pic = fp.draw_floorplan(plan, d / "floorplan.png", d / "floorplan.svg")
        t_draw = time.perf_counter() - t0
        furn = ", ".join(sorted({f["label"] for f in plan["furniture"]}))
        rows.append({"prompt": prompt, "seed": seed, **fp.plan_summary(plan), "furniture_labels": furn,
                     "build_s": t_build, "draw_s": t_draw})
        figs.append((prompt, img, pic))
    del seg
    ev.release_memory()

    fig, axes = plt.subplots(2, 6, figsize=(24, 9.5), dpi=110, gridspec_kw={"height_ratios": [1, 1.35]})
    for j, (prompt, img, pic) in enumerate(figs):
        axes[0, j].imshow(img)
        axes[0, j].set_title(prompt, fontsize=10)
        axes[1, j].imshow(pic)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("Floor plans from one generated image (walls, doors/windows, furniture, approximate dimensions)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/floorplan_demo.png", facecolor="white")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/tables/floorplan_demo.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["furniture_labels"]).round(2).to_string(index=False))
    print(f"FOV assumed: {r3.DEFAULT_FOV_DEG} deg")


if __name__ == "__main__":
    main()
