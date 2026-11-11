"""Before/after figures for the depth stage (a few NYU test images; one model on the GPU at a time).

  results/figures/depth_preprocessing_steps.png   input | denoise | CLAHE | full, for clean / noisy / dark input
  results/figures/analysis_edges_segmentation.png RGB | Canny | raw segmentation | cleaned room masks
  results/figures/depth_refinement_steps.png      each refinement step on one prediction
  results/figures/depth_qualitative.png           RGB | ground truth | prediction | refined
Usage: python scripts/depth_figures.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import preprocessing as pp  # noqa: E402

FIG = ROOT / "results" / "figures"
ROWS = [60, 230, 400, 570]          # positions in the NYU test index used for the figures
DEPTH_CMAP = "magma_r"              # near = light, far = dark


def show(ax, img, title=None, **kw):
    ax.imshow(img, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9.5)


def fig_preprocessing(row):
    """Each preprocessing step on a clean, a noisy and a dark copy of the same image."""
    img = pp.load_rgb(row["rgb_path"])
    kinds = ["clean", "noise25", "low_light"]
    steps = [("Input", lambda x: x), ("Denoise (bilateral)", pp.denoise), ("CLAHE (L channel)", pp.clahe_lab),
             ("Full: denoise + CLAHE + unsharp", pp.preprocess_for_depth)]
    fig, axes = plt.subplots(len(kinds), len(steps), figsize=(3.6 * len(steps), 2.85 * len(kinds)))
    for r, kind in enumerate(kinds):
        x = ev.degrade(img, kind, int(row["mat_index"]))
        for c, (name, fn) in enumerate(steps):
            y = fn(x)
            note = f"noise est. {pp.estimate_noise_sigma(y):.1f}, mean {y.mean():.0f}"
            show(axes[r, c], y, f"{name}\n{note}" if r == 0 else note)
        axes[r, 0].set_ylabel(ev.DEGRADATION_LABELS[kind], fontsize=10)
    fig.suptitle("Depth-stage preprocessing, step by step (noise est. = estimated noise std, mean = brightness)",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "depth_preprocessing_steps.png", dpi=120)
    plt.close(fig)


def fig_analysis(rows, device):
    """Canny edges and room segmentation (before / after morphological cleaning)."""
    model, proc = an.load_segmenter(device)
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.4, 2.85 * len(rows)))
    titles = ["RGB", "Canny edges", "Segmentation (raw masks)", "After opening + closing"]
    for r, (_, row) in enumerate(rows.iterrows()):
        img = pp.load_rgb(row["rgb_path"])
        cls = an.segment_classes(model, proc, img, device)
        raw = an.room_masks(cls, model.config.id2label, clean=False)
        clean = an.room_masks(cls, model.config.id2label, clean=True)
        panels = [img, an.overlay_edges(img, an.canny_edges(img)), an.overlay_masks(img, raw),
                  an.overlay_masks(img, clean)]
        for c, p in enumerate(panels):
            show(axes[r, c], p, titles[c] if r == 0 else None)
    handles = [Patch(color=np.array(c) / 255, label=g) for g, c in an.GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("Image analysis: Canny edges and SegFormer (ADE20K) room masks", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "analysis_edges_segmentation.png", dpi=120)
    plt.close(fig)
    del model, proc
    ev.release_memory()


def fig_depth(rows, device):
    """Refinement steps on one image + RGB | GT | prediction | refined for a few images."""
    model, proc = dp.load_depth_model("depth_anything", device)

    # --- refinement steps (first image, noisy input so the effect is visible) ---
    row = rows.iloc[0]
    noisy = ev.degrade(pp.load_rgb(row["rgb_path"]), "noise25", int(row["mat_index"]))
    img = dp.prepare_input(noisy, dp.DEFAULT_INPUT)      # the pipeline refines with the preprocessed image as guide
    disp = dp.predict_disparity(model, proc, img, device)
    _, st = dp.refine_depth(disp, img, return_steps=True)
    diff = st["hole_filled"] - st["input"]
    lim = float(np.abs(diff).max())
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.3))
    show(axes[0], an.overlay_edges(img, st["edges"]), "Preprocessed input (was noise σ=25) + Canny edges")
    show(axes[1], st["input"], "Predicted disparity", cmap="magma")
    show(axes[2], st["median_bilateral"], "After median + bilateral", cmap="magma")
    show(axes[3], st["hole_filled"], "After edge-aware smoothing + hole filling", cmap="magma")
    im = axes[4].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    show(axes[4], diff, "Change made by refinement", cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axes[4], fraction=0.035, pad=0.02)
    fig.suptitle("Depth refinement steps (disparity normalised to 0-1; light = near)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "depth_refinement_steps.png", dpi=120)
    plt.close(fig)

    # --- qualitative: RGB | GT | prediction | refined -----------------------------
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.6, 2.9 * len(rows)), layout="constrained")
    for r, (_, row) in enumerate(rows.iterrows()):
        img = pp.load_rgb(row["rgb_path"])
        gt = np.load(row["depth_path"])
        mask = ev.nyu_valid_mask(gt)
        x = dp.prepare_input(img, dp.DEFAULT_INPUT)
        disp = dp.predict_disparity(model, proc, x, device)
        pred = ev.align_scale_shift(disp, gt, mask)
        ref = ev.align_scale_shift(dp.refine_depth(disp, x), gt, mask)
        vmin, vmax = float(gt[mask].min()), float(gt[mask].max())
        kw = {"cmap": DEPTH_CMAP, "vmin": vmin, "vmax": vmax}
        show(axes[r, 0], img, "RGB" if r == 0 else None)
        show(axes[r, 1], np.where(gt > 0, gt, np.nan), "Ground truth (Kinect)" if r == 0 else None, **kw)
        a1, a2 = ev.depth_metrics(pred, gt, mask)["abs_rel"], ev.depth_metrics(ref, gt, mask)["abs_rel"]
        show(axes[r, 2], pred, ("Depth Anything, full preprocessing\n" if r == 0 else "") + f"AbsRel {a1:.3f}", **kw)
        im = axes[r, 3].imshow(ref, **kw)
        show(axes[r, 3], ref, ("+ depth refinement\n" if r == 0 else "") + f"AbsRel {a2:.3f}", **kw)
        fig.colorbar(im, ax=list(axes[r]), fraction=0.012, pad=0.01, label="depth (m)")
    fig.suptitle("NYU Depth V2 test images: prediction aligned to ground truth (scale + shift); "
                 "same colour scale per row", fontsize=11.5)
    fig.savefig(FIG / "depth_qualitative.png", dpi=120)
    plt.close(fig)
    del model, proc
    ev.release_memory()


def main():
    device = torch.device("cuda")
    rows = ev.load_nyu_index(ROOT / "data/nyu/test").iloc[ROWS]
    fig_preprocessing(rows.iloc[1])
    fig_analysis(rows, device)
    fig_depth(rows, device)
    print("saved figures to", FIG)


if __name__ == "__main__":
    main()
