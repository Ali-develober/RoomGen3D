"""After LoRA training: final loss curve + one figure comparing validation grids across steps.

Outputs:
  results/figures/<run>_loss.png            full loss curve (moving average over 100 steps)
  results/figures/<run>_val_progress.png    validation grids at the chosen steps, 2x2
Usage: python scripts/lora_progress_figure.py [--run lora_mmis] [--steps 0 1000 2000 3000]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_lora import plot_loss  # noqa: E402

FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lora_mmis")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1000, 2000, 3000])
    args = ap.parse_args()

    plot_loss(TAB / f"{args.run}_loss.csv", FIG / f"{args.run}_loss.png", window=100)

    grids = [(s, FIG / f"{args.run}_val_step{s:05d}.png") for s in args.steps]
    missing = [str(p) for _, p in grids if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing validation grids: {missing}")
    cols = 2
    rows = (len(grids) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(12 * cols / 1.4, 8.6 * rows / 1.4), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, (step, path) in zip(ax.ravel(), grids):
        a.imshow(mpimg.imread(path))
        a.set_title("step 0 (base SD v1.5)" if step == 0 else f"step {step} (SD v1.5 + LoRA)",
                    fontsize=14, fontweight="bold")
    fig.suptitle("Same prompts, same seeds: how the LoRA changes generation during training", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG / f"{args.run}_val_progress.png", dpi=110)
    print("saved", FIG / f"{args.run}_loss.png", "and", FIG / f"{args.run}_val_progress.png")


if __name__ == "__main__":
    main()
