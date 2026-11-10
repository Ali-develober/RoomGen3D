"""Before/after figure + sharpness table for the LoRA training preprocessing.

For sample training images: original 256 (shown with nearest-neighbour so pixels stay
visible), bicubic 512, bicubic 512 + unsharp mask, and a zoomed crop of each.
Sharpness (Laplacian variance) is measured on 200 training images.

Outputs: results/figures/train_preprocessing.png, results/tables/train_preprocessing_sharpness.csv
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import (USM_AMOUNT, USM_SIGMA, load_rgb, sharpness,  # noqa: E402
                               unsharp_mask, upsample_bicubic)

ROOT = Path(__file__).resolve().parents[1]
MMIS = ROOT / "data" / "mmis"


def main():
    train = pd.read_csv(MMIS / "train.csv")

    # sharpness table on 200 images (all measured at 512 so values are comparable)
    rows = []
    for p in train.sample(n=200, random_state=42)["path"]:
        img = load_rgb(MMIS / p)
        nearest = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        up = upsample_bicubic(img)
        rows.append({"nearest_512": sharpness(nearest), "bicubic_512": sharpness(up),
                     "bicubic_512_usm": sharpness(unsharp_mask(up))})
    t = pd.DataFrame(rows).describe().loc[["mean", "50%", "std"]].round(1)
    t.to_csv(ROOT / "results" / "tables" / "train_preprocessing_sharpness.csv")
    print(t.to_string())

    # figure
    picks = train.sample(n=3, random_state=7)["path"].tolist()
    fig, ax = plt.subplots(len(picks), 4, figsize=(16, 4.2 * len(picks)))
    for r, p in enumerate(picks):
        img = load_rgb(MMIS / p)
        up = upsample_bicubic(img)
        usm = unsharp_mask(up)
        y, x = 192, 192  # 128x128 crop from the centre of the 512 image
        panels = [(cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST), "original 256x256 (nearest, for display)"),
                  (up, "bicubic 512x512"),
                  (usm, f"bicubic + unsharp mask (sigma={USM_SIGMA}, amount={USM_AMOUNT})"),
                  (np.hstack([up[y:y + 128, x:x + 128], usm[y:y + 128, x:x + 128]]), "zoom: bicubic | + unsharp")]
        for c, (im, title) in enumerate(panels):
            ax[r, c].imshow(im)
            ax[r, c].set_title(f"{title}\nLaplacian var = {sharpness(im):.0f}" if c < 3 else title, fontsize=9)
            ax[r, c].axis("off")
        ax[r, 0].text(-0.04, 0.5, Path(p).parts[-3] + "\n" + Path(p).parts[-2], transform=ax[r, 0].transAxes,
                      rotation=90, ha="right", va="center", fontsize=10)
    fig.suptitle("LoRA training preprocessing: bicubic upsampling 256->512, then mild unsharp masking (no CLAHE)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "train_preprocessing.png", dpi=100)


if __name__ == "__main__":
    main()
