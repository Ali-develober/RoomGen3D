"""Extract the official NYU Depth V2 test (654) or train (795) images from the labeled .mat.

The train split is only used to SELECT preprocessing settings (scripts/eval_gated.py); all
reported depth metrics are on the test split.

Outputs (data/nyu/<split>/):
  rgb/0001.png          RGB image, 640x480, uint8 (name = 1-based index in the labeled .mat)
  depth/0001.npy        depth in metres, float32 (640x480)
  depth_png/0001.png    same depth as 16-bit PNG in millimetres (for quick viewing; test only)
  index.csv             file name, original index, depth min/max
Plus, for the test split, results/figures/nyu_test_samples.png and results/tables/nyu_test_stats.csv.

Usage: python scripts/extract_nyu.py [--split test|train] [--limit 5]
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import nyu_read_pair, nyu_split_indices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "nyu" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only extract N images (test)")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    args = ap.parse_args()
    OUT = ROOT / "data" / "nyu" / args.split
    is_test = args.split == "test"
    for d in ["rgb", "depth"] + (["depth_png"] if is_test else []):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    idx = nyu_split_indices(RAW / "splits.mat", args.split)
    print(f"official {args.split} split: {len(idx)} images")
    if args.limit:
        idx = idx[: args.limit]

    rows = []
    with h5py.File(RAW / "nyu_depth_v2_labeled.mat", "r") as h5:
        print("labeled split:", h5["images"].shape, h5["depths"].shape)
        for i in tqdm(idx, unit="img"):
            name = f"{i + 1:04d}"
            rgb, depth = nyu_read_pair(h5, int(i))
            cv2.imwrite(str(OUT / "rgb" / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(OUT / "depth" / f"{name}.npy", depth)
            if is_test:
                cv2.imwrite(str(OUT / "depth_png" / f"{name}.png"),
                            np.clip(depth * 1000, 0, 65535).astype(np.uint16))
            rows.append({"name": name, "mat_index": int(i), "h": rgb.shape[0], "w": rgb.shape[1],
                         "depth_min_m": float(depth.min()), "depth_max_m": float(depth.max()),
                         "depth_mean_m": float(depth.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "index.csv", index=False)
    if not is_test:
        print(f"extracted {len(df)} pairs -> {OUT}")
        return

    if not args.limit:
        (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
        stats = df[["depth_min_m", "depth_max_m", "depth_mean_m"]].describe().round(3)
        stats.loc["n_images"] = len(df)
        stats.to_csv(ROOT / "results" / "tables" / "nyu_test_stats.csv")

    # sample figure: RGB + depth for 4 images
    pick = df["name"].iloc[np.linspace(0, len(df) - 1, min(4, len(df))).astype(int)]
    fig, ax = plt.subplots(2, len(pick), figsize=(4 * len(pick), 6), squeeze=False)
    for c, name in enumerate(pick):
        rgb = cv2.cvtColor(cv2.imread(str(OUT / "rgb" / f"{name}.png")), cv2.COLOR_BGR2RGB)
        d = np.load(OUT / "depth" / f"{name}.npy")
        ax[0, c].imshow(rgb); ax[0, c].set_title(f"RGB {name}"); ax[0, c].axis("off")
        im = ax[1, c].imshow(d, cmap="inferno"); ax[1, c].set_title(f"depth {d.min():.1f}-{d.max():.1f} m"); ax[1, c].axis("off")
        fig.colorbar(im, ax=ax[1, c], fraction=0.046)
    fig.suptitle("NYU Depth V2 test split: RGB and ground-truth depth")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "nyu_test_samples.png", dpi=110)
    print(f"extracted {len(df)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
