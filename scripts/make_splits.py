"""Split data/mmis/subset_clean.csv into a LoRA training set and a balanced FID hold-out.

Outputs:
  data/mmis/train.csv          training images (the rest)
  data/mmis/fid_holdout.csv    500 held-out images, balanced across style x room
  results/tables/mmis_split_counts.csv

Usage: python scripts/make_splits.py [--n-holdout 500]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import holdout_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MMIS = ROOT / "data" / "mmis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-holdout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(MMIS / "subset_clean.csv")
    train, hold = holdout_split(df, args.n_holdout, args.seed)
    assert not set(train["path"]) & set(hold["path"]), "train/holdout overlap"
    train.to_csv(MMIS / "train.csv", index=False)
    hold.to_csv(MMIS / "fid_holdout.csv", index=False)

    counts = pd.DataFrame({"train": train.groupby(["style", "room"]).size(),
                           "fid_holdout": hold.groupby(["style", "room"]).size()}).reset_index()
    counts.to_csv(ROOT / "results" / "tables" / "mmis_split_counts.csv", index=False)
    print(f"train {len(train)}, holdout {len(hold)}; holdout per cell {counts['fid_holdout'].min()}-"
          f"{counts['fid_holdout'].max()}, train per cell {counts['train'].min()}-{counts['train'].max()}")


if __name__ == "__main__":
    main()
