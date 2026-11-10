"""Clean the MMIS subset: score every image, pick thresholds from the score
distributions, drop blurry / badly exposed images and pHash near-duplicates,
write the kept list + rejected CSV, figures and count tables.

Usage:
  python scripts/clean_data.py --limit 200   # test run, outputs tagged _test200
  python scripts/clean_data.py               # full subset
Outputs:
  data/mmis/subset_clean.csv                 kept images (path, style, room, caption, extra_text)
  results/tables/mmis_duplicate_pairs[_tag].csv  near-duplicate pairs + Hamming distance
  results/figures/mmis_phash_distances[_tag].png / mmis_duplicate_pairs[_tag].png
  results/tables/mmis_rejected[_tag].csv     rejected images + reason + scores
  results/tables/mmis_scores[_tag].csv       scores for every image
  results/tables/mmis_thresholds[_tag].csv   chosen thresholds
  results/tables/mmis_counts_*[_tag].csv     counts per style/room before vs after
  results/figures/mmis_score_distributions[_tag].png
  results/figures/mmis_kept_vs_rejected[_tag].png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_cleaning import (apply_thresholds, choose_duplicate_threshold, choose_thresholds,  # noqa: E402
                               count_table, duplicate_groups, duplicate_pairs, hashes_to_uint64,
                               mark_duplicates, nearest_neighbor_distance, plot_duplicate_distances,
                               plot_duplicate_pairs, plot_kept_vs_rejected, plot_score_distributions,
                               score_images)
from src.utils import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MMIS = ROOT / "data" / "mmis"
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only clean N already-downloaded images (test)")
    ap.add_argument("--k", type=float, default=1.5, help="Tukey fence multiplier")
    args = ap.parse_args()
    set_seed(42)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MMIS / "subset.csv")
    present = df["path"].map(lambda p: (MMIS / p).exists())
    if not present.all():
        print(f"WARNING: {(~present).sum()} subset images are not downloaded; cleaning the {present.sum()} present")
    df = df[present].reset_index(drop=True)
    tag = ""
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
        tag = f"_test{args.limit}"

    scores = score_images([MMIS / p for p in df["path"]])
    df = pd.concat([df, scores], axis=1)
    th = choose_thresholds(df, k=args.k)
    df["reason"] = apply_thresholds(df, th)

    # near-duplicates (pHash): threshold from the nearest-neighbour distance histogram
    ok = df["phash"].notna()
    h = hashes_to_uint64(df.loc[ok, "phash"])
    idx = np.flatnonzero(ok)
    nn = nearest_neighbor_distance(h)
    th["phash_max_dist"] = choose_duplicate_threshold(nn)
    pairs = [(int(idx[i]), int(idx[j]), d) for i, j, d in duplicate_pairs(h, th["phash_max_dist"])]
    df["reason"] = mark_duplicates(df, duplicate_groups(len(df), pairs))
    pd.DataFrame([{"path_a": df.at[i, "path"], "path_b": df.at[j, "path"], "hamming": d,
                   "same_style": df.at[i, "style"] == df.at[j, "style"],
                   "same_room": df.at[i, "room"] == df.at[j, "room"]} for i, j, d in pairs]
                 ).to_csv(TAB / f"mmis_duplicate_pairs{tag}.csv", index=False)

    kept, rej = df[df["reason"] == ""], df[df["reason"] != ""]
    cols = ["path", "style", "room", "lap_var", "brightness", "pct_dark", "pct_bright", "phash"]
    df[cols + ["reason"]].round(3).to_csv(TAB / f"mmis_scores{tag}.csv", index=False)
    rej[cols + ["reason"]].round(3).to_csv(TAB / f"mmis_rejected{tag}.csv", index=False)
    pd.DataFrame([th]).round(3).to_csv(TAB / f"mmis_thresholds{tag}.csv", index=False)
    if not tag:
        kept[["path", "style", "room", "split", "caption", "extra_text"]].to_csv(MMIS / "subset_clean.csv", index=False)

    cells = count_table(df)
    cells.to_csv(TAB / f"mmis_counts_style_room{tag}.csv", index=False)
    for key in ["room", "style"]:
        t = cells.groupby(key)[["before", "after", "rejected"]].sum()
        t["kept_pct"] = (100 * t["after"] / t["before"]).round(1)
        t.loc["TOTAL"] = [t["before"].sum(), t["after"].sum(), t["rejected"].sum(),
                          round(100 * t["after"].sum() / t["before"].sum(), 1)]
        t.to_csv(TAB / f"mmis_counts_by_{key}{tag}.csv")
    reasons = rej["reason"].str.split(";").explode().value_counts()
    reasons.rename_axis("reason").rename("images").to_csv(TAB / f"mmis_reject_reasons{tag}.csv")

    plot_score_distributions(df, th, FIG / f"mmis_score_distributions{tag}.png")
    plot_kept_vs_rejected(df, MMIS, FIG / f"mmis_kept_vs_rejected{tag}.png")
    plot_duplicate_distances(nn, th["phash_max_dist"], FIG / f"mmis_phash_distances{tag}.png")
    if pairs:
        plot_duplicate_pairs(df, pairs, MMIS, FIG / f"mmis_duplicate_pairs{tag}.png")

    print("thresholds:", {k: round(v, 2) for k, v in th.items()})
    print(f"duplicate pairs (<= {th['phash_max_dist']} bits): {len(pairs)}")
    print(f"kept {len(kept)} / {len(df)} ({100 * len(kept) / len(df):.1f}%), rejected {len(rej)}")
    print("reasons:", reasons.to_dict())


if __name__ == "__main__":
    main()
