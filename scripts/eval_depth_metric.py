"""Score the metric depth model of the 3D stage on the 654 NYU Depth V2 test images (clean, raw input).

Two protocols:
  aligned   - least-squares scale + shift in inverse depth, exactly as for MiDaS / Depth Anything,
              so the row is comparable with results/tables/depth_models.csv (it is added there).
  metric    - the predicted metres are used as they are (no alignment at all), capped at 10 m.
              This is the number that matters for 3D, where no ground truth is available.

Outputs:
  outputs/depth_eval/depth_anything_metric_clean.csv   per image (resumable: skipped if complete)
  results/tables/depth_models.csv                      row "depth_anything_metric" added / replaced
  results/tables/depth_metric_model.csv                both protocols
Usage: python scripts/eval_depth_metric.py [--limit 20] [--fresh]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.utils import set_seed  # noqa: E402

NAME = "depth_anything_metric"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N evenly spaced images (no tables written)")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    df = ev.load_nyu_index(ROOT / "data/nyu/test", limit=args.limit)
    tag = "" if args.limit is None else "_smoke"
    part = ROOT / f"outputs/depth_eval{tag}/{NAME}_clean.csv"
    part.parent.mkdir(parents=True, exist_ok=True)

    if part.exists() and not args.fresh and pd.read_csv(part)["name"].nunique() == len(df):
        print(f"{part.name}: done, skipping")
    else:
        model, proc = dp.load_depth_model(NAME, device)
        rows = []
        for _, r in tqdm(df.iterrows(), total=len(df), desc=NAME, unit="img"):
            gt = np.load(r["depth_path"])
            mask = ev.nyu_valid_mask(gt)
            edge = ev.depth_edge_mask(gt, mask)
            pred = dp.predict_depth(model, proc, pp.load_rgb(r["rgb_path"]), device)
            aligned = ev.align_scale_shift(1.0 / pred, gt, mask)
            metric = np.minimum(pred, ev.NYU_MAX_DEPTH)
            for protocol, d in (("aligned", aligned), ("metric", metric)):
                rows.append({"name": r["name"], "protocol": protocol, **ev.depth_metrics(d, gt, mask),
                             **ev.boundary_metrics(d, gt, mask, edge)})
        del model, proc
        ev.release_memory()
        tmp = part.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(part)

    per = pd.read_csv(part, dtype={"name": str})
    both = per.groupby("protocol", sort=False)[ev.ALL_DEPTH_METRICS].mean().reset_index()
    both.insert(0, "n_images", per["name"].nunique())
    both.insert(0, "label", dp.DEPTH_MODEL_LABELS[NAME])
    both.insert(0, "model", NAME)
    pd.set_option("display.width", 220)
    print(both.round(4).to_string(index=False))
    if args.limit is not None:
        return
    tables = ROOT / "results/tables"
    both.to_csv(tables / "depth_metric_model.csv", index=False, float_format="%.5f")
    models = pd.read_csv(tables / "depth_models.csv")
    models = models[models["model"] != NAME]
    row = both[both["protocol"] == "aligned"].drop(columns="protocol")
    models = pd.concat([models, row[models.columns]], ignore_index=True)
    models.to_csv(tables / "depth_models.csv", index=False, float_format="%.5f")
    print("\ndepth_models.csv:\n", models.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
