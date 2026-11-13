"""Validate the floor plan (src/floorplan.py) on the 654 NYU Depth V2 test images.

For every image the plan is built three times from the SAME segmentation and the real NYU camera:
  gt        ground-truth Kinect depth                        (reference)
  metric    Depth Anything V2 Metric Indoor, as predicted   (what the app does: no ground truth)
  scaled    the same prediction multiplied by median(gt / pred) over the valid pixels
            (diagnostic only: separates the global-scale error from the shape error)
Only pixels inside the Eigen crop with valid ground truth are used, for all three.

Metrics of metric / scaled against gt (src/floorplan.compare_plans):
  walls      matched by the overlap of their viewing directions (scale-independent);
             length error (m and relative), median length ratio pred / gt, wall-distance error
  furniture  matched by shared image instances; footprint IoU (completed rectangle and observed
             part), centroid error (m and relative to the object's distance)
  gravity    angle between the up directions estimated from the two depth maps

Stages (each resumable, one model on the GPU at a time):
  1. segmentation -> outputs/floorplan_eval/seg/<name>.png        (ADE20K class ids, uint8)
  2. depth        -> outputs/floorplan_eval/pred/<name>.npy       (gated input + refinement, float16)
  3. plans + metrics on the CPU -> per-image table
Outputs:
  results/tables/floorplan_nyu_per_image.csv, floorplan_nyu.csv, floorplan_nyu_stats.csv
  results/figures/floorplan_nyu_examples.png
Usage: python scripts/eval_floorplan.py [--limit 20]
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
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.utils import set_seed  # noqa: E402

NYU_K = (518.8579, 519.4696, 325.5824, 253.7362)   # fx, fy, cx, cy of the NYU Kinect RGB camera (640x480)
CONDITIONS = ["metric", "scaled"]
METRICS = ["matched_walls", "wall_len_abs_err_m", "wall_len_rel_err", "wall_len_ratio", "wall_dist_rel_err",
           "matched_furniture", "furn_iou", "furn_iou_observed", "furn_centroid_err_m", "furn_centroid_rel_err",
           "gravity_angle_deg"]


def seg_stage(df, out: Path, device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (out / f"{r['name']}.png").exists()]
    model, proc = an.load_segmenter(device)
    id2label = model.config.id2label
    for r in tqdm(todo, desc="segmentation", unit="img"):
        cm = an.segment_classes(model, proc, pp.load_rgb(r["rgb_path"]), device)
        cv2.imwrite(str(out / f"{r['name']}.png"), cm.astype(np.uint8))
    del model, proc
    ev.release_memory()
    return id2label


def depth_stage(df, out: Path, device) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (out / f"{r['name']}.npy").exists()]
    if not todo:
        return
    model, proc = dp.load_depth_model(pl.DEPTH_MODEL_3D, device)
    for r in tqdm(todo, desc="depth", unit="img"):
        img, _ = pl.preprocess(pp.load_rgb(r["rgb_path"]))
        d = dp.refine_metric_depth(dp.predict_depth(model, proc, img, device), img)
        np.save(out / f"{r['name']}.npy", d.astype(np.float16))
    del model, proc
    ev.release_memory()


def plan_stage(df, seg_dir: Path, pred_dir: Path, id2label: dict, keep: set) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, stats, kept = [], [], {}
    for _, r in tqdm(df.iterrows(), total=len(df), desc="plans", unit="img"):
        name = r["name"]
        gt = np.load(r["depth_path"]).astype(np.float64)
        pred = np.load(pred_dir / f"{name}.npy").astype(np.float64)
        cm = cv2.imread(str(seg_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        mask = ev.nyu_valid_mask(gt)
        scale = float(np.median(gt[mask] / pred[mask]))
        depths = {"gt": gt, "metric": pred, "scaled": pred * scale}
        plans = {}
        for c, d in depths.items():
            t0 = time.perf_counter()
            plans[c] = fp.build_floorplan(d, NYU_K, cm, id2label, valid=mask)
            stats.append({"name": name, "condition": c, "seconds": time.perf_counter() - t0,
                          **fp.plan_summary(plans[c])})
        for c in CONDITIONS:
            rows.append({"name": name, "condition": c, "depth_scale_gt_over_pred": scale,
                         **fp.compare_plans(plans["gt"], plans[c])})
        if name in keep:
            kept[name] = plans
    return pd.DataFrame(rows), pd.DataFrame(stats), kept


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c, g in per.groupby("condition", sort=False):
        w = g[g["matched_walls"] > 0]
        f = g[g["matched_furniture"] > 0]
        out.append({
            "condition": c, "images": len(g),
            "gt_walls": int(g["gt_walls"].sum()), "pred_walls": int(g["pred_walls"].sum()),
            "matched_walls": int(g["matched_walls"].sum()),
            "wall_recall": g["matched_walls"].sum() / max(g["gt_walls"].sum(), 1),
            "wall_precision": g["matched_walls"].sum() / max(g["pred_walls"].sum(), 1),
            "wall_len_abs_err_m": w["wall_len_abs_err_m"].mean(), "wall_len_rel_err": w["wall_len_rel_err"].mean(),
            "wall_len_rel_err_median": w["wall_len_rel_err"].median(), "wall_len_ratio_median": w["wall_len_ratio"].median(),
            "wall_dist_rel_err": w["wall_dist_rel_err"].mean(),
            "gt_furniture": int(g["gt_furniture"].sum()), "matched_furniture": int(g["matched_furniture"].sum()),
            "furn_iou": f["furn_iou"].mean(), "furn_iou_observed": f["furn_iou_observed"].mean(),
            "furn_centroid_err_m": f["furn_centroid_err_m"].mean(),
            "furn_centroid_err_m_median": f["furn_centroid_err_m"].median(),
            "furn_centroid_rel_err": f["furn_centroid_rel_err"].mean(),
            "gravity_angle_deg_median": g["gravity_angle_deg"].median(),
        })
    return pd.DataFrame(out)


def fig_examples(df, kept: dict, path: Path) -> None:
    names = list(kept)
    fig, axes = plt.subplots(len(names), 4, figsize=(16, 4.1 * len(names)), dpi=110)
    axes = np.atleast_2d(axes)
    for i, n in enumerate(names):
        r = df[df["name"] == n].iloc[0]
        axes[i, 0].imshow(pp.load_rgb(r["rgb_path"]))
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"NYU test {n}", fontsize=9)
        for j, (c, lab) in enumerate((("gt", "ground-truth depth"), ("metric", "predicted depth (as is)"),
                                      ("scaled", "predicted, median-scaled"))):
            fp.draw_floorplan(kept[n][c], ax=axes[i, j + 1], title=f"Plan from {lab}")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N evenly spaced images")
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    df = ev.load_nyu_index(ROOT / "data/nyu/test", limit=args.limit)
    tag = "" if args.limit is None else "_smoke"
    work = ROOT / "outputs" / "floorplan_eval"
    id2label = seg_stage(df, work / "seg", device)
    depth_stage(df, work / "pred", device)
    keep = set(df["name"].iloc[np.linspace(0, len(df) - 1, 5).round().astype(int)])
    per, stats, kept = plan_stage(df, work / "seg", work / "pred", id2label, keep)
    summ = summarize(per)
    st = stats.groupby("condition", sort=False).agg(
        walls=("walls", "mean"), wall_length_m=("wall_length_m", "mean"), furniture=("furniture", "mean"),
        doors=("doors", "mean"), windows=("windows", "mean"), manhattan=("manhattan", "mean"),
        closed=("closed", "mean"), camera_height_m=("camera_height_m", "median"),
        ceiling_height_m=("ceiling_height_m", "median"), floor_plane=("gravity", lambda g: (g == "floor plane").mean()),
        seconds=("seconds", "mean")).reset_index()
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())
    print(st.round(3).to_string(index=False))
    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    per.to_csv(tables / f"floorplan_nyu_per_image{tag}.csv", index=False, float_format="%.5f")
    summ.to_csv(tables / f"floorplan_nyu{tag}.csv", index=False, float_format="%.5f")
    st.to_csv(tables / f"floorplan_nyu_stats{tag}.csv", index=False, float_format="%.5f")
    fig_examples(df, kept, figs / f"floorplan_nyu_examples{tag}.png")


if __name__ == "__main__":
    main()
