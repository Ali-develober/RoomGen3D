"""Noise-gated preprocessing: select on the NYU TRAIN split, then evaluate once on the TEST split.

preprocessing.preprocess_gated denoises an image only if its estimated noise std is above a
threshold (no CLAHE, no sharpening). This script:

  Stage 1 (select, 795 train images): for both depth models and all 8 degradations, score the raw,
      Gaussian-denoised and adaptive-bilateral-denoised image. From those per-image numbers every
      (denoiser, threshold) pair is scored without more forward passes; the best one is selected
      by a rule fixed in advance (src/evaluate.py, "Selection rule"). The test split is not read.
  Stage 2 (test, 654 test images): the selected config, once, against the raw image, for both
      models and all degradations, with and without depth refinement.

Full run (resumable: finished parts in outputs/depth_gated/ are skipped):
  python scripts/eval_gated.py
Smoke test (N evenly spaced TRAIN images for BOTH stages, so the test split stays untouched;
outputs get a _smoke suffix):
  python scripts/eval_gated.py --limit 20

Needs data/nyu/train (python scripts/extract_nyu.py --split train).

Outputs (results/tables/, {tag} = "" or "_smoke"):
  depth_gated_selection{tag}.csv          every denoiser x threshold, ranked (train)
  depth_gated_selection_cells{tag}.csv    the same per model x degradation (train)
  depth_gated_selected{tag}.csv           the selected config (read by stage 2)
  depth_gated_test{tag}.csv               raw vs. gated on test, with recovery % and gated share
  depth_gated_per_image{tag}.csv          per-image test metrics
Figures (results/figures/): depth_gated_selection{tag}.png, depth_gated_test{tag}.png
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.utils import set_seed  # noqa: E402

MODEL_ORDER = ["midas", "depth_anything"]           # one on the GPU at a time
TEST_CONDITIONS = ["raw", "gated", "raw_refine", "gated_refine"]


def part_ok(path: Path, n_images: int, inputs: set, fresh: bool) -> bool:
    """A part is reused only if it exists, covers every image and has exactly the expected inputs."""
    if fresh or not path.exists():
        return False
    old = pd.read_csv(path, dtype={"name": str})
    return old["name"].nunique() == n_images and set(old["input"]) == inputs


def run_parts(stage: str, parts_dir: Path, df, transforms: dict, refine_for: list, boundary: bool, device,
              fresh: bool, timing: dict) -> pd.DataFrame:
    """Run every (model, degradation) part of one stage that is not done yet; return all rows."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    cache, paths = {}, []
    for name in MODEL_ORDER:
        for deg in ev.DEGRADATIONS:
            path = parts_dir / f"{name}_{deg}.csv"
            paths.append(path)
            if part_ok(path, len(df), set(transforms), fresh):
                print(f"{stage} / {path.name}: done, skipping")
                continue
            if cache.get("name") != name:           # one depth model on the GPU at a time
                cache.clear()
                ev.release_memory()
                cache["model"], cache["proc"] = dp.load_depth_model(name, device)
                cache["name"] = name
            t0 = time.perf_counter()
            out = ev.eval_inputs_run(cache["model"], cache["proc"], df, deg, transforms, refine_for, device,
                                     desc=f"{stage} / {name} / {deg}", boundary=boundary)
            out.insert(0, "model", name)
            tmp = path.with_suffix(".tmp")
            out.to_csv(tmp, index=False)
            tmp.replace(path)                       # atomic: a killed run never leaves a half-written part
            timing[stage] = timing.get(stage, 0.0) + time.perf_counter() - t0
    cache.clear()
    ev.release_memory()
    return pd.concat([pd.read_csv(p, dtype={"name": str}) for p in paths], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N evenly spaced TRAIN images for both stages")
    ap.add_argument("--fresh", action="store_true", help="recompute parts even if they already exist")
    ap.add_argument("--select-only", action="store_true", help="stop after the selection on the train split")
    args = ap.parse_args()
    smoke = args.limit is not None
    tag = "_smoke" if smoke else ""

    set_seed(42)
    device = torch.device("cuda")
    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    out_root = ROOT / f"outputs/depth_gated{tag}"
    timing = {}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    # ---- stage 1: selection on the TRAIN split -----------------------------------
    train = ev.load_nyu_index(ROOT / "data/nyu/train", limit=args.limit)
    print(f"stage 1 (select): {len(train)} NYU train images")
    per = run_parts("select", out_root / "select", train, ev.gate_transforms(), [], False, device, args.fresh, timing)
    summary, cells = ev.gate_selection_tables(per)
    best = summary.iloc[0]
    method, threshold = str(best["method"]), float(best["threshold"])
    summary.to_csv(tables / f"depth_gated_selection{tag}.csv", index=False, float_format="%.5f")
    cells.to_csv(tables / f"depth_gated_selection_cells{tag}.csv", index=False, float_format="%.5f")
    pd.DataFrame([{"method": method, "threshold": threshold, "score": best["score"],
                   "selected_on": f"NYU train, {len(train)} images", "rule": "lowest mean relative AbsRel change "
                   "vs. raw over 2 models x 8 degradations; ties -> higher threshold"}]
                 ).to_csv(tables / f"depth_gated_selected{tag}.csv", index=False)
    ev.plot_gate_selection(summary, figs / f"depth_gated_selection{tag}.png", len(train))
    print("\nSelection (train), best 12 of", len(summary), ":\n", summary.head(12).to_string(index=False))
    never = not np.isfinite(threshold)
    print(f"\nSELECTED: method = {method}, threshold = {threshold:g}"
          + ("  (= never denoise: no gated config beat the raw image on train)" if never else ""))
    if args.select_only:
        return

    # ---- stage 2: the selected config, once, on the TEST split -------------------
    # (smoke test: the same train subset, so the test images are never read before the real run)
    test = train if smoke else ev.load_nyu_index(ROOT / "data/nyu/test")
    split = "train subset (smoke)" if smoke else "test"
    print(f"\nstage 2 (test): {len(test)} NYU {split} images, gate = {method} above {threshold:g}")
    transforms = {"raw": lambda x: x, "gated": partial(pp.preprocess_gated, threshold=threshold, method=method)}
    parts_dir = out_root / f"test_{method}_thr{threshold:g}"       # a different config never reuses old parts
    per = run_parts("test", parts_dir, test, transforms, ["raw", "gated"], True, device, args.fresh, timing)
    per["condition"] = per["input"] + np.where(per["refined"].astype(bool), "_refine", "")
    per["gated_share"] = (per["noise_est"] > threshold).astype(float)
    per = per[["model", "degradation", "condition", "name", "noise_est", "gated_share"] + ev.ALL_DEPTH_METRICS]
    per.to_csv(tables / f"depth_gated_per_image{tag}.csv", index=False, float_format="%.5f")

    result = ev.summarize_depth(per, ["model", "degradation", "condition"])
    share = per.groupby(["model", "degradation", "condition"], sort=False)["gated_share"].mean().reset_index()
    result = result.merge(share, on=["model", "degradation", "condition"])
    rank = {v: i for i, v in enumerate(MODEL_ORDER + ev.DEGRADATIONS + TEST_CONDITIONS)}
    result = result.sort_values(["model", "degradation", "condition"], key=lambda s: s.map(rank)).reset_index(drop=True)
    rec = pd.concat([ev.recovery_table(g).assign(model=m) for m, g in result.groupby("model", sort=False)])
    result = result.merge(rec, on=["model", "degradation", "condition"], how="left")
    result.insert(0, "gate", f"{method} > {threshold:g}")
    result.to_csv(tables / f"depth_gated_test{tag}.csv", index=False, float_format="%.5f")
    ev.plot_gated_result(result, figs / f"depth_gated_test{tag}.png", dp.DEPTH_MODEL_LABELS,
                         f"Noise-gated denoising ({method}, noise est. > {threshold:g}) vs. raw image: "
                         f"NYU {split}, {len(test)} images")

    print(f"\nRaw vs. gated ({split}):\n", result.drop(columns=["gate", "n_images"]).to_string(index=False))
    if timing:
        n_tr, n_te = len(train), len(test)
        est = timing.get("select", 0) / n_tr * 795 + timing.get("test", 0) / n_te * 654
        print("\ntime: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timing.items())
              + f" -> full run (795 train + 654 test) ~ {est / 60:.0f} min")


if __name__ == "__main__":
    main()
