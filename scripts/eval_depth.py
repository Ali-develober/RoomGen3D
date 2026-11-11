"""Depth evaluation on the NYU Depth V2 test set (see src/evaluate.py, src/depth.py).

One run covers, for BOTH depth models (MiDaS DPT-Hybrid and Depth Anything ViT-B):
  1. Ablation on clean images: raw -> +denoise -> +CLAHE -> +full preprocessing (noise-aware and
     plain 3-step version) -> + depth refinement
  2. Robustness: the same ablation on degraded copies, a moderate and a severe level of each:
     Gaussian noise 15 / 25 / 50, motion blur 15 / 25 px, low light (gamma 2.2 + noise 10 /
     gamma 3.0 + noise 20)
  3. Boundary metrics for every row: AbsRel / RMSE on GT depth-edge pixels, flying-pixel ratio
  4. Denoising quality: PSNR / SSIM against the clean image per noise level (CPU only)

Full run (resumable: finished parts in outputs/depth_eval/ are skipped):
  python scripts/eval_depth.py
Smoke test (20 evenly spaced images, tables/figures get a _smoke suffix):
  python scripts/eval_depth.py --limit 20

Outputs (results/tables/, {tag} = "" or "_smoke"):
  depth_models{tag}.csv               the two models on clean raw images
  depth_ablation{tag}.csv             ablation on clean images, per model
  depth_robustness{tag}.csv           model x degradation x ablation step, with recovery %
  depth_refinement_boundary{tag}.csv  boundary metrics before / after depth refinement
  depth_denoise_quality{tag}.csv      PSNR / SSIM per noise level and denoiser
  depth_per_image{tag}.csv            per-image metrics behind the depth tables
Figures (results/figures/): depth_ablation{tag}.png, depth_robustness_<model>{tag}.png
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.utils import set_seed  # noqa: E402

MODEL_ORDER = ["midas", "depth_anything"]           # one on the GPU at a time


def part_ok(path: Path, n_images: int, columns: list[str], fresh: bool) -> bool:
    """A part is reused only if it exists, has the current columns and covers every image."""
    if fresh or not path.exists():
        return False
    old = pd.read_csv(path, dtype={"name": str})
    ok = set(columns) <= set(old.columns) and old["name"].nunique() == n_images
    if not ok:
        print(f"{path.name}: from an older version of the script, recomputing")
    return ok


def write_part(out: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)                               # atomic: a killed run never leaves a half-written part


def run_part(part_path: Path, model_name: str, df, degradation, device, cache: dict) -> float:
    """Run one (model, degradation) part unless it is already done. Loads the model lazily."""
    if part_ok(part_path, len(df), ["input", "refined"] + ev.ALL_DEPTH_METRICS, cache["fresh"]):
        old = pd.read_csv(part_path)
        if set(old["input"]) == set(ev.INPUT_VARIANTS):
            print(f"{part_path.name}: done, skipping")
            return 0.0
    if cache.get("name") != model_name:             # one depth model on the GPU at a time
        cache.pop("model", None)
        cache.pop("proc", None)
        ev.release_memory()
        cache["model"], cache["proc"] = dp.load_depth_model(model_name, device)
        cache["name"] = model_name
    t0 = time.perf_counter()
    out = ev.eval_depth_run(cache["model"], cache["proc"], df, degradation, ev.INPUT_VARIANTS,
                            ev.REFINE_VARIANTS, device, desc=f"{model_name} / {degradation}")
    out.insert(0, "model", model_name)
    write_part(out, part_path)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N evenly spaced images")
    ap.add_argument("--fresh", action="store_true", help="recompute parts even if they already exist")
    args = ap.parse_args()
    tag = "" if args.limit is None else "_smoke"

    set_seed(42)
    device = torch.device("cuda")
    df = ev.load_nyu_index(ROOT / "data/nyu/test", limit=args.limit)
    parts_dir = ROOT / f"outputs/depth_eval{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    print(f"{len(df)} NYU test images -> parts in {parts_dir}")
    timing = {}

    # ---- denoising PSNR / SSIM (CPU, no model loaded) ----------------------------
    dq_path = parts_dir / "denoise_quality.csv"
    if part_ok(dq_path, len(df), ["noise_sigma", "method", "psnr", "ssim"], args.fresh):
        print(f"{dq_path.name}: done, skipping")
    else:
        t0 = time.perf_counter()
        write_part(ev.denoise_quality_run(df), dq_path)
        timing["denoise quality"] = time.perf_counter() - t0
    dq = ev.summarize_denoise_quality(pd.read_csv(dq_path, dtype={"name": str}))
    dq.to_csv(tables / f"depth_denoise_quality{tag}.csv", index=False, float_format="%.4f")

    # ---- depth parts: model x degradation ----------------------------------------
    cache = {"fresh": args.fresh}
    part_paths = []
    for name in MODEL_ORDER:
        for deg in ev.DEGRADATIONS:
            part_paths.append(parts_dir / f"{name}_{deg}.csv")
            timing[f"{name} / {deg}"] = run_part(part_paths[-1], name, df, deg, device, cache)
    cache.clear()
    ev.release_memory()

    # ---- combine parts -> tables ------------------------------------------------
    per = pd.concat([pd.read_csv(p, dtype={"name": str}) for p in part_paths], ignore_index=True)
    key = {(inp, ref): c for c, (inp, ref) in ev.ABLATION.items()}
    per["condition"] = [key[(i, bool(r))] for i, r in zip(per["input"], per["refined"])]
    per = per[["model", "degradation", "condition", "name"] + ev.ALL_DEPTH_METRICS]
    per.to_csv(tables / f"depth_per_image{tag}.csv", index=False, float_format="%.5f")

    summary = ev.summarize_depth(per, ["model", "degradation", "condition"])
    rank = {v: i for i, v in enumerate(MODEL_ORDER + ev.DEGRADATIONS + list(ev.ABLATION))}
    summary = summary.sort_values(["model", "degradation", "condition"],
                                  key=lambda s: s.map(rank)).reset_index(drop=True)

    models = summary[(summary["degradation"] == "clean") & (summary["condition"] == "raw")].copy()
    models.insert(1, "label", models["model"].map(dp.DEPTH_MODEL_LABELS))
    models = models.drop(columns=["degradation", "condition"])
    models.to_csv(tables / f"depth_models{tag}.csv", index=False, float_format="%.5f")

    ablation = summary[summary["degradation"] == "clean"].drop(columns="degradation").copy()
    ablation.insert(2, "label", ablation["condition"].map(ev.ABLATION_LABELS))
    ablation.to_csv(tables / f"depth_ablation{tag}.csv", index=False, float_format="%.5f")

    rec = pd.concat([ev.recovery_table(g).assign(model=m) for m, g in summary.groupby("model", sort=False)])
    robust = summary.merge(rec, on=["model", "degradation", "condition"], how="left")
    robust.to_csv(tables / f"depth_robustness{tag}.csv", index=False, float_format="%.5f")

    refine = ev.refinement_table(summary)
    refine = refine.sort_values(["model", "degradation"], key=lambda s: s.map(rank), kind="stable")
    refine.to_csv(tables / f"depth_refinement_boundary{tag}.csv", index=False, float_format="%.5f")

    ev.plot_depth_ablation(ablation, figs / f"depth_ablation{tag}.png", dp.DEPTH_MODEL_LABELS)
    for name in MODEL_ORDER:
        ev.plot_depth_robustness(robust[robust["model"] == name], figs / f"depth_robustness_{name}{tag}.png",
                                 dp.DEPTH_MODEL_LABELS[name])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nDenoising quality (vs. clean image):\n", dq.to_string(index=False))
    print("\nModel comparison (clean, raw):\n", models.to_string(index=False))
    print("\nAblation (clean):\n", ablation.drop(columns="label").to_string(index=False))
    print("\nRobustness:\n", robust.drop(columns="n_images").to_string(index=False))
    print("\nDepth refinement, boundary metrics (before -> after):\n", refine.to_string(index=False))
    ran = {k: v for k, v in timing.items() if v > 0}
    if ran:
        total = sum(ran.values())
        print(f"\ntime: {total / 60:.1f} min for {len(df)} images "
              f"-> full run (654 images) ~ {total / len(df) * 654 / 60:.0f} min")


if __name__ == "__main__":
    main()
