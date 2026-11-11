"""2D evaluation: vanilla SD v1.5 vs. LoRA checkpoints on the held-out MMIS captions (see src/evaluate.py).

Full run (resumable: images already in outputs/eval_2d/<model>/ are skipped):
  python scripts/eval_2d.py
Smoke test (20 evenly spaced captions; images are reused by the full run; tables/figures get a _smoke suffix):
  python scripts/eval_2d.py --limit 20
Only recompute metrics/figures from existing images:
  python scripts/eval_2d.py --skip-gen

Outputs:
  outputs/eval_2d/<model>/NNNN.png                 generated images (not in git)
  results/tables/eval_2d{tag}.csv                  FID, KID mean/std, CLIP mean/std per model (+ real reference)
  results/tables/eval_2d_clip_per_image{tag}.csv   CLIP Score of every image
  results/figures/eval_2d_metrics{tag}.png         bar charts
  results/figures/eval_2d_qualitative{tag}.png     6 captions x (real + 4 models)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate as ev  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N evenly spaced captions")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--out-root", default="outputs/eval_2d")
    args = ap.parse_args()
    tag = "" if args.limit is None else "_smoke"

    set_seed(42)
    device = torch.device("cuda")
    df = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis", limit=args.limit)
    out_root = ROOT / args.out_root
    print(f"{len(df)} held-out captions, {len(ev.MODELS)} models -> {out_root}")

    t0 = time.perf_counter()
    if not args.skip_gen:
        timing = ev.generate_all(df, out_root, device, batch_size=args.batch_size, root=ROOT)
        if timing:
            spi = sum(t["sec_per_image"] * t["generated"] for t in timing.values()) / \
                sum(t["generated"] for t in timing.values())
            print(f"generation: {spi:.2f} s/image -> full run (500 x {len(ev.MODELS)} images) "
                  f"~ {spi * 500 * len(ev.MODELS) / 3600:.2f} h of generation")
    t_gen = time.perf_counter() - t0

    t1 = time.perf_counter()
    fake = {m: [str(out_root / m / f) for f in df["file"]] for m in ev.MODELS}
    fk = ev.fid_kid(list(df["real_path"]), fake, device)
    clip_df = ev.clip_all(df, {**fake, "real_holdout": list(df["real_path"])}, device)
    summary = ev.summarize(fk, clip_df, len(df))
    best = ev.select_best(summary)
    t_metrics = time.perf_counter() - t1

    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    summary.to_csv(tables / f"eval_2d{tag}.csv", index=False, float_format="%.6g")
    clip_df.to_csv(tables / f"eval_2d_clip_per_image{tag}.csv", index=False, float_format="%.4f")
    ev.plot_metrics(summary, figs / f"eval_2d_metrics{tag}.png", best=best)
    ev.plot_qualitative(ev.qualitative_rows(df), out_root, figs / f"eval_2d_qualitative{tag}.png")

    print(summary.drop(columns=["label"]).to_string(index=False))
    print(f"best (FID+KID rank, CLIP tiebreak): {best}")
    print(f"time: generation {t_gen / 60:.1f} min, metrics {t_metrics / 60:.1f} min")


if __name__ == "__main__":
    main()
