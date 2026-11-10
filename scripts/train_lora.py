"""Train the interior-design LoRA on SD v1.5 (see src/train_lora.py).

Full run (resumes automatically from checkpoints/lora_mmis/latest.txt):
  python scripts/train_lora.py
Smoke test (50 steps, separate output folder, prints VRAM + time estimate):
  python scripts/train_lora.py --smoke
Start over instead of resuming:
  python scripts/train_lora.py --no-resume
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_lora import TrainConfig, estimate_total_time, train  # noqa: E402


def parse_args() -> tuple[TrainConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--smoke", action="store_true", help="50-step test run into checkpoints/lora_smoke")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-val-at-start", action="store_true", help="skip the step-0 (base model) grid")
    args = ap.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    if args.smoke:
        cfg.max_steps, cfg.checkpoint_every = 50, 50
        cfg.output_dir, cfg.run_name = "checkpoints/lora_smoke", "lora_smoke"
    # resolve relative paths against the project root
    for name in ["train_csv", "image_root", "output_dir", "fig_dir", "table_dir"]:
        p = Path(getattr(cfg, name))
        setattr(cfg, name, str(p if p.is_absolute() else ROOT / p))
    return cfg, args


def main():
    cfg, args = parse_args()
    train(cfg, resume=not args.no_resume, val_at_start=not args.no_val_at_start)
    if args.smoke:
        est = estimate_total_time(Path(cfg.table_dir) / f"{cfg.run_name}_loss.csv", TrainConfig.max_steps)
        print(f"smoke test: {est['sec_per_step']:.2f} s/step, peak VRAM {est['vram_peak_gb']:.2f} GB "
              f"-> {TrainConfig.max_steps} steps ~ {est['est_hours']:.1f} h ({est['note']})")


if __name__ == "__main__":
    main()
