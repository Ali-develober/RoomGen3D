"""PLAN_V5 final: headless app test. Builds the UI, then runs Quick, 360° v5 (default) and 360° v4 (toggle) through
app.run exactly as the Generate button does, checks every output, and records time, peak VRAM and peak RAM per request.
Output: results/tables/v5_app_test.csv (+ PASS / FAIL per mode on stdout; exit code 1 on any failure).
Usage: python scripts/test_app_v5.py [--modes quick v5 v4] [--prompt "..."] [--seed 44]
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import app  # noqa: E402
from src import memory as mem  # noqa: E402


def check(out, mode):
    gallery, plan, viewer = out[0], out[1], out[2]
    assert isinstance(gallery, list) and len(gallery) >= 1, "empty gallery"
    assert plan is not None, "no floor plan"
    assert isinstance(viewer, str) and len(viewer) > 50, "no viewer html"
    if mode == "v5":
        assert "iframe" in viewer and "inside_v5.html" in viewer, "v5 inside viewer missing"
        assert len(gallery) >= 5, "v5 should show hero + 4 wall shots"
        assert gallery[0][0].shape[1] == 1536, f"v5 shots should be 1536 wide, got {gallery[0][0].shape}"
        assert out[-1] and len(out[-1]) == len(gallery), "raw crops missing in Analysis"
        assert "walkable" in out[-2] or "Experimental" in out[-2], "mesh tab missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["quick", "v5", "v4"])
    ap.add_argument("--prompt", default="a mid-century modern style dining room")
    ap.add_argument("--seed", type=int, default=44)
    a = ap.parse_args()
    app.OUT.mkdir(parents=True, exist_ok=True)
    app.build()
    rows, ok_all = [], True
    for mode in a.modes:
        args = {"quick": (app.QUICK, app.V5, 1), "v5": (app.ROOM360, app.V5, app.V5_CANDIDATES),
                "v4": (app.ROOM360, app.V4, 3)}[mode]
        try:
            with mem.PeakMonitor() as pm:
                out = app.run(args[0], a.prompt, a.seed, 30, 7.5, args[1], args[2], False, 0, True)
            check(out, mode)
            status = "PASS"
        except Exception:                            # noqa: BLE001
            traceback.print_exc()
            status, ok_all = "FAIL", False
        rows.append({"mode": mode, "status": status, "seconds": pm.seconds, "peak_vram_gb": pm.peak_vram_gb,
                     "peak_ram_gb": pm.peak_ram_gb, "candidates": args[2], "prompt": a.prompt, "seed": a.seed})
        print(rows[-1], flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results/tables/v5_app_test.csv", index=False, float_format="%.2f")
    print(t.to_string())
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
