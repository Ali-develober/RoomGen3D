"""PLAN_V6 milestone 5: app test. Runs ONE mode per fresh process (the owner's "fresh process" timing), through
app.run exactly as the Generate button does, checks every output, records time, peak VRAM, peak RAM, and scans the
captured log for the run-time issues of the acceptance list:
    hub          any "huggingface.co" / "Could not reach the Hub" / "Couldn't connect to the Hub" line (issue 44)
    truncation   CLIP token truncation warnings ("truncated because CLIP can only handle", issue 42)
    missing      "MISSING" weight load reports (issue 43)
    sd15_loads   how many times the SD 1.5 base is loaded during the request (issue 44)
    startup      models loaded at UI build time (issue 41): the build must load nothing
Usage: python scripts/test_app_v6.py --mode v6|v5|quick [--prompt ...] [--seed 44]   (one mode per process)
       python scripts/test_app_v6.py --all      (spawns one fresh process per mode, writes results/tables/v6_app_test.csv)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))


class Tee(io.TextIOBase):
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for t in self.s:
            t.write(x)
        return len(x)

    def flush(self):
        for t in self.s:
            t.flush()


def check(out, mode):
    gallery, plan, viewer = out[0], out[1], out[2]
    assert isinstance(gallery, list) and len(gallery) >= 1, "empty gallery"
    assert plan is not None, "no floor plan"
    assert isinstance(viewer, str) and len(viewer) > 50, "no viewer html"
    if mode == "v6":
        assert "inside_v6.html" in viewer, "v6 inside viewer missing"
        assert len(gallery) >= 6, f"v6 should show H, H2 and four wall shots, got {len(gallery)}"
        assert gallery[0][0].shape[1] == 1536, f"shots should be 1536 wide, got {gallery[0][0].shape}"
        assert "dollhouse_v6.glb" in out[3], "dollhouse viewer missing"
        assert out[14] is not None and out[14].shape[1] >= 1024, "top-down missing or < 1024 px"
    if mode == "v5":
        assert "inside_v5.html" in viewer, "v5 inside viewer missing"


def one(mode, prompt, seed):
    import torch
    buf = io.StringIO()
    err = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)), contextlib.redirect_stderr(Tee(sys.stderr, err)):
        import app
        from src import memory as mem
        app.OUT.mkdir(parents=True, exist_ok=True)
        app.build()
        startup_models = list(app.MODELS)
        build_s = time.perf_counter() - t0
        n_before = buf.getvalue().count("Loading pipeline components")
        status = "PASS"
        try:
            with mem.PeakMonitor() as pm:
                args = {"quick": (app.QUICK, app.V6), "v6": (app.ROOM360, app.V6), "v5": (app.ROOM360, app.V5)}[mode]
                out = app.run(args[0], prompt, seed, 30, 7.5, args[1], app.V5_CANDIDATES, False, 0, True, True)
            check(out, mode)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            status = "FAIL"
    log = buf.getvalue() + err.getvalue()
    sd15 = log.count("stable-diffusion-v1-5") + log.count("models--stable-diffusion-v1-5")
    row = {"mode": mode, "status": status, "seconds": round(pm.seconds, 1), "process_s": round(time.perf_counter() - t0, 1),
           "ui_build_s": round(build_s, 1), "peak_vram_gb": round(pm.peak_vram_gb, 2), "peak_ram_gb": round(pm.peak_ram_gb, 2),
           "startup_models": ";".join(startup_models),
           "hub_lines": sum(("huggingface.co" in ln or "reach the Hub" in ln or "connect to the Hub" in ln) for ln in log.splitlines()),
           "truncation_warnings": log.count("truncated because CLIP can only handle"),
           "missing_weight_reports": log.count("MISSING"),
           "pipeline_loads": log.count("Loading pipeline components") - n_before, "sd15_mentions": sd15,
           "prompt": prompt, "seed": seed}
    (ROOT / "outputs" / f"app_test_v6_{mode}.log").write_text(log, encoding="utf-8")
    print("RESULT " + json.dumps(row), flush=True)
    _ = torch
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="v6")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prompt", default="a mid-century modern style dining room")
    ap.add_argument("--seed", type=int, default=44)
    a = ap.parse_args()
    if not a.all:
        r = one(a.mode, a.prompt, a.seed)
        sys.exit(0 if r["status"] == "PASS" else 1)
    import pandas as pd
    rows = []
    for mode in ("quick", "v6", "v5"):
        p = subprocess.run([sys.executable, "-u", __file__, "--mode", mode, "--prompt", a.prompt, "--seed", str(a.seed)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        rows.append(json.loads(line[7:]) if line else {"mode": mode, "status": "FAIL (no result)"})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results/tables/v6_app_test.csv", index=False)
    print(t.to_string())
    sys.exit(0 if all(r.get("status") == "PASS" for r in rows) else 1)


if __name__ == "__main__":
    main()
