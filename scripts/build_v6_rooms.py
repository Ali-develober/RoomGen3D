"""Build v6 rooms with the full pipeline (src/pipeline_v6.run) -> outputs/v6_demo/<slug>/ (resumable: a room with
timing.json is skipped unless --force). Default rooms: the six demo prompts (incl. the owner's dining room seed 44 and
coastal bedroom seed 22). Usage: python scripts/build_v6_rooms.py [--rooms "prompt|seed;prompt|seed"] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import memory as mem  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v6 as P6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="")
    ap.add_argument("--out", default="outputs/v6_demo")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rooms = [(r.split("|")[0], int(r.split("|")[1])) for r in a.rooms.split(";")] if a.rooms else pl.DEMO_PROMPTS
    for prompt, seed in rooms:
        d = ROOT / a.out / pl.slug(prompt, seed)
        if (d / "timing.json").exists() and not a.force:
            print("skip", d.name)
            continue
        r = P6.run(prompt, seed, d)
        t = json.loads((d / "timing.json").read_text())
        print(d.name, {k: round(v, 1) for k, v in t.items()}, flush=True)
        del r
        mem.free()


if __name__ == "__main__":
    main()
