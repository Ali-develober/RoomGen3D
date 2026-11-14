"""PLAN_V3 test bench: layout-conditioned panoramas with ground-truth geometry.

40 rooms = 5 room types x layout seeds 1-8. Seeds 1-4 are the DEVELOPMENT set (all tuning), seeds 5-8
the HELD-OUT set (reported once). Per room, in outputs/bench/<room>_L<seed>/:
    layout.json            the parametric layout (src/layout.py): room size, furniture boxes, openings, camera
    gt_dist.npy            exact equirectangular distance (m) from the camera, 1024 x 512
    gt_labels.png          ADE20K class per pixel (uint8)
    gt_inst.png            furniture box index + 1 per pixel (0 = room shell), uint8
    pano.png               ControlNet-depth panorama (SD 1.5 + pano LoRA, circular padding), milestone 6 settings
    texture.png            Real-ESRGAN x2 (2048 x 1024)
    dist.npy               DAP + wrap fix distance of the generated panorama (what the 360 pipeline sees)
    seg_v2.png             SegFormer on 14 views, labels projected back (the v2 segmentation)
Index: outputs/bench/index.csv and results/tables/bench_v3_rooms.csv. Resumable (skips existing files).
Usage: python scripts/make_bench.py [--limit N]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import layout as ly  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "bench"
ROOM_TYPES = ["bedroom", "living room", "kitchen", "dining room", "bathroom"]
STYLES = ["contemporary", "coastal", "rustic", "mid-century modern", "art deco", "mediterranean", "scandinavian",
          "industrial"]
LAYOUT_SEEDS = list(range(1, 9))
DEV_SEEDS = {1, 2, 3, 4}


def rooms() -> pd.DataFrame:
    rows = []
    for ti, room in enumerate(ROOM_TYPES):
        for s in LAYOUT_SEEDS:
            style = STYLES[(s + 3 * ti) % len(STYLES)]
            prompt = f"a{'n' if style[0] in 'aeiou' else ''} {style} style {room}"
            rows.append({"id": f"{room.replace(' ', '_')}_L{s}", "room": room, "layout_seed": s, "style": style,
                         "prompt": prompt, "gen_seed": 1000 + 10 * s + ti,
                         "split": "dev" if s in DEV_SEEDS else "test"})
    return pd.DataFrame(rows)


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def save_rgb(p, a):
    cv2.imwrite(str(p), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def load_gt(d: Path) -> dict:
    """Ground truth of one bench room."""
    import json
    return {"layout": json.loads((d / "layout.json").read_text()), "dist": np.load(d / "gt_dist.npy"),
            "labels": cv2.imread(str(d / "gt_labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
            "inst": cv2.imread(str(d / "gt_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    df = rooms()
    if args.limit:
        df = df.groupby("room").head(1).head(args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        d = OUT / r.id
        d.mkdir(exist_ok=True)
        if not (d / "gt_inst.png").exists():
            L = ly.generate_layout(r.room, r.layout_seed)
            ly.save_json(L, d / "layout.json")
            dist, lab, inst = ly.render_full(L)
            np.save(d / "gt_dist.npy", dist)
            cv2.imwrite(str(d / "gt_labels.png"), lab.astype(np.uint8))
            cv2.imwrite(str(d / "gt_inst.png"), (inst + 1).astype(np.uint8))

    todo = [r for r in df.itertuples() if not (OUT / r.id / "pano.png").exists()]
    if todo:
        pipe = gp.load_controlnet_pipeline(device)
        for r in tqdm(todo, desc="panoramas", unit="room"):
            c = ly.control_image(np.load(OUT / r.id / "gt_dist.npy"))
            save_rgb(OUT / r.id / "pano.png", gp.generate_pano_controlled(pipe, r.prompt, r.gen_seed, c, 1.0))
        del pipe
        ev.release_memory()
    todo = [r for r in df.itertuples() if not (OUT / r.id / "texture.png").exists()]
    if todo:
        net = gp.load_esrgan(device)
        for r in tqdm(todo, desc="Real-ESRGAN", unit="room"):
            save_rgb(OUT / r.id / "texture.png", gp.esrgan_upscale(net, rgb(OUT / r.id / "pano.png")))
        del net
        ev.release_memory()
    todo = [r for r in df.itertuples() if not (OUT / r.id / "dist.npy").exists()]
    if todo:
        dap, _ = pdp.load_dap(device)
        for r in tqdm(todo, desc="DAP", unit="room"):
            np.save(OUT / r.id / "dist.npy", pdp.dap_depth_wrap(dap, rgb(OUT / r.id / "pano.png"), device))
        del dap
        ev.release_memory()
    todo = [r for r in df.itertuples() if not (OUT / r.id / "seg_v2.png").exists()]
    if todo:
        seg = an.load_segmenter(device)
        for r in tqdm(todo, desc="SegFormer", unit="room"):
            lab = pdp.segment_pano(*seg, rgb(OUT / r.id / "pano.png"), device)
            cv2.imwrite(str(OUT / r.id / "seg_v2.png"), lab.astype(np.uint8))
        del seg
        ev.release_memory()

    full = rooms()
    full["built"] = [(OUT / i / "seg_v2.png").exists() for i in full["id"]]
    full.to_csv(OUT / "index.csv", index=False)
    if full["built"].all():
        full.drop(columns="built").to_csv(ROOT / "results/tables/bench_v3_rooms.csv", index=False)
    print(full.groupby("split")["built"].sum().to_string())


if __name__ == "__main__":
    main()
