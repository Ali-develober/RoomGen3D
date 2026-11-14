"""360 degree room mode on the six demo prompts (same prompts and seeds as the Quick mode).

One model on the GPU at a time: all panoramas, then super-resolution, then depth, then segmentation;
the mesh, the dollhouse variant and the floor plan run on the CPU. Resumable: every stage skips
rooms whose output file exists.

Outputs:
  outputs/room360/<prompt>_s<seed>/  pano.png, texture.png, distance.npy, class_map.png,
                                     room.glb, dollhouse.glb, floorplan.png / .svg   (not in git)
  results/tables/room360_timing.csv  seconds per stage and room, peak VRAM per GPU stage, mesh sizes
Usage: python scripts/run_room360.py [--limit 2]
"""

import argparse
import json
import sys
import time
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
from src import pano_depth as pdp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "room360"


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def gpu_stage(name, loader, items, fn, out_name, saver, timing, vram):
    todo = [(p, s, d) for p, s, d in items if not (d / out_name).exists()]
    if not todo:
        return
    ev.release_memory()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = loader()
    torch.cuda.synchronize()
    timing.setdefault("_load", {})[name] = time.perf_counter() - t0
    for prompt, seed, d in tqdm(todo, desc=name, unit="room"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = fn(model, prompt, seed, d)
        torch.cuda.synchronize()
        timing.setdefault(d.name, {})[name] = time.perf_counter() - t0
        saver(res, d)
        tf = d / "timing.json"                       # per-room record, so a resumed run keeps the times
        rec = json.loads(tf.read_text()) if tf.exists() else {}
        rec[name] = timing[d.name][name]
        tf.write_text(json.dumps(rec))
    vram[name] = torch.cuda.max_memory_allocated() / 2 ** 30
    (OUT / f"vram_{name}.json").write_text(json.dumps({"gb": vram[name], "load_s": timing["_load"][name]}))
    del model
    ev.release_memory()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    prompts = pl.DEMO_PROMPTS[: args.limit] if args.limit else pl.DEMO_PROMPTS
    items = [(p, s, OUT / pl.slug(p, s)) for p, s in prompts]
    for _, _, d in items:
        d.mkdir(parents=True, exist_ok=True)
    timing, vram = {}, {}
    save_img = lambda name: (lambda a, d: cv2.imwrite(str(d / name), cv2.cvtColor(a, cv2.COLOR_RGB2BGR)))  # noqa: E731

    gpu_stage("panorama", lambda: gp.load_alt_pipeline(device), items,
              lambda m, p, s, d: p3.generate(m, p, s), "pano.png", save_img("pano.png"), timing, vram)
    gpu_stage("super_res", lambda: gp.load_esrgan(device), items,
              lambda m, p, s, d: p3.super_resolve(m, rgb(d / "pano.png")), "texture.png", save_img("texture.png"),
              timing, vram)
    gpu_stage("depth", lambda: pdp.load_dap(device)[0], items,
              lambda m, p, s, d: p3.distance(m, rgb(d / "pano.png"), device), "distance.npy",
              lambda a, d: np.save(d / "distance.npy", a), timing, vram)
    seg_holder = {}

    def load_seg():
        seg_holder["m"] = an.load_segmenter(device)
        return seg_holder["m"]
    gpu_stage("segmentation", load_seg, items,
              lambda m, p, s, d: p3.segment(m, rgb(d / "pano.png"), device), "class_map.png",
              lambda a, d: cv2.imwrite(str(d / "class_map.png"), a.astype(np.uint8)), timing, vram)
    from transformers import AutoConfig
    id2label = AutoConfig.from_pretrained(an.SEG_MODEL).id2label

    rows = []
    for prompt, seed, d in tqdm(items, desc="mesh + plan", unit="room"):
        dist = np.load(d / "distance.npy")
        cm = cv2.imread(str(d / "class_map.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        t0 = time.perf_counter()
        res = p3.build(dist, rgb(d / "texture.png"), cm, id2label)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        p3.save(res, d, plan_title=prompt)
        t_save = time.perf_counter() - t0
        st = res["mesh"]["stats"]
        tf = d / "timing.json"
        t = json.loads(tf.read_text()) if tf.exists() else {}
        rows.append({"prompt": prompt, "seed": seed, **{k: t.get(k, np.nan) for k in ("panorama", "super_res", "depth", "segmentation")},
                     "mesh_plan_s": t_build, "export_s": t_save, "triangles": st["triangles"],
                     "dollhouse_triangles": res["dollhouse"]["stats"]["dollhouse_triangles"],
                     "box_source": res["mesh"]["box"]["source"], "camera_height_m": res["mesh"]["camera_height"],
                     "ceiling_height_m": res["mesh"]["box"]["ceiling_height"], "box_area_m2": res["mesh"]["box"]["area"],
                     "glb_mb": (d / "room.glb").stat().st_size / 2 ** 20, **fp.plan_summary(res["plan"])})
    df = pd.DataFrame(rows)
    gpu = [c for c in ("panorama", "super_res", "depth", "segmentation") if df[c].notna().all()]
    if gpu:
        df["total_s"] = df[gpu + ["mesh_plan_s", "export_s"]].sum(1)
    for k in ("panorama", "super_res", "depth", "segmentation"):
        f = OUT / f"vram_{k}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            df[f"vram_{k}_gb"], df[f"load_{k}_s"] = rec["gb"], rec["load_s"]
    tag = "" if args.limit is None else "_smoke"
    df.to_csv(ROOT / f"results/tables/room360_timing{tag}.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(df.round(2).T.to_string())


if __name__ == "__main__":
    main()
