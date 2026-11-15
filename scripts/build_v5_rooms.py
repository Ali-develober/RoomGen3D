"""Build complete v5 rooms (PLAN_V5) for the demo rooms (or any v5_common set) from the milestone-2 selection.

Per room (selected candidate in outputs/v5_rooms/selected.json, its DAP / SegFormer / Mask2Former results):
    A  4096 panorama (milestone-3 crops directory if present, else Real-ESRGAN per view), room build, v5 shots
    B  shot re-rendering with the milestone-3 choice (results/tables/v5_shots_decision.json) + Real-ESRGAN
    C  clean dollhouse (results/tables/v5_doll_decision.json), v4 inside mesh = "Experimental: walkable mesh",
       viewer page, plan with view cones, files -> outputs/v5_demo/<slug>/ (demo) or outputs/v5_built/<id>/
One model in memory at a time; per-stage timing, peak VRAM and peak RAM per room -> timing.json.
Usage: python scripts/build_v5_rooms.py [--sets demo] [--limit 1]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v5_rooms as rr  # noqa: E402
import v5_common as vc  # noqa: E402
from src import dollhouse_v5 as dv  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v4 as p4v  # noqa: E402
from src import pipeline_v5 as p5  # noqa: E402
from src import room_v4 as r4  # noqa: E402
from src import shot_render as sr  # noqa: E402

ROOT = vc.ROOT
DEVICE = torch.device("cuda")


def out_dir(r):
    return ROOT / "outputs/v5_demo" / r.id[len("demo_"):] if r.set == "demo" else ROOT / "outputs/v5_built" / r.id


def shot_config():
    d = p5.decision("v5_shots_decision.json") or {}
    return d.get("selected", "cn_lora_s0.45")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["demo"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = vc.rooms(a.sets)
    if a.limit:
        df = df.head(a.limit)
    sel = json.loads((ROOT / "outputs/v5_rooms/selected.json").read_text())
    id2label = vc.ade_id2label()
    rooms, timing = {}, {}
    # A: 4096 panorama, room, shots
    net = gp.load_esrgan(DEVICE)
    for r in tqdm(list(df.itertuples()), desc="A room + shots"):
        n = sel[r.id]
        t0 = time.perf_counter()
        crop_pano = ROOT / "outputs/v5_shots" / r.id / "pano_4096.jpg"
        src = vc.rgb(ROOT / "outputs/v5_pano" / p5.source() / f"{r.id}.png") if n == "before" else \
            vc.rgb(ROOT / "outputs/v5_rooms" / r.id / f"{n}.png")
        if src.shape[1] < 2048:
            src = gp.esrgan_upscale(net, src)
        hires = vc.rgb(crop_pano) if crop_pano.exists() else pr.refine(src, net)[0]
        g = np.load(ROOT / "outputs/v5_rooms" / r.id / f"{n}_geo.npz")
        dist = g["dist"]
        ana = rr.load_ana(r.id, n)
        res = p5.build(dist, src, hires, ana, id2label, r.prompt)
        rooms[r.id] = (r, dist, ana, res, n, src)
        timing[r.id] = {"room_shots_s": time.perf_counter() - t0}
    # B: re-render shots
    cfg = shot_config()
    if cfg != "esrgan":
        net = net.to("cpu")
        mem.free()
        lora = cfg.startswith("cn_lora")
        pipe = sr.load_cn_img2img(DEVICE, lora=lora) if cfg.startswith("cn") else sr.load_xl_img2img()
        size = (sr.W, sr.H) if cfg.startswith("cn") else (1024, 768)
        st = float(cfg.rsplit("_s", 1)[1])
        for rid, (r, dist, ana, res, n, src) in tqdm(rooms.items(), desc="B re-render"):
            with mem.PeakMonitor() as pm:
                ims = sr.rerender_shots(pipe, res["shots"]["shots"], dist, res["box"], r.prompt, st, r.seed, size)
            res["_rerendered"] = ims
            timing[rid].update({"rerender_s": pm.seconds, "rerender_vram_gb": pm.peak_vram_gb, "rerender_ram_gb": pm.peak_ram_gb})
        pipe = None
        mem.free()
        net = net.to(DEVICE)
    for rid, (r, dist, ana, res, n, src) in rooms.items():
        t0 = time.perf_counter()
        for s, im in zip(res["shots"]["shots"], res.pop("_rerendered", [s["raw"] for s in res["shots"]["shots"]])):
            s["image"] = sr.upscale(net, im)
        timing[rid]["upscale_s"] = time.perf_counter() - t0
    net = None
    mem.free()
    # C: dollhouse, experimental mesh, save
    dd = p5.decision("v5_doll_decision.json") or {"fill": "lama", "clean": "bilateral"}
    for rid, (r, dist, ana, res, n, src) in tqdm(rooms.items(), desc="C dollhouse + save"):
        with mem.PeakMonitor() as pm:
            res["doll_v5"] = dv.build_dollhouse(dist, res["hires"], res, ana, id2label, DEVICE, dd["fill"], dd["clean"])
            m = r4.build_meshes(dist, res["hires"], res, ana, id2label)
            res["mesh"], res["dollhouse"] = m["inside"], res["doll_v5"]
            files = p5.save(res, out_dir(r), r.prompt)
            vc.save_rgb(out_dir(r) / "pano_4096.jpg", res["hires"], 92)
        timing[rid].update({"doll_mesh_save_s": pm.seconds, "doll_vram_gb": pm.peak_vram_gb, "doll_ram_gb": pm.peak_ram_gb,
                            "candidate": n, "files": files})
        vc.write_json(out_dir(r) / "timing.json", timing[rid])
        res["shots"]["shots"] = None
    dv.free_lama()
    dv.free_sd()
    mem.free()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in timing.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
