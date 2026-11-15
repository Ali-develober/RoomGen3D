"""PLAN_V5 milestone 3: real detail in the 2D design shots (re-rendering instead of upscaling only).

Per room (selected candidate of milestone 2, outputs/v5_rooms/selected.json): 4096 panorama (Real-ESRGAN per view, as
v4), v5 shots (hero + walls, src/shots_v5.py), raw crops 1024 x 768 and their z-depth. Configs per shot:
    esrgan           v4 polish-round default: Real-ESRGAN on the crop -> 1536 x 1152
    cn_lora_s<s>     SD 1.5 + MMIS LoRA + depth ControlNet img2img at 768 x 576, strength s -> Real-ESRGAN 1536
    cn_base_s<s>     the same without our LoRA
    xl_s<s>          SDXL base img2img at 1024 x 768 (only if milestone 1 adopted SDXL) -> Real-ESRGAN 1536
Metrics per shot: sharpness (Laplacian variance), BRISQUE, CLIP vs the room prompt, structure kept: edge F1 vs the crop
(structural edges at 512 x 384, src/pano_refine.edge_f1) and depth agreement (Depth Anything relative disparity of
output vs crop, scale + shift aligned, mean error / disparity range); KID / FID vs the MMIS hold-out per config and set.
Selection on dev (fixed in PLAN_V5.md): guard edge F1 >= 0.6 and depth AbsRel <= 0.15; mean rank of BRISQUE,
sharpness, CLIP, KID; report with and without the LoRA.
Usage: python scripts/eval_v5_shots.py --stage crops|render|metrics|report [--sets dev] [--configs ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v5_rooms as rr  # noqa: E402
import v5_common as vc  # noqa: E402
from src import design_shots as ds  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline_v5 as p5  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import shot_render as sr  # noqa: E402
from src import shots_v5 as sv  # noqa: E402

ROOT = vc.ROOT
OUT = ROOT / "outputs" / "v5_shots"
ROOMS = ROOT / "outputs" / "v5_rooms"
DEVICE = torch.device("cuda")
STRENGTHS = [0.35, 0.45, 0.55, 0.65]
XL_STRENGTHS = [0.3, 0.45]


def selected() -> dict:
    return json.loads((ROOMS / "selected.json").read_text())


def pano_path(rid, name):
    gen = p5.source()
    return ROOT / "outputs/v5_pano" / gen / f"{rid}.png" if name == "before" else ROOMS / rid / f"{name}.png"


def stage_crops(df):
    sel = selected()
    todo = [r for r in df.itertuples() if r.id in sel and not (OUT / r.id / "meta.json").exists()]
    if not todo:
        return
    id2label = vc.ade_id2label()
    net = gp.load_esrgan(DEVICE)
    for r in tqdm(todo, desc="crops"):
        n = sel[r.id]
        pano = vc.rgb(pano_path(r.id, n))
        if pano.shape[1] < 2048:
            pano = gp.esrgan_upscale(net, pano)
        hires, _ = pr.refine(pano, net)
        g = np.load(ROOMS / r.id / f"{n}_geo.npz")
        dist = g["dist"]
        ana = rr.load_ana(r.id, n)
        box = rb.fit_box(dist, ana["class_map"], id2label)
        sh = sv.select_shots(hires, box, ana["class_map"], ana["inst_map"], ana["instances"], id2label, dist,
                             ds.room_type_of(r.prompt))
        d = OUT / r.id
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        for s in sh["shots"]:
            vc.save_rgb(d / f"{s['letter']}_raw.png", s["image"])
            np.save(d / f"{s['letter']}_z.npy", sr.shot_depth(dist, box, s["yaw"], s["fov"], 768, 576).astype(np.float32))
            meta.append({"letter": s["letter"], "kind": s["kind"], "yaw": s["yaw"], "fov": s["fov"], "valid": s["valid"]})
        vc.save_rgb(d / "pano_4096.jpg", hires, 92)
        vc.write_json(d / "meta.json", {"name": n, "shots": meta})
    net = None
    mem.free()


def shots_of(df):
    out = []
    for r in df.itertuples():
        mp = OUT / r.id / "meta.json"
        if mp.exists():
            for s in json.loads(mp.read_text())["shots"]:
                out.append((r, s))
    return out


def stage_render(df, configs):
    items = shots_of(df)
    net = gp.load_esrgan(DEVICE)
    # esrgan baseline (cheap)
    for r, s in items:
        p = OUT / r.id / "esrgan" / f"{s['letter']}.jpg"
        if "esrgan" in configs and not p.exists():
            raw = vc.rgb(OUT / r.id / f"{s['letter']}_raw.png")
            t0 = time.perf_counter()
            vc.save_rgb(p, sr.upscale(net, raw), 94)
            vc.write_json(p.with_suffix(".json"), {"seconds": time.perf_counter() - t0})
    groups = {}
    for c in configs:
        if c.startswith("cn_"):
            groups.setdefault(("cn", c.startswith("cn_lora")), []).append(c)
        elif c.startswith("xl_"):
            groups.setdefault(("xl", False), []).append(c)
    for (kind, lora), cs in groups.items():
        todo = [(r, s, c) for c in cs for r, s in items if not (OUT / r.id / c / f"{s['letter']}.jpg").exists()]
        if not todo:
            continue
        net = net.to("cpu")
        mem.free()
        pipe = sr.load_cn_img2img(DEVICE, lora=lora) if kind == "cn" else sr.load_xl_img2img()
        size = (sr.W, sr.H) if kind == "cn" else (1024, 768)
        outs = []
        for r, s, c in tqdm(todo, desc=f"render {kind} lora={lora}"):
            st = float(c.rsplit("_s", 1)[1])
            raw = vc.rgb(OUT / r.id / f"{s['letter']}_raw.png")
            z = np.load(OUT / r.id / f"{s['letter']}_z.npy")
            with mem.PeakMonitor() as pm:
                im = sr.rerender(pipe, raw, z, r.prompt, st, r.seed + ord(s["letter"][0]), *size)
            outs.append((r, s, c, im, pm))
        pipe = None
        mem.free()
        net = net.to(DEVICE)
        for r, s, c, im, pm in tqdm(outs, desc="upscale"):
            t0 = time.perf_counter()
            vc.save_rgb(OUT / r.id / c / f"{s['letter']}.jpg", sr.upscale(net, im), 94)
            vc.write_json(OUT / r.id / c / f"{s['letter']}.json", {"seconds": pm.seconds + time.perf_counter() - t0,
                                                                   "peak_vram_gb": pm.peak_vram_gb,
                                                                   "peak_ram_gb": pm.peak_ram_gb})
    net = None
    mem.free()


def depth_err(a: np.ndarray, b: np.ndarray) -> float:
    """Depth agreement: disparity a (output) aligned to b (crop) by least-squares scale + shift, mean absolute
    difference divided by the crop's disparity range (2-98 %). 0 = same depth layout; 0.15 = off by 15 % of the range.
    (A plain AbsRel is unusable here: relative disparities approach 0 at far points.)"""
    A = np.stack([a.ravel(), np.ones(a.size)], 1)
    x, *_ = np.linalg.lstsq(A, b.ravel(), rcond=None)
    al = a * x[0] + x[1]
    rng = max(float(np.percentile(b, 98) - np.percentile(b, 2)), 1e-6)
    return float(np.mean(np.abs(al - b)) / rng)


def stage_metrics(df, configs):
    from src import depth as dp
    path = OUT / "metrics.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "letter", "config"])
    items = [(r, s, c) for r, s in shots_of(df) for c in configs + ["raw"]
             if ((OUT / r.id / c / f"{s['letter']}.jpg").exists() or c == "raw")
             and not ((have["id"] == r.id) & (have["letter"] == s["letter"]) & (have["config"] == c)).any()]
    if not items:
        return
    model, proc = dp.load_depth_model("depth_anything", DEVICE)
    rows, dcache = [], {}
    for r, s, c in tqdm(items, desc="metrics"):
        raw = vc.rgb(OUT / r.id / f"{s['letter']}_raw.png")
        im = raw if c == "raw" else vc.rgb(OUT / r.id / c / f"{s['letter']}.jpg")
        im15 = cv2.resize(im, (1536, 1152), interpolation=cv2.INTER_CUBIC) if im.shape[1] != 1536 else im
        key = (r.id, s["letter"])
        if key not in dcache:
            dcache[key] = dp.predict_disparity(model, proc, cv2.resize(raw, (768, 576), interpolation=cv2.INTER_AREA), DEVICE)
        dout = dp.predict_disparity(model, proc, cv2.resize(im, (768, 576), interpolation=cv2.INTER_AREA), DEVICE)
        j = OUT / r.id / c / f"{s['letter']}.json"
        rows.append({"id": r.id, "set": r.set, "letter": s["letter"], "kind": s["kind"], "config": c,
                     "sharpness": vc.sharpness([im15]), "brisque": vc.brisque([cv2.resize(im15, (768, 576), interpolation=cv2.INTER_AREA)]),
                     "brisque_full": vc.brisque([im15]),
                     "edge_f1": pr.edge_f1(raw, im, size=(512, 384)), "depth_err": depth_err(dout, dcache[key]),
                     **(json.loads(j.read_text()) if j.exists() else {})})
    model = None
    mem.free()
    new = pd.DataFrame(rows)
    clip_m, cp = ev.load_clip(DEVICE)
    paths = [str(OUT / x.id / f"{x.letter}_raw.png") if x.config == "raw" else str(OUT / x.id / x.config / f"{x.letter}.jpg")
             for x in new.itertuples()]
    prm = df.set_index("id")["prompt"]
    new["clip"] = ev.clip_scores(clip_m, cp, paths, [prm[i] for i in new["id"]], DEVICE)
    clip_m = None
    mem.free()
    pd.concat([have, new], ignore_index=True).to_csv(path, index=False)


def report(df, tag="v5_shots"):
    met = pd.read_csv(OUT / "metrics.csv")
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    rows = []
    for split, g in met.groupby("set"):
        sets = {}
        for c, gc in g.groupby("config"):
            sets[c] = [str(OUT / x.id / f"{x.letter}_raw.png") if c == "raw" else str(OUT / x.id / c / f"{x.letter}.jpg")
                       for x in gc.itertuples()]
        fk = ev.fid_kid(list(hold["real_path"]), sets, DEVICE)
        rows += [{"set": split, "config": k, "kid_x1e3": v["kid_mean"] * 1e3, "kid_std_x1e3": v["kid_std"] * 1e3,
                  "fid": v["fid"]} for k, v in fk.items()]
    kid = pd.DataFrame(rows)
    cols = ["sharpness", "brisque", "brisque_full", "clip", "edge_f1", "depth_err", "seconds", "peak_vram_gb", "peak_ram_gb"]
    summ = met.groupby(["set", "config"])[[c for c in cols if c in met]].mean().reset_index().merge(kid, on=["set", "config"])
    summ["n"] = met.groupby(["set", "config"]).size().values
    summ.to_csv(vc.TAB / f"{tag}_summary.csv", index=False, float_format="%.4f")
    dev = summ[(summ["set"] == "dev") & ~summ["config"].isin(["raw"])].set_index("config")
    ok = dev[(dev["edge_f1"] >= 0.6) & (dev["depth_err"] <= 0.15)].copy()
    rk = pd.DataFrame({"brisque": ok["brisque"].rank(), "sharp": ok["sharpness"].rank(ascending=False),
                       "clip": ok["clip"].rank(ascending=False), "kid": ok["kid_x1e3"].rank()})
    ok["mean_rank"] = rk.mean(1)
    best = ok["mean_rank"].idxmin()
    dec = {"selected": best, "mean_rank": ok["mean_rank"].sort_values().to_dict(),
           "failed_guard": sorted(set(dev.index) - set(ok.index)),
           "rule": "guard edge F1 >= 0.6 and depth error <= 0.15 of the disparity range on dev; mean rank of BRISQUE (768 px), sharpness, CLIP, KID"}
    if best.startswith("cn_lora"):
        twin = best.replace("cn_lora", "cn_base")
        if twin in dev.index:
            dec["lora_vs_base_dev"] = {c: [float(dev.loc[best, c]), float(dev.loc[twin, c])]
                                       for c in ("brisque", "sharpness", "clip", "kid_x1e3", "edge_f1", "depth_err")}
    vc.write_json(vc.TAB / f"{tag}_decision.json", dec)
    met.to_csv(vc.TAB / f"{tag}_per_shot.csv", index=False, float_format="%.4f")
    print(summ.round(3).to_string())
    print(dec)
    return summ, dec


def figure(df, configs, tag="v5_shots", n=4):
    items = [(r, s) for r, s in shots_of(df) if s["kind"] == "hero"][:n]
    cols = ["raw"] + configs
    fig, axes = plt.subplots(len(items), len(cols), figsize=(4.2 * len(cols), 3.4 * len(items)), dpi=90, squeeze=False)
    for i, (r, s) in enumerate(items):
        for j, c in enumerate(cols):
            p = OUT / r.id / f"{s['letter']}_raw.png" if c == "raw" else OUT / r.id / c / f"{s['letter']}.jpg"
            ax = axes[i, j]
            if p.exists():
                im = vc.rgb(p)
                h, w = im.shape[:2]
                ax.imshow(im[h // 4: h // 4 + h // 2, w // 4: w // 4 + w // 2])    # centre crop at 100 %-ish
            ax.set_title(f"{c}" + (f"\n{r.id[:34]}" if j == 0 else ""), fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(vc.FIG / f"{tag}_crops.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--sets", nargs="+", default=["dev"])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = vc.rooms(a.sets)
    if a.limit:
        df = df.groupby("set").head(a.limit).reset_index(drop=True)
    configs = a.configs or (["esrgan"] + [f"cn_{l}_s{s}" for l in ("lora", "base") for s in STRENGTHS]
                            + ([f"xl_s{s}" for s in XL_STRENGTHS] if p5.source().startswith("xl") else []))
    if a.stage in ("all", "crops"):
        stage_crops(df)
    if a.stage in ("all", "render"):
        stage_render(df, configs)
    if a.stage in ("all", "metrics"):
        stage_metrics(df, configs)
    if a.stage in ("all", "report"):
        report(vc.rooms(["dev", "test", "demo"]))
        figure(df, configs)


if __name__ == "__main__":
    main()
