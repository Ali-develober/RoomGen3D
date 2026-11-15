"""PLAN_V5 milestone 6: Quick-mode hires pass vs the current 512 output.

Prompts: v5_common rooms (dev = scandinavian / farmhouse, held-out = industrial / bohemian + the six demo prompts),
seeds 701 and 702 (demo: its own seed and seed + 1). Configs:
    base512         current Quick output (SD 1.5 + MMIS LoRA, 512), shown at 1024 (bicubic)
    hires_s<s>      + img2img at 1024 with the LoRA, strength s in STRENGTHS
    hires_s<s>_esr  + Real-ESRGAN x2 (2048), compared at 1024 (area downscale)
Metrics at the 1024 display size: sharpness, BRISQUE, CLIP vs the prompt, KID / FID vs the MMIS hold-out; time, VRAM.
Selection on dev (fixed beforehand): mean rank of BRISQUE, sharpness, CLIP, KID among the hires configs.
Usage: python scripts/eval_v5_quick.py [--stage gen|metrics|report]
"""

import argparse
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
import v5_common as vc  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import memory as mem  # noqa: E402
from src import quick_v5 as q5  # noqa: E402

ROOT = vc.ROOT
OUT = ROOT / "outputs" / "v5_quick"
DEVICE = torch.device("cuda")
STRENGTHS = [0.35, 0.4, 0.45]


def items():
    df = vc.rooms()
    out = []
    for r in df.itertuples():
        for k in range(2):
            out.append({"id": f"{r.id}_{k}", "prompt": r.prompt, "seed": r.seed + k, "set": r.set})
    return pd.DataFrame(out)


def stage_gen(df):
    todo = [r for r in df.itertuples() if not (OUT / "done" / f"{r.id}.txt").exists()]
    if not todo:
        return
    pipe = g2.load_pipeline(g2.DEFAULT_LORA, DEVICE)
    i2i = q5.img2img_from(pipe)
    rows = []
    for r in tqdm(todo, desc="quick"):
        with mem.PeakMonitor() as pm0:
            base = g2.generate(pipe, r.prompt, r.seed)
        vc.save_rgb(OUT / "base512" / f"{r.id}.png", base)
        rows.append({"id": r.id, "config": "base512", "seconds": pm0.seconds, "peak_vram_gb": pm0.peak_vram_gb})
        for s in STRENGTHS:
            with mem.PeakMonitor() as pm:
                hi = q5.hires_pass(i2i, base, r.prompt, r.seed, s)
            vc.save_rgb(OUT / f"hires_s{s}" / f"{r.id}.png", hi)
            rows.append({"id": r.id, "config": f"hires_s{s}", "seconds": pm0.seconds + pm.seconds,
                         "peak_vram_gb": max(pm0.peak_vram_gb, pm.peak_vram_gb)})
        (OUT / "done").mkdir(parents=True, exist_ok=True)
        (OUT / "done" / f"{r.id}.txt").write_text("ok")
    pipe = i2i = None
    mem.free()
    net = gp.load_esrgan(DEVICE)
    from src.shot_enhance import esrgan_shot
    for r in tqdm(todo, desc="esrgan"):
        for s in STRENGTHS:
            hi = vc.rgb(OUT / f"hires_s{s}" / f"{r.id}.png")
            t0 = time.perf_counter()
            e = esrgan_shot(net, hi, 2048, 2048)[0]
            dt = time.perf_counter() - t0
            vc.save_rgb(OUT / f"hires_s{s}_esr" / f"{r.id}.jpg", e, 95)
            base_s = [x for x in rows if x["id"] == r.id and x["config"] == f"hires_s{s}"][0]
            rows.append({**base_s, "config": f"hires_s{s}_esr", "seconds": base_s["seconds"] + dt})
    net = None
    mem.free()
    t = OUT / "timing.csv"
    old = pd.read_csv(t) if t.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(t, index=False)


def path_of(c, rid):
    ext = "jpg" if c.endswith("_esr") else "png"
    return OUT / c / f"{rid}.{ext}"


def configs():
    return ["base512"] + [f"hires_s{s}" for s in STRENGTHS] + [f"hires_s{s}_esr" for s in STRENGTHS]


def at1024(im):
    if im.shape[1] == 1024:
        return im
    return cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_CUBIC if im.shape[1] < 1024 else cv2.INTER_AREA)


def stage_metrics(df):
    rows = []
    disp = OUT / "display"
    for c in configs():
        for r in tqdm(list(df.itertuples()), desc=f"metrics {c}"):
            p = path_of(c, r.id)
            if not p.exists():
                continue
            im = at1024(vc.rgb(p))
            q = disp / c / f"{r.id}.jpg"
            if not q.exists():
                vc.save_rgb(q, cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA), 95)
            rows.append({"id": r.id, "set": r.set, "config": c, "sharpness": vc.sharpness([im]),
                         "brisque": vc.brisque([im])})
    met = pd.DataFrame(rows)
    clip_m, cp = ev.load_clip(DEVICE)
    prm = df.set_index("id")["prompt"]
    met["clip"] = ev.clip_scores(clip_m, cp, [str(path_of(c, i)) for c, i in zip(met["config"], met["id"])],
                                 [prm[i] for i in met["id"]], DEVICE)
    clip_m = None
    mem.free()
    tim = pd.read_csv(OUT / "timing.csv")
    met = met.merge(tim, on=["id", "config"], how="left")
    met.to_csv(vc.TAB / "v5_quick_per_image.csv", index=False, float_format="%.4f")


def report():
    met = pd.read_csv(vc.TAB / "v5_quick_per_image.csv")
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    kid = []
    for split in ("dev", "test", "demo"):
        g = met[met["set"] == split]
        sets = {c: [str(OUT / "display" / c / f"{i}.jpg") for i in gc["id"]] for c, gc in g.groupby("config")}
        fk = ev.fid_kid(list(hold["real_path"]), sets, DEVICE)
        kid += [{"set": split, "config": k, "kid_x1e3": v["kid_mean"] * 1e3, "fid": v["fid"]} for k, v in fk.items()]
    summ = met.groupby(["set", "config"])[["sharpness", "brisque", "clip", "seconds", "peak_vram_gb"]].mean().reset_index()
    summ = summ.merge(pd.DataFrame(kid), on=["set", "config"])
    summ.to_csv(vc.TAB / "v5_quick_summary.csv", index=False, float_format="%.4f")
    dev = summ[(summ["set"] == "dev") & (summ["config"] != "base512")].set_index("config")
    rk = pd.DataFrame({"b": dev["brisque"].rank(), "s": dev["sharpness"].rank(ascending=False),
                       "c": dev["clip"].rank(ascending=False), "k": dev["kid_x1e3"].rank()})
    dev["mean_rank"] = rk.mean(1)
    best = dev["mean_rank"].idxmin()
    vc.write_json(vc.TAB / "v5_quick_decision.json", {"selected": best, "mean_rank": dev["mean_rank"].to_dict(),
                                                     "rule": "dev: mean rank of BRISQUE, sharpness, CLIP, KID (hires configs)"})
    # figure: base vs selected, centre crops at 100 % of the 1024 display
    ids = met[met["set"] == "demo"]["id"].unique()[:4]
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 5 * len(ids)), dpi=80, squeeze=False)
    for i, rid in enumerate(ids):
        for j, c in enumerate(["base512", best]):
            im = at1024(vc.rgb(path_of(c, rid)))
            axes[i, j].imshow(im[256:768, 256:768])
            axes[i, j].set_title(f"{c} (centre 512 px of the 1024 view) - {rid[:30]}", fontsize=9)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(vc.FIG / "v5_quick_crops.png", facecolor="white")
    plt.close(fig)
    print(summ.round(3).to_string())
    print("selected", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    df = items()
    if a.stage in ("all", "gen"):
        stage_gen(df)
    if a.stage in ("all", "metrics"):
        stage_metrics(df)
    if a.stage in ("all", "report"):
        report()


if __name__ == "__main__":
    main()
