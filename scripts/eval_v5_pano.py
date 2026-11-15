"""PLAN_V5 milestone 1: a better panorama source. SDXL (2048 x 1024, two 360° adapters) vs the current generator.

Configs (same prompts and seeds; rooms from v5_common: 10 dev, 10 held-out, 6 demo):
    sd21        current v4 source: SD 2.1 panorama model, one-room template, 1024 x 512 -> Real-ESRGAN x2 -> 2048
    xl_redmond  SDXL base + 360Redmond LoRA, 2048 x 1024 native, circular padding, CPU offload
    xl_pgg      SDXL with the ProGamerGov sdxl-360-diffusion UNet, 2048 x 1024 native, circular padding, CPU offload
Stages (resumable; one model in memory at a time):
    gen       panoramas -> outputs/v5_pano/<config>/<id>.png (+ .json: seconds, peak VRAM, peak RAM)
    esrgan    sd21 only: 1024 -> 2048 texture
    geometry  DAP distance + SegFormer classes -> boxiness, room area, nearest-furniture distance
    views     8 level views (90°, 768 x 576) per panorama -> sharpness, BRISQUE, seam ratio
    clip      CLIP score of the views vs the prompt
    report    KID / FID of the views vs the 500 MMIS hold-out photos per config and set; decision on dev
Decision rule (PLAN_V5.md, fixed before the run): drop configs that fail (crash / OOM / RAM > 15 GB) or have a median
seam ratio > 1.5; then mean rank on dev of KID, CLIP, BRISQUE, sharpness, boxiness; ties -> faster.
Usage: python scripts/eval_v5_pano.py --configs sd21 xl_redmond [--sets dev] [--limit 2] [--stage gen]
"""

import argparse
import json
import sys
import time
import traceback
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
from src import boxiness as bx  # noqa: E402
from src import camera_v5 as cam  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import room_box as rb  # noqa: E402
from src.utils import set_seed  # noqa: E402

ROOT = vc.ROOT
OUT = ROOT / "outputs" / "v5_pano"
DEVICE = torch.device("cuda")
CONFIGS = ["sd21", "xl_redmond", "xl_pgg"]


def stage_gen(df, configs, template=None, negative=None, out=OUT, steps=None):
    for cfg in configs:
        d = out / cfg
        d.mkdir(parents=True, exist_ok=True)
        todo = [r for r in df.itertuples() if not (d / f"{r.id}.json").exists()]
        if not todo:
            continue
        mem.free()
        try:
            if cfg == "sd21":
                pipe = gp.load_alt_pipeline(DEVICE)
            else:
                from src import generate_pano_xl as gx
                pipe = gx.load_xl_pipeline(cfg.split("_", 1)[1])
        except Exception as e:                       # noqa: BLE001
            (d / "_load_error.txt").write_text(traceback.format_exc())
            print(f"{cfg}: load failed: {e}")
            continue
        for r in tqdm(todo, desc=f"gen {cfg}", unit="pano"):
            mem.wait_for_ram(1.0, label=cfg)
            try:
                with mem.PeakMonitor() as pm:
                    if cfg == "sd21":
                        tpl = template or gp.ONE_ROOM_TEMPLATE
                        img = gp.generate_pano(pipe, r.prompt, r.seed, prompt=tpl.format(room=r.prompt),
                                               negative=negative or gp.ONE_ROOM_NEGATIVE)
                    else:
                        from src import generate_pano_xl as gx
                        v = cfg.split("_", 1)[1]
                        p = gx.xl_prompt(r.prompt, v, template) if template else None
                        img = gx.generate_pano_xl(pipe, r.prompt, r.seed, prompt=p, negative=negative,
                                                  **({"steps": steps} if steps else {}))
                name = f"{r.id}_native.png" if cfg == "sd21" else f"{r.id}.png"
                vc.save_rgb(d / name, img)
                vc.write_json(d / f"{r.id}.json", {"seconds": pm.seconds, "peak_vram_gb": pm.peak_vram_gb,
                                                   "peak_ram_gb": pm.peak_ram_gb, "ok": True,
                                                   "ram_free_gb": mem.ram_free_gb()})
            except torch.OutOfMemoryError:
                vc.write_json(d / f"{r.id}.json", {"ok": False, "error": "CUDA OOM"})
                mem.free()
        pipe = None
        mem.free()


def stage_esrgan(df, out=OUT):
    d = out / "sd21"
    todo = [r for r in df.itertuples() if (d / f"{r.id}_native.png").exists() and not (d / f"{r.id}.png").exists()]
    if not todo:
        return
    net = gp.load_esrgan(DEVICE)
    for r in tqdm(todo, desc="esrgan"):
        t0 = time.perf_counter()
        tex = gp.esrgan_upscale(net, vc.rgb(d / f"{r.id}_native.png"))
        vc.save_rgb(d / f"{r.id}.png", tex)
        j = json.loads((d / f"{r.id}.json").read_text())
        j["seconds"] += time.perf_counter() - t0
        vc.write_json(d / f"{r.id}.json", j)
    net = None
    mem.free()


def stage_geometry(df, configs, out=OUT):
    """DAP + SegFormer on every panorama -> geometry.csv rows (boxiness, area, nearest furniture)."""
    from src import analysis as an
    id2label = vc.ade_id2label()
    gpath = out / "geometry.csv"
    have = pd.read_csv(gpath) if gpath.exists() else pd.DataFrame(columns=["config", "id"])
    todo = [(c, r) for c in configs for r in df.itertuples() if (out / c / f"{r.id}.png").exists()
            and not ((have["config"] == c) & (have["id"] == r.id)).any()]
    if not todo:
        return
    dap = pdp.load_dap(DEVICE)[0]
    dists = {}
    for c, r in tqdm(todo, desc="DAP"):
        pano = cv2.resize(vc.rgb(out / c / f"{r.id}.png"), (1024, 512), interpolation=cv2.INTER_AREA)
        dists[(c, r.id)] = pdp.dap_depth_wrap(dap, pano, DEVICE)
        np.save(out / c / f"{r.id}_dist.npy", dists[(c, r.id)])
    dap = None
    mem.free()
    seg = an.load_segmenter(DEVICE)
    rows = []
    for c, r in tqdm(todo, desc="SegFormer + box"):
        pano = cv2.resize(vc.rgb(out / c / f"{r.id}.png"), (1024, 512), interpolation=cv2.INTER_AREA)
        cm = pdp.segment_pano(*seg, pano, DEVICE)
        cv2.imwrite(str(out / c / f"{r.id}_class.png"), cm.astype(np.uint16))
        dist = dists[(c, r.id)]
        try:
            box = rb.fit_box(dist, cm, id2label)
            sc = bx.score(dist, cm, id2label, box)
            dn = cam.nearest_furniture(dist, cm, id2label, box)
            rows.append({"config": c, "id": r.id, "boxiness": sc["score"], "area_m2": sc["area_m2"],
                         "d_near_m": dn, "camera_h": box["camera_height"], "sides": box["n_detected"]})
        except Exception as e:                      # noqa: BLE001
            print(c, r.id, "geometry failed:", e)
            rows.append({"config": c, "id": r.id, "boxiness": np.nan})
    seg = None
    mem.free()
    pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(gpath, index=False)


def stage_views(df, configs, out=OUT):
    vpath = out / "views.csv"
    have = pd.read_csv(vpath) if vpath.exists() else pd.DataFrame(columns=["config", "id"])
    rows = []
    for c in configs:
        for r in tqdm(list(df.itertuples()), desc=f"views {c}"):
            if not (out / c / f"{r.id}.png").exists() or ((have["config"] == c) & (have["id"] == r.id)).any():
                continue
            pano = vc.rgb(out / c / f"{r.id}.png")
            views = vc.level_views(pano)
            vd = out / c / "views"
            for k, v in enumerate(views):
                vc.save_rgb(vd / f"{r.id}_v{k}.png", cv2.resize(v, (512, 384), interpolation=cv2.INTER_AREA))
            rows.append({"config": c, "id": r.id, "sharpness": vc.sharpness(views), "brisque": vc.brisque(views),
                         "seam_ratio": eq.seam_ratio(pano)})
    if rows:
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(vpath, index=False)


def stage_clip(df, configs, out=OUT):
    model, proc = ev.load_clip(DEVICE)
    rows = []
    for c in configs:
        for r in df.itertuples():
            paths = [str(out / c / "views" / f"{r.id}_v{k}.png") for k in range(len(vc.VIEW_YAWS))]
            if not Path(paths[0]).exists():
                continue
            rows.append({"config": c, "id": r.id,
                         "clip": float(ev.clip_scores(model, proc, paths, [r.prompt] * len(paths), DEVICE).mean())})
    model = None
    mem.free()
    pd.DataFrame(rows).to_csv(out / "clip.csv", index=False)


def per_room(df, configs, out=OUT) -> pd.DataFrame:
    rows = []
    for c in configs:
        for r in df.itertuples():
            j = out / c / f"{r.id}.json"
            if j.exists():
                rows.append({"config": c, "id": r.id, "set": r.set, **json.loads(j.read_text())})
    per = pd.DataFrame(rows)
    for f in ("geometry.csv", "views.csv", "clip.csv"):
        if (out / f).exists():
            per = per.merge(pd.read_csv(out / f), on=["config", "id"], how="left")
    return per


def kid(per: pd.DataFrame, out=OUT) -> pd.DataFrame:
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    rows = []
    for split in ("dev", "test", "demo"):
        sets = {}
        for c, g in per[(per["set"] == split) & per["ok"].astype(bool)].groupby("config"):
            sets[c] = [str(out / c / "views" / f"{i}_v{k}.png") for i in g["id"] for k in range(len(vc.VIEW_YAWS))
                       if (out / c / "views" / f"{i}_v{k}.png").exists()]
        if not sets:
            continue
        fk = ev.fid_kid(list(hold["real_path"]), sets, DEVICE)
        rows += [{"config": k, "set": split, "kid_x1e3": v["kid_mean"] * 1e3, "kid_std_x1e3": v["kid_std"] * 1e3,
                  "fid": v["fid"]} for k, v in fk.items()]
    return pd.DataFrame(rows)


COLS = ["seconds", "peak_vram_gb", "peak_ram_gb", "boxiness", "area_m2", "d_near_m", "camera_h", "sharpness",
        "brisque", "seam_ratio", "clip"]


def decide(summ: pd.DataFrame) -> dict:
    dev = summ[summ["set"] == "dev"].set_index("config")
    ok = dev[(dev["failures"] == 0) & (dev["peak_ram_gb"] < 15) & (dev["seam_ratio_median"] <= 1.5)].copy()
    ranks = pd.DataFrame({"kid": ok["kid_x1e3"].rank(), "clip": ok["clip"].rank(ascending=False),
                          "brisque": ok["brisque"].rank(), "sharpness": ok["sharpness"].rank(ascending=False),
                          "boxiness": ok["boxiness"].rank(ascending=False)})
    ok["mean_rank"] = ranks.mean(1)
    ok = ok.sort_values(["mean_rank", "seconds"])
    return {"selected": ok.index[0] if len(ok) else "sd21", "mean_rank": ok["mean_rank"].to_dict(),
            "excluded": sorted(set(dev.index) - set(ok.index)),
            "rule": "drop failures / RAM >= 15 GB / median seam ratio > 1.5; mean rank on dev of KID, CLIP, BRISQUE, "
                    "sharpness, boxiness; ties -> faster"}


def report(df, configs, out=OUT, tag="v5_pano"):
    per = per_room(df, configs, out)
    per.to_csv(vc.TAB / f"{tag}_per_room.csv", index=False)
    g = per.groupby(["config", "set"])
    summ = g[[c for c in COLS if c in per]].mean()
    summ["seam_ratio_median"] = g["seam_ratio"].median()
    summ["failures"] = g["ok"].apply(lambda s: int((~s.astype(bool)).sum()))
    summ["n"] = g.size()
    summ = summ.reset_index().merge(kid(per, out), on=["config", "set"], how="left")
    summ.to_csv(vc.TAB / f"{tag}_summary.csv", index=False, float_format="%.4f")
    dec = decide(summ)
    vc.write_json(vc.TAB / f"{tag}_decision.json", dec)
    print(summ.round(3).to_string())
    print(dec)
    figure(df, configs, out, tag)
    return summ, dec


def figure(df, configs, out=OUT, tag="v5_pano", n=4):
    ids = [r.id for r in df.itertuples() if r.set in ("dev", "demo")][:: max(1, len(df) // n)][:n]
    fig, axes = plt.subplots(len(ids), len(configs), figsize=(6.5 * len(configs), 3.6 * len(ids)), dpi=80, squeeze=False)
    for i, rid in enumerate(ids):
        for j, c in enumerate(configs):
            p = out / c / "views" / f"{rid}_v0.png"
            ax = axes[i, j]
            if p.exists():
                v = vc.rgb(out / c / f"{rid}.png")
                ax.imshow(vc.level_views(v, 768, 576, 90.0, [0])[0])
            ax.set_title(f"{c} - {rid[:40]}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(vc.FIG / f"{tag}_views.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "demo"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    set_seed(42)
    df = vc.rooms(a.sets)
    if a.limit:
        df = df.groupby("set").head(a.limit).reset_index(drop=True)
    st = a.stage
    if st in ("all", "gen"):
        stage_gen(df, a.configs)
    if st in ("all", "esrgan"):
        stage_esrgan(df)
    if st in ("all", "geometry"):
        stage_geometry(df, a.configs)
    if st in ("all", "views"):
        stage_views(df, a.configs)
    if st in ("all", "clip"):
        stage_clip(df, a.configs)
    if st in ("all", "report"):
        report(vc.rooms(a.sets), a.configs)


if __name__ == "__main__":
    main()
