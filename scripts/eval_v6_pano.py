"""PLAN_V6 milestone 2: panorama on the proxy scene — sweep of the depth conditioning and gates.

Stages (resumable; one GPU model at a time):
    gen      SDXL + 360Redmond + {cn_small, cn_mid, t2i} at the configs of CONFIGS on the chosen rooms
             -> outputs/v6_pano/<config>/<room id>.jpg (+ .json: seconds, peak VRAM / RAM)
    views    8 level views (yaw 0..315, 90°, 768 x 576) of each panorama + the aligned proxy renders
    depth    Depth Anything (relative) on the views vs the proxy planar depth (scale/shift aligned):
             AbsRel, delta1 (depth agreement gate)
    seg      SegFormer ADE20K on the views: per-instance IoU of the matching classes against the proxy
             instance mask (furniture present where the proxy puts it), window share generated / proxy
    edges    straight structure: recall of proxy structural edges (wall / ceiling / floor / window / door
             boundaries, straight by construction) by Canny edges of the view within 3 px; proxy-free
             straightness = share of Canny edge pixels on long LSD segments (also computed for v5)
    quality  BRISQUE, Laplacian sharpness, seam ratio
    clip     CLIP score of the views vs the user prompt
    report   KID vs the 500 MMIS hold-out photos; summary per config and set; v5 rows (demo panoramas of
             outputs/v5_demo + the v5 dev numbers of results/tables/v5_pano_summary.csv)
Usage: python scripts/eval_v6_pano.py --stage gen --configs cn_small_s0.7_e1.0 --set dev5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import layout_v6 as LV  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pano_v6 as PV  # noqa: E402
from src import raycast_v6 as R  # noqa: E402
from src import scene_v6 as S  # noqa: E402
from src import views_v6 as V  # noqa: E402
import v5_common as C  # noqa: E402

OUT = ROOT / "outputs" / "v6_pano"
DEVICE = "cuda"
YAWS = (0, 45, 90, 135, 180, 225, 270, 315)
VW, VH, VFOV = 768, 576, 90.0


def parse_config(name: str) -> dict:
    """'cn_small_s0.7_e1.0' -> cond cn_small, scale 0.7, end 1.0."""
    parts = name.split("_")
    s = float(next(p[1:] for p in parts if p.startswith("s") and p[1:2].isdigit()))
    e = float(next(p[1:] for p in parts if p.startswith("e") and p[1:2].isdigit()))
    cond = "_".join(p for p in parts if not ((p.startswith("s") or p.startswith("e")) and p[1:2].isdigit()))
    return {"cond": cond, "scale": s, "end": e}


def room_df(which: str) -> pd.DataFrame:
    df = C.rooms()
    if which == "dev5":
        return df[(df.set == "dev") & df.prompt.str.contains("scandinavian")].reset_index(drop=True)
    if which == "all":
        return df
    return df[df.set.isin(which.split(","))].reset_index(drop=True)


def scene_of(prompt: str, seed: int):
    sc = S.build(LV.generate(prompt, seed))
    return sc, S.Renderer(sc)


# --------------------------------------------------------------------------- #
def stage_gen(df, configs):
    by_cond = {}
    for c in configs:
        by_cond.setdefault(parse_config(c)["cond"], []).append(c)
    for cond, cs in by_cond.items():
        todo = [(c, r) for c in cs for r in df.itertuples() if not (OUT / c / f"{r.id}.jpg").exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        pipe = PV.load_pipeline(cond)
        load_s = time.perf_counter() - t0
        print(f"load {cond}: {load_s:.0f} s", flush=True)
        for c, r in todo:
            cfg = parse_config(c)
            sc, rd = scene_of(r.prompt, r.seed)
            lay = sc["layout"]
            e = rd.equirect(lay["camera"], 2048, 1024)
            ctrl = S.depth_control(e["t"])
            prompt = PV.pano_prompt(lay)
            with mem.PeakMonitor() as pm:
                img = PV.generate(pipe, ctrl, prompt, r.seed, scale=cfg["scale"], end=cfg["end"])
            (OUT / c).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(OUT / c / f"{r.id}.jpg", quality=95)
            (OUT / c / f"{r.id}.json").write_text(json.dumps(
                {"seconds": pm.seconds, "peak_vram_gb": pm.peak_vram_gb, "peak_ram_gb": pm.peak_ram_gb,
                 "load_s": load_s, "prompt": prompt, "tokens": PV.n_tokens(prompt)}, indent=1))
            print(c, r.id, f"{pm.seconds:.0f} s", f"{pm.peak_vram_gb:.2f} GB", flush=True)
        mem.free(pipe)
        del pipe
        mem.free()


# --------------------------------------------------------------------------- #
def views_of(cfg: str, r) -> tuple[list, list]:
    """Generated views and proxy renders (dicts) of a room."""
    pano = C.rgb(OUT / cfg / f"{r.id}.jpg")
    sc, rd = scene_of(r.prompt, r.seed)
    C0 = sc["layout"]["camera"]
    gen, prox = [], []
    for y in YAWS:
        gen.append(V.sample_equirect(pano, R.persp_rays(y, 0.0, VFOV, VW, VH)))
        prox.append(rd.perspective(C0, y, 0.0, VFOV, VW, VH))
    return gen, prox, sc


def save_views(cfg, df):
    d = OUT / cfg / "views"
    d.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        if not (OUT / cfg / f"{r.id}.jpg").exists() or (d / f"{r.id}_7.jpg").exists():
            continue
        gen, _, _ = views_of(cfg, r)
        for k, v in enumerate(gen):
            Image.fromarray(v).save(d / f"{r.id}_{k}.jpg", quality=95)


STRUCT = {S.WALL, S.FLOOR, S.CEIL, S.WINDOW, S.DOOR}


def struct_edges(inst: np.ndarray) -> np.ndarray:
    """Boundaries between structural surfaces (straight lines by construction)."""
    st = np.where(np.isin(inst, list(STRUCT)), inst, -1)
    e = np.zeros(inst.shape, bool)
    e[:, 1:] |= (st[:, 1:] != st[:, :-1]) & (st[:, 1:] >= 0) & (st[:, :-1] >= 0)
    e[1:, :] |= (st[1:, :] != st[:-1, :]) & (st[1:, :] >= 0) & (st[:-1, :] >= 0)
    return e


def edge_metrics(img: np.ndarray, inst: np.ndarray) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    can = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 40, 110) > 0
    se = struct_edges(inst)
    near = cv2.dilate(can.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    recall = float(near[se].mean()) if se.sum() > 50 else np.nan
    return {"struct_recall": recall, **straightness(img)}


def straightness(img: np.ndarray) -> dict:
    """Share of Canny edge pixels that lie on long (> 60 px) LSD line segments."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    can = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 40, 110) > 0
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(g)[0]
    m = np.zeros(g.shape, np.uint8)
    if lines is not None:
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            if np.hypot(x1 - x0, y1 - y0) > 60:
                cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, 3)
    return {"straight_share": float((m > 0)[can].mean()) if can.any() else np.nan}


# ADE20K ids per layout class (analysis.SEG_MODEL label ids)
ADE = {"wall": 0, "floor": 3, "ceiling": 5, "bed": 7, "window": 8, "cabinet": 10, "door": 14, "table": 15, "plant": 17,
       "curtain": 18, "chair": 19, "sofa": 23, "shelf": 24, "armchair": 30, "seat": 31, "desk": 33, "wardrobe": 35,
       "lamp": 36, "bathtub": 37, "chest": 44, "counter": 45, "sink": 47, "fridge": 50, "bookcase": 62,
       "coffee table": 64, "toilet": 65, "flower": 66, "stove": 71, "island": 73, "swivel chair": 75, "tv": 89,
       "buffet": 99, "stool": 110, "pot": 125, "screen": 130, "monitor": 141}
CABS = [ADE[k] for k in ("cabinet", "wardrobe", "chest", "shelf", "bookcase", "buffet", "counter", "desk", "table")]
CLASS_ADE = {"bed": [ADE["bed"]], "chair": [ADE["chair"], ADE["armchair"], ADE["seat"], ADE["swivel chair"], ADE["stool"]],
             "armchair": [ADE["armchair"], ADE["chair"], ADE["sofa"], ADE["seat"]], "sofa": [ADE["sofa"], ADE["armchair"], ADE["seat"]],
             "dining table": [ADE["table"], ADE["desk"], ADE["coffee table"]], "coffee table": [ADE["coffee table"], ADE["table"]],
             "desk": [ADE["desk"], ADE["table"]], "island": [ADE["island"], ADE["counter"], ADE["table"], ADE["cabinet"]],
             "plant": [ADE["plant"], ADE["flower"], ADE["pot"]], "bathtub": [ADE["bathtub"]], "toilet": [ADE["toilet"]],
             "television": [ADE["tv"], ADE["screen"], ADE["monitor"]], "fridge": [ADE["fridge"], ADE["cabinet"]],
             "stove": [ADE["stove"], ADE["counter"], ADE["cabinet"]], "sink": [ADE["sink"], ADE["counter"], ADE["cabinet"]],
             "floor lamp": [ADE["lamp"]]}


def classes_for(cls: str) -> list:
    return CLASS_ADE.get(cls, CABS)


def instance_metrics(seg: np.ndarray, prox: dict, lay: dict) -> list:
    rows = []
    inst = prox["inst"]
    n = inst.size
    for p in lay["items"]:
        m = inst == S.FURN0 + p["id"]
        if m.sum() < 0.005 * n:
            continue
        pred = np.isin(seg, classes_for(p["cls"]))
        iou = float((m & pred).sum() / max((m | pred).sum(), 1))
        cover = float(pred[m].mean())
        rows.append({"cls": p["cls"], "share": float(m.mean()), "iou": iou, "cover": cover})
    return rows


def stage_metrics(cfg, df):
    """depth + seg + edges + quality for one config (models loaded one at a time)."""
    out = OUT / cfg / "metrics.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["id"])
    todo = [r for r in df.itertuples() if (OUT / cfg / f"{r.id}.jpg").exists() and r.id not in set(have.id)]
    if not todo:
        return
    cache = {}
    for r in todo:
        cache[r.id] = views_of(cfg, r)
    rows = {r.id: {"id": r.id, "set": r.set, "room": r.room, "config": cfg} for r in todo}
    # depth
    model, proc = dp.load_depth_model("depth_anything", DEVICE)
    for r in todo:
        gen, prox, _ = cache[r.id]
        absrel, d1 = [], []
        for g, p in zip(gen, prox):
            disp = dp.predict_disparity(model, proc, g, DEVICE)
            z = p["z"]
            ok = np.isfinite(z) & (z > 0.1) & (p["inst"] != S.WINDOW)
            pred = an_align(disp, z, ok)
            rel = np.abs(pred[ok] - z[ok]) / z[ok]
            ratio = np.maximum(pred[ok] / z[ok], z[ok] / pred[ok])
            absrel.append(rel.mean())
            d1.append((ratio < 1.25).mean())
        rows[r.id].update({"depth_absrel": float(np.mean(absrel)), "depth_d1": float(np.mean(d1))})
    mem.free(model, proc)
    del model, proc
    # segmentation
    model, proc = an.load_segmenter(DEVICE)
    for r in todo:
        gen, prox, sc = cache[r.id]
        inst_rows, win_gen, win_prox = [], [], []
        for g, p in zip(gen, prox):
            seg = an.segment_classes(model, proc, g, DEVICE)
            inst_rows += instance_metrics(seg, p, sc["layout"])
            win_gen.append((seg == ADE["window"]).mean())
            win_prox.append((p["inst"] == S.WINDOW).mean())
        ious = [x["iou"] for x in inst_rows]
        rows[r.id].update({"inst_iou": float(np.mean(ious)) if ious else np.nan,
                           "inst_present": float(np.mean([x["cover"] >= 0.5 for x in inst_rows])) if ious else np.nan,
                           "window_share_gen": float(np.mean(win_gen)), "window_share_proxy": float(np.mean(win_prox))})
    mem.free(model, proc)
    del model, proc
    # edges + quality
    for r in todo:
        gen, prox, _ = cache[r.id]
        em = [edge_metrics(g, p["inst"]) for g, p in zip(gen, prox)]
        pano = C.rgb(OUT / cfg / f"{r.id}.jpg")
        meta = json.loads((OUT / cfg / f"{r.id}.json").read_text())
        rows[r.id].update({"struct_recall": float(np.nanmean([x["struct_recall"] for x in em])),
                           "straight_share": float(np.nanmean([x["straight_share"] for x in em])),
                           "brisque": C.brisque(gen), "sharpness": C.sharpness(gen),
                           "seam_ratio": float(eq.seam_ratio(pano)), "seconds": meta["seconds"],
                           "peak_vram_gb": meta["peak_vram_gb"], "peak_ram_gb": meta["peak_ram_gb"]})
    new = pd.DataFrame(rows.values())
    pd.concat([have, new], ignore_index=True).to_csv(out, index=False)


def an_align(disp: np.ndarray, z: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Least-squares scale / shift of a disparity to 1 / z on valid pixels -> depth."""
    A = np.stack([disp[ok], np.ones(ok.sum())], 1)
    s, t = np.linalg.lstsq(A, 1.0 / z[ok], rcond=None)[0]
    return 1.0 / np.maximum(s * disp + t, 1e-3)


def stage_clip(cfgs, df):
    model, proc = ev.load_clip(DEVICE)
    for cfg in cfgs:
        out = OUT / cfg / "clip.csv"
        have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["id"])
        rows = []
        for r in df.itertuples():
            if r.id in set(have.id) or not (OUT / cfg / "views" / f"{r.id}_7.jpg").exists():
                continue
            paths = [OUT / cfg / "views" / f"{r.id}_{k}.jpg" for k in range(8)]
            rows.append({"id": r.id, "clip": float(ev.clip_scores(model, proc, paths, [r.prompt] * 8, DEVICE).mean())})
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
    mem.free(model, proc)


def stage_kid(cfgs, df) -> pd.DataFrame:
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    sets = {}
    for cfg in cfgs:
        for split in sorted(df.set.unique()):
            paths = [OUT / cfg / "views" / f"{r.id}_{k}.jpg" for r in df[df.set == split].itertuples() for k in range(8)
                     if (OUT / cfg / "views" / f"{r.id}_{k}.jpg").exists()]
            if len(paths) >= 16:
                sets[(cfg, split)] = [str(p) for p in paths]
    fk = ev.fid_kid(list(hold["real_path"]), {f"{a}|{b}": v for (a, b), v in sets.items()}, DEVICE)
    return pd.DataFrame([{"config": k.split("|")[0], "set": k.split("|")[1], "kid_x1e3": v["kid_mean"] * 1e3,
                          "fid": v["fid"]} for k, v in fk.items()])


def v5_demo_rows() -> pd.DataFrame:
    """Proxy-free metrics of the v5 demo panoramas (outputs/v5_demo), for the v5 vs v6 comparison."""
    rows = []
    d = OUT / "v5" / "views"
    d.mkdir(parents=True, exist_ok=True)
    for p, s in C.pl.DEMO_PROMPTS:
        rid = f"demo_{C.pl.slug(p, s)}"
        src = ROOT / "outputs" / "v5_demo" / C.pl.slug(p, s) / "pano_4096.jpg"
        if not src.exists():
            continue
        pano = C.rgb(src)
        pano = cv2.resize(pano, (2048, 1024), interpolation=cv2.INTER_AREA)
        gen = V.level_views(pano, YAWS, VFOV, VW, VH)
        for k, v in enumerate(gen):
            Image.fromarray(v).save(d / f"{rid}_{k}.jpg", quality=95)
        rows.append({"id": rid, "set": "demo", "config": "v5", "straight_share": float(np.mean([straightness(g)["straight_share"] for g in gen])),
                     "brisque": C.brisque(gen), "sharpness": C.sharpness(gen), "seam_ratio": float(eq.seam_ratio(pano))})
    return pd.DataFrame(rows)


def report(cfgs, df, tag="v6_pano"):
    per = []
    for cfg in cfgs:
        if not (OUT / cfg / "metrics.csv").exists():
            continue
        m = pd.read_csv(OUT / cfg / "metrics.csv")
        if (OUT / cfg / "clip.csv").exists():
            m = m.merge(pd.read_csv(OUT / cfg / "clip.csv"), on="id", how="left")
        per.append(m)
    per = pd.concat(per, ignore_index=True)
    per = per[per.id.isin(df.id)]
    per.to_csv(C.TAB / f"{tag}_per_room.csv", index=False)
    cols = ["depth_absrel", "depth_d1", "inst_iou", "inst_present", "struct_recall", "straight_share",
            "window_share_gen", "window_share_proxy", "brisque", "sharpness", "seam_ratio", "clip", "seconds",
            "peak_vram_gb", "peak_ram_gb"]
    summ = per.groupby(["config", "set"])[[c for c in cols if c in per]].mean().reset_index()
    summ["n"] = per.groupby(["config", "set"]).size().values
    return per, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--configs", default="cn_small_s0.7_e1.0")
    ap.add_argument("--set", default="dev5")
    ap.add_argument("--tag", default="v6_pano")
    a = ap.parse_args()
    cfgs = a.configs.split(",")
    df = room_df(a.set)
    st = a.stage.split(",")
    if "gen" in st or "all" in st:
        stage_gen(df, cfgs)
    if "metrics" in st or "all" in st:
        for c in cfgs:
            save_views(c, df)
            stage_metrics(c, df)
    if "clip" in st or "all" in st:
        stage_clip(cfgs, df)
    if "report" in st or "all" in st:
        per, summ = report(cfgs, df, a.tag)
        if "kid" in st or "all" in st:
            k = stage_kid(cfgs, df)
            summ = summ.merge(k, on=["config", "set"], how="left")
        summ.to_csv(C.TAB / f"{a.tag}_summary.csv", index=False)
        pd.set_option("display.width", 250)
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
