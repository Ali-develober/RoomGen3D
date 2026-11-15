"""PLAN_V5 milestone 5: a clean dollhouse (furniture removed from wall / floor textures, flat-coloured hidden faces).

Rooms: milestone-3 crops directory (4096 panorama + selected candidate per room). Per room, the wall + floor plane
textures of the dollhouse for
    v4                  v4 dollhouse textures (projection, hidden texels patch-filled, furniture smeared in)
    <fill>_<clean>      v5: object + hidden mask (dilated 5 cm) filled with telea / lama / sd, then clean = none /
                        bilateral / nlm (edge-preserving)
Metrics per plane (no reference exists): residual furniture share (SegFormer on the rectified texture: share of
furniture / object classes), BRISQUE (768 px), flat-area noise sigma (Immerkaer), seconds.
Selection on dev (fixed beforehand): mean rank of residual furniture share, BRISQUE, noise.
Figures: dollhouse renders v4 vs selected per room (results/figures/v5_doll_<set>.png), plane textures.
Usage: python scripts/eval_v5_doll.py --sets dev [--methods ...] [--stage run|report]
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
from src import dollhouse_v5 as dv  # noqa: E402
from src import memory as mem  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import room_v4 as r4  # noqa: E402

ROOT = vc.ROOT
SHOTS = ROOT / "outputs" / "v5_shots"
OUT = ROOT / "outputs" / "v5_doll"
DEVICE = torch.device("cuda")
PREREG = ["v4", "telea_bilateral", "lama_none", "lama_bilateral", "lama_nlm", "sd_bilateral"]     # fixed before the run
# self-review variants added after the first run (reported, but they do not take part in the ranking: adding
# alternatives would change the mean ranks of the pre-registered ones)
EXTRA = ["lamacc_none", "lamacc_bilateral", "lamabase_bilateral"]
METHODS = PREREG + EXTRA


def furniture_share(seg, tex, id2label, obj_ids):
    from src import analysis as an
    s = min(1.0, 1024 / max(tex.shape[:2]))
    im = cv2.resize(tex, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else tex
    cm = an.segment_classes(*seg, im, DEVICE)
    return float(np.isin(cm, obj_ids).mean())


def load_room(r, id2label):
    meta = json.loads((SHOTS / r.id / "meta.json").read_text())
    n = meta["name"]
    g = np.load(ROOT / "outputs/v5_rooms" / r.id / f"{n}_geo.npz")
    dist = g["dist"]
    ana = rr.load_ana(r.id, n)
    hires = vc.rgb(SHOTS / r.id / "pano_4096.jpg")
    tex = cv2.resize(hires, (2048, 1024), interpolation=cv2.INTER_AREA)
    res = p3v.build(dist, tex, ana, id2label, open_sides=True)
    return dist, ana, hires, res


def planes_for(method, dist, ana, hires, res, id2label):
    box = res["box"]
    if method == "v4":
        sh = rb.build_shell(box, hires, dist, r4.DEFAULTS["fill"], texel_m=0.005, max_tex=4096, band_m=0.0)
        return {p["kind"]: p["texture"] for p in sh["parts"] if p["kind"] != "ceiling"}
    fill, clean = method.split("_")
    from src import furniture as fu
    inst_map = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)[0]
    obj = dv.object_mask(ana["class_map"], inst_map, id2label)
    sh = dv.clean_shell(box, hires, dist, obj, fill, clean, DEVICE)
    return {p["kind"]: p["texture"] for p in sh["parts"] if p["kind"] != "ceiling"}


def stage_run(df, methods):
    from src import analysis as an
    from src.floorplan import FURNITURE, class_ids
    id2label = vc.ade_id2label()
    obj_ids = class_ids(id2label, set(FURNITURE) | dv.OBJECT_CLASSES)
    path = OUT / "metrics.csv"
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "method"])
    rows = []
    for r in tqdm(list(df.itertuples()), desc="rooms"):
        if not (SHOTS / r.id / "meta.json").exists():
            continue
        todo = [m for m in methods if not ((have["id"] == r.id) & (have["method"] == m)).any()]
        if not todo:
            continue
        dist, ana, hires, res = load_room(r, id2label)
        textures = {}
        for m in todo:
            t0 = time.perf_counter()
            textures[m] = planes_for(m, dist, ana, hires, res, id2label)
            dt = time.perf_counter() - t0
            d = OUT / r.id / m
            d.mkdir(parents=True, exist_ok=True)
            for k, tx in textures[m].items():
                vc.save_rgb(d / f"{k}.jpg", cv2.resize(tx, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 90)
            textures[m] = (textures[m], dt)
        dv.free_lama()
        dv.free_sd()
        mem.free()
        seg = an.load_segmenter(DEVICE)
        for m, (tx, dt) in textures.items():
            for k, t in tx.items():
                s768 = cv2.resize(t, (768, int(768 * t.shape[0] / t.shape[1])) if t.shape[1] >= t.shape[0]
                                  else (int(768 * t.shape[1] / t.shape[0]), 768), interpolation=cv2.INTER_AREA)
                rows.append({"id": r.id, "set": r.set, "method": m, "plane": k, "seconds": dt,
                             "furniture_share": furniture_share(seg, t, id2label, obj_ids),
                             "brisque": vc.brisque([s768]),
                             "noise": r4.noise_sigma(t, np.ones(t.shape[:2], bool))})
        seg = None
        mem.free()
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def report(tag="v5_doll"):
    met = pd.read_csv(OUT / "metrics.csv")
    met["is_floor"] = met["plane"] == "floor"
    summ = met.groupby(["set", "method"])[["furniture_share", "brisque", "noise", "seconds"]].mean().reset_index()
    fl = met[met["is_floor"]].groupby(["set", "method"])[["furniture_share", "brisque", "noise"]].mean()
    fl.columns = [f"floor_{c}" for c in fl.columns]
    summ = summ.merge(fl.reset_index(), on=["set", "method"], how="left")
    summ.to_csv(vc.TAB / f"{tag}_summary.csv", index=False, float_format="%.4f")
    dev = summ[(summ["set"] == "dev") & (summ["method"] != "v4") & summ["method"].isin(PREREG)].set_index("method")
    rk = pd.DataFrame({"f": dev["furniture_share"].rank(), "b": dev["brisque"].rank(), "n": dev["noise"].rank()})
    dev["mean_rank"] = rk.mean(1)
    best = dev["mean_rank"].idxmin()
    fill, clean = best.split("_")
    vc.write_json(vc.TAB / f"{tag}_decision.json", {"selected": best, "fill": fill, "clean": clean,
                                                   "mean_rank": dev["mean_rank"].sort_values().to_dict(),
                                                   "rule": "dev: mean rank of residual furniture share, BRISQUE, noise"})
    print(summ.round(3).to_string())
    print("selected", best)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    df = vc.rooms(a.sets)
    if a.limit:
        df = df.groupby("set").head(a.limit).reset_index(drop=True)
    if a.stage in ("all", "run"):
        stage_run(df, a.methods)
    if a.stage in ("all", "report"):
        report()


if __name__ == "__main__":
    main()
