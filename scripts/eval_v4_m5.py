"""PLAN_V4 milestone 5: the remaining v3 weaknesses.

Part "fill"  - hidden floor / wall areas (texels behind furniture) in the v4 shell textures (5 mm texels, 4096
    panorama). Methods (src/room_box.inpaint): ns (v3), patch (v3 alternative), tile (periodic copy along the
    texture's dominant repetition), patch_base / tile_base (only the fill's detail band on a smooth NS base).
    (a) synthetic furniture-shaped holes in VISIBLE parts of walls and floor: PSNR / SSIM against the truth;
    (b) the REAL hidden regions (no truth): texture-energy ratio (std of a 1-4 px DoG band inside the fill / in the
        visible part of the same plane; 1 = same texture, ~0 = smooth blotch), darkness (|mean grey of the fill -
        median grey of the visible plane|, the "dark patches"), BRISQUE of the filled region's bounding box.
    Rule (dev rooms): among methods whose synthetic SSIM is >= NS's - 0.02, the best mean rank of energy-ratio
    error, darkness and BRISQUE on the real holes.
Part "open"  - open sides (room_box.mark_open_sides): the beyond-share of every INFERRED side; the bench rooms are
    closed (layout ground truth), so a side marked open there is a false alarm. Threshold = the lowest value in
    THRS with <= 5 % false-open sides on the dev rooms. Reported: held-out false-open rate and area error before /
    after; the six demo rooms (open-plan dining / living rooms were boxed at 74.7 / 56 m2): sides marked open, area.
Part "doors" - doors / windows from Mask2Former on (v3) the 13 views of the 1024 x 512 panorama at 640 px vs
    (v4) the same 13 directions cut from the 4096 panorama at 1024 px, and (v4b) the 26 milestone-2 directions
    at 1024 px; openings on the box walls (plan_v3.openings_on_box) vs the layout's doors / windows (same wall,
    same kind, centre within 0.5 m): precision / recall, held-out (choice of v4 / v4b on dev by F1).
Outputs: results/tables/fill_*.csv, fill_selection.json, open_sides*.csv/json, doors_v4*.csv/json;
figures fill_examples.png, open_sides_demo.png
Usage: python scripts/eval_v4_m5.py --part fill|open|doors|all [--limit N]
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import equirect as eq  # noqa: E402
from src import instances as ins  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import plan_v3 as pv  # noqa: E402
from src import room_box as rb  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402
import plan_v3_eval as pve  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEVICE = torch.device("cuda")
METHODS = ["ns", "patch", "tile", "patch_base", "tile_base"]
THRS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def rooms(limit=None):
    rows = []
    for r in mb.rooms().itertuples():
        d = mb.OUT / r.id
        rows.append({"id": r.id, "set": "dev" if r.split == "dev" else "test", "d": d, "ana": d, "dist": d / "dist.npy",
                     "hires": ROOT / "outputs/objects_eval" / r.id / "pano_4096.jpg", "pano": d / "pano.png"})
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        rows.append({"id": "demo_" + pl.slug(prompt, seed), "set": "demo", "d": d, "ana": d / "v3",
                     "dist": d / "distance.npy", "hires": d / "v4/pano_4096_detail.png", "pano": d / "pano.png"})
    df = pd.DataFrame(rows)
    return df.groupby("set").head(limit) if limit else df


def load(r):
    cm = cv2.imread(str(r.ana / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    return np.load(r.dist), cm


# --------------------------------------------------------------------------- #
def dog_std(g, m):
    b = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 4.0)
    return float(b[m].std()) if m.sum() > 200 else np.nan


def part_fill(df, id2label):
    import piq
    rows, real, ex = [], [], []
    rng = np.random.default_rng(0)
    for r in tqdm(list(df.itertuples()), desc="fill"):
        dist, cm = load(r)
        hires = mb.rgb(r.hires)
        box = rb.fit_box(dist, cm, id2label)
        for pl_ in rb.shell_planes(box)[:5]:                      # 4 walls + floor
            img, hidden, beyond = rb.plane_texture(pl_, box, hires, dist, "ns", texel_m=0.005, max_tex=2048)
            raw, _, _ = rb.plane_texture(pl_, box, hires, dist, "none", texel_m=0.005, max_tex=2048)
            vis = ~hidden & ~beyond
            h, w = vis.shape
            g_vis = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY).astype(np.float32)
            # (a) synthetic holes in visible texture
            for _ in range(2):
                hw, hh = int(w * rng.uniform(0.12, 0.3)), int(h * rng.uniform(0.15, 0.35))
                x0 = int(rng.integers(0, max(w - hw, 1)))
                y0 = int(rng.integers(0, max(h - hh, 1))) if pl_["kind"] == "floor" else h - hh - int(rng.integers(0, max(h // 10, 1)))
                hole = np.zeros_like(vis)
                hole[y0:y0 + hh, x0:x0 + hw] = True
                if vis[hole].mean() < 0.98 or vis.mean() < 0.3:
                    continue
                ys, xs = np.nonzero(hole)
                for meth in METHODS:
                    t0 = time.perf_counter()
                    out = rb.inpaint(raw, hole, meth)
                    a = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    b = raw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    go = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
                    rows.append({"id": r.id, "set": r.set, "plane": pl_["kind"], "method": meth,
                                 "time_s": time.perf_counter() - t0,
                                 "psnr": peak_signal_noise_ratio(b, a, data_range=255),
                                 "ssim": structural_similarity(b, a, channel_axis=2, data_range=255),
                                 "energy_ratio": dog_std(go, hole) / max(dog_std(g_vis, vis & ~hole), 1e-6)})
            # (b) the real hidden region
            fillm = hidden & ~beyond
            if fillm.sum() < 2000 or vis.mean() < 0.2:
                continue
            ys, xs = np.nonzero(fillm)
            for meth in METHODS:
                out = rb.inpaint(raw, fillm, meth)
                go = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
                crop = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                bq = float(piq.brisque(torch.from_numpy(cv2.resize(crop, (256, 256))).permute(2, 0, 1)[None].float() / 255,
                                       data_range=1.0)) if min(crop.shape[:2]) >= 32 else np.nan
                real.append({"id": r.id, "set": r.set, "plane": pl_["kind"], "method": meth,
                             "hidden_share": float(fillm.mean()),
                             "energy_ratio": dog_std(go, fillm) / max(dog_std(g_vis, vis), 1e-6),
                             "darkness": abs(float(go[fillm].mean()) - float(np.median(g_vis[vis]))),
                             "brisque": bq})
                if r.set == "demo" and pl_["kind"] == "floor" and len(ex) < 4 * len(METHODS):
                    ex.append((r.id, meth, out))
    syn, rl = pd.DataFrame(rows), pd.DataFrame(real)
    syn.to_csv(TAB / "fill_synthetic_per_hole.csv", index=False)
    rl.to_csv(TAB / "fill_real_per_plane.csv", index=False)
    rl["energy_err"] = (rl["energy_ratio"] - 1).abs()
    s1 = syn.groupby(["set", "method"])[["psnr", "ssim", "energy_ratio", "time_s"]].mean()
    s2 = rl.groupby(["set", "method"])[["energy_ratio", "energy_err", "darkness", "brisque"]].mean()
    summ = s1.join(s2, rsuffix="_real").reset_index()
    summ.to_csv(TAB / "fill_summary.csv", index=False)
    print(summ.round(3).to_string())
    dev = summ[summ["set"] == "dev"].set_index("method")
    ok = dev[dev["ssim"] >= dev.loc["ns", "ssim"] - 0.02]
    rank = pd.DataFrame({"e": ok["energy_err"].rank(), "d": ok["darkness"].rank(), "b": ok["brisque"].rank()}).mean(1)
    sel = {"method": rank.idxmin(), "passed_ssim_guard": list(ok.index), "mean_rank": rank.to_dict(),
           "rule": "SSIM >= NS - 0.02 on synthetic holes, then mean rank of |energy ratio - 1|, darkness, BRISQUE on the real holes (dev)"}
    (TAB / "fill_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    if ex:
        ids = list(dict.fromkeys(e[0] for e in ex))[:3]
        fig, axes = plt.subplots(len(ids), len(METHODS), figsize=(4 * len(METHODS), 3.4 * len(ids)), dpi=70, squeeze=False)
        for i, rid in enumerate(ids):
            for j, meth in enumerate(METHODS):
                im = [e[2] for e in ex if e[0] == rid and e[1] == meth][0]
                axes[i, j].imshow(im)
                axes[i, j].set_title(f"{meth}" + (f" - {rid[5:]}" if j == 0 else ""), fontsize=10)
                axes[i, j].axis("off")
        fig.suptitle("Floor texture (5 mm texels) with the area hidden behind furniture filled, by method", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / "fill_examples.png", facecolor="white")
        plt.close(fig)
    return sel


# --------------------------------------------------------------------------- #
def part_open(df, id2label):
    rows, demo = [], []
    for r in tqdm(list(df.itertuples()), desc="open sides"):
        dist, cm = load(r)
        box = rb.fit_box(dist, cm, id2label)
        bo = rb.mark_open_sides(box, dist, cm, id2label, 1.1)          # shares + main-wall gaps (thr > 1: none open)
        for k, s in bo["sides"].items():
            rows.append({"id": r.id, "set": r.set, "side": k, "detected": s["detected"], "beyond_share": s["beyond_share"]})
        if r.set != "demo":
            G = erb.gt_box(json.loads((r.d / "layout.json").read_text()))
            rows[-1]["gt_area"] = None
            for t in THRS:
                b = rb.mark_open_sides(box, dist, cm, id2label, t)
                ga = (G["x"][1] - G["x"][0]) * (G["z"][1] - G["z"][0])
                rows.append({"id": r.id, "set": r.set, "side": "room", "thr": t, "area": b["area"], "gt_area": ga,
                             "area_err": abs(b["area"] / ga - 1), "area_err_v3": abs(box["area"] / ga - 1),
                             "n_open": b["n_open"]})
    per = pd.DataFrame(rows)
    per.to_csv(TAB / "open_sides_per_side.csv", index=False)
    sides = per[per["side"] != "room"]
    rm_ = per[per["side"] == "room"]
    tab = []
    for t in THRS:
        for st in ("dev", "test"):
            x = rm_[(rm_["set"] == st) & (rm_["thr"] == t)]
            tab.append({"thr": t, "set": st, "rooms": len(x),
                        "false_open_share_of_all_sides": float(x["n_open"].sum() / max(4 * len(x), 1)),
                        "area_err_v3": x["area_err_v3"].median(), "area_err": x["area_err"].median()})
    tab = pd.DataFrame(tab)
    tab.to_csv(TAB / "open_sides_thresholds.csv", index=False)
    print(tab.round(3).to_string())
    d = tab[tab["set"] == "dev"]
    okt = d[d["false_open_share_of_all_sides"] <= 0.05]
    thr = float(okt["thr"].min()) if len(okt) else float(max(THRS))
    sel = {"threshold": thr, "rule": "lowest threshold with <= 5 % false-open sides on the closed dev rooms"}
    (TAB / "open_sides_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    # demo rooms: plans before / after
    figs = []
    for r in df[df["set"] == "demo"].itertuples():
        dist, cm = load(r)
        ana = {"class_map": cm, "inst_map": cv2.imread(str(r.ana / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((r.ana / "inst_v3.json").read_text())}
        tex = mb.rgb(r.d / "texture.png")
        rb.OPEN_THR = thr
        r3 = p3v.build(dist, tex, ana, id2label)
        r4 = p3v.build(dist, tex, ana, id2label, open_sides=True)
        i3, _ = pv.draw_plan(r3["box"], r3["solids"], r3["openings"], title=r.id[5:])
        i4, st = pv.draw_plan(r4["box"], r4["solids"], r4["openings"], title=r.id[5:])
        demo.append({"room": r.id[5:], "area_v3": r3["box"]["area"], "area_v4": r4["box"]["area"],
                     "open_sides": r4["box"]["n_open"],
                     "beyond_shares": {k: round(v["beyond_share"], 2) for k, v in r4["box"]["sides"].items()}})
        figs.append((r.id[5:], i3, i4))
    pd.DataFrame(demo).to_csv(TAB / "open_sides_demo.csv", index=False)
    print(pd.DataFrame(demo).to_string())
    fig, axes = plt.subplots(len(figs), 2, figsize=(16, 7 * len(figs)), dpi=60, squeeze=False)
    for i, (t, a, b) in enumerate(figs):
        axes[i, 0].imshow(a); axes[i, 0].set_title(f"v3 plan - {t}", fontsize=13)
        axes[i, 1].imshow(b); axes[i, 1].set_title("v4 plan: open sides marked, not pushed out", fontsize=13)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "open_sides_demo.png", facecolor="white")
    plt.close(fig)
    return sel


# --------------------------------------------------------------------------- #
def part_doors(df, id2label):
    from src.floorplan import class_ids
    model, proc = ins.load_panoptic(DEVICE)
    door_ids = class_ids(id2label, list(pv.OPENING_CLASSES))
    cfgs = {"v3": None, "v4_13x1024": (ins.VIEWS, ins.FOV, 1024),
            "v4_26x1024": ([(y, p) for y, p in pr.view_grid()], pr.VIEW_FOV, 1024)}
    rows = []
    for r in tqdm([r for r in df.itertuples() if r.set != "demo"], desc="doors"):
        dist, cm = load(r)
        hires = mb.rgb(r.hires)
        box = rb.fit_box(dist, cm, id2label)
        gto = pve.gt_openings(json.loads((r.d / "layout.json").read_text()))
        for name, cfg in cfgs.items():
            cm2 = cm
            if cfg is not None:
                views, fov, size = cfg
                vo = ins.run_views(model, proc, hires, DEVICE, views=views, fov=fov, size=size)
                mv = ins.merge_views(vo, id2label, cm.shape[1], cm.shape[0], views=views, fov=fov, size=size)
                hyb = ins.hybrid_class_map(cm, mv)
                cm2 = cm.copy()
                was = np.isin(cm, door_ids)
                cm2[was] = hyb[was]                                 # v3 door / window pixels: relabelled by v4
                now = np.isin(hyb, door_ids)
                cm2[now] = hyb[now]
            ops = pv.openings_on_box(dist, cm2, id2label, box)
            used, tp = set(), 0
            for o in ops:
                c = (o["s0"] + o["s1"]) / 2
                for k, q in enumerate(gto):
                    if k not in used and q["wall"] == o["wall"] and q["kind"] == o["kind"] and abs(q["c"] - c) < 0.5:
                        used.add(k)
                        tp += 1
                        break
            rows.append({"id": r.id, "set": r.set, "config": name, "pred": len(ops), "gt": len(gto), "tp": tp,
                         "door_window_px": int(np.isin(cm2, door_ids).sum())})
    per = pd.DataFrame(rows)
    per.to_csv(TAB / "doors_v4_per_room.csv", index=False)
    s = per.groupby(["set", "config"])[["pred", "gt", "tp", "door_window_px"]].sum().reset_index()
    s["precision"] = s["tp"] / s["pred"].clip(lower=1)
    s["recall"] = s["tp"] / s["gt"].clip(lower=1)
    s["f1"] = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"]).clip(lower=1e-9)
    s.to_csv(TAB / "doors_v4.csv", index=False)
    print(s.round(3).to_string())
    d = s[(s["set"] == "dev") & (s["config"] != "v3")].sort_values("f1", ascending=False)
    sel = {"config": d.iloc[0]["config"], "rule": "best dev F1 among the v4 view configurations"}
    (TAB / "doors_v4_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    del model
    torch.cuda.empty_cache()
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    id2label = erb.id2label_ade()
    df = rooms(a.limit)
    if a.part in ("open", "all"):
        part_open(df, id2label)
    if a.part in ("fill", "all"):
        part_fill(df, id2label)
    if a.part in ("doors", "all"):
        part_doors(df, id2label)


if __name__ == "__main__":
    main()
