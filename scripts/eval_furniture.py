"""PLAN_V3 milestone 3: solid furniture, v2 vs v3 against the layout boxes of the test bench.

    v2  furniture rectangles of the v2 360 floor plan (SegFormer classes, connected components,
        floorplan.extract_furniture: min-area rectangle + depth-prior completion); no height
    v3  furniture solids (src/furniture.py) from the v3 panoptic instances inside the v3 room box
Ground truth: layout boxes (axis-aligned, camera frame). Footprints are matched class-agnostically by
Hungarian matching on bird's-eye-view IoU (GT boxes with >= 0.12% visible pixels only, as in milestone 1).
    detection    precision / recall / F1 at IoU >= 0.25
    accuracy     on matched pairs (IoU > 0.05): footprint IoU, centre error (m), size error (long / short side,
                 relative), orientation error (deg, distance of the rectangle angle to the GT axes), height error (v3)
    "_scaled"    centre / size after scaling by 1.4 m / predicted camera height (removes DAP's global scale error)
Tuning: erosion x trimming x orientation snap x wall snap x priors x leftover components on the DEV rooms (objective F1@0.25,
tie-break mean IoU); held-out rooms reported once. Counts of clamped / rejected / merged instances reported.
Outputs: results/tables/furn_*.csv; results/figures/furn_plans.png, furn_dollhouse.png
Usage: python scripts/eval_furniture.py [--limit N]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import furniture as fu  # noqa: E402
from src import instances as ins  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import room_mesh as rm  # noqa: E402
from src.utils import set_seed  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

GRID = {"erode": [1, 3], "trim": [2.0, 5.0], "snap_deg": [0.0, 20.0], "wall_snap": [0.0, 0.3], "priors": [False, True],
        "leftovers": [False, True]}


def gt_rects(g: dict) -> tuple[list[np.ndarray], list[dict]]:
    L = g["layout"]
    C = np.array(L["camera"])
    vis = np.bincount(g["inst"][g["inst"] >= 0].ravel(), minlength=len(L["boxes"]))
    min_px = int(ins.MIN_SHARE * g["inst"].size)
    rects, boxes = [], []
    for i, b in enumerate(L["boxes"]):
        if vis[i] < min_px:
            continue
        rects.append(np.array([[b["x0"], b["z0"]], [b["x1"], b["z0"]], [b["x1"], b["z1"]], [b["x0"], b["z1"]]]) - C[[0, 2]])
        boxes.append(b)
    return rects, boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    if inter <= 0:
        return 0.0
    A, B = cv2.contourArea(pa), cv2.contourArea(pb)
    return float(inter / (A + B - inter))


def rect_stats(c: np.ndarray):
    e = np.linalg.norm(c[[1, 3]] - c[0], axis=1)
    d = c[1] - c[0] if e[0] >= e[1] else c[3] - c[0]
    ang = np.degrees(np.arctan2(d[1], d[0])) % 90
    return c.mean(0), np.sort(e)[::-1], min(ang, 90 - ang)


def match(gt: list, pred: list, heights_gt=None, heights_pred=None, scale: float = 1.0) -> tuple[dict, list]:
    out = {"n_gt": len(gt), "n_pred": len(pred), "tp25": 0}
    rows = []
    if not gt or not pred:
        return out, rows
    M = np.array([[iou(g, p) for p in pred] for g in gt])
    r, c = linear_sum_assignment(-M)
    for i, j in zip(r, c):
        if M[i, j] <= 0.05:
            continue
        cg, eg, _ = rect_stats(gt[i])
        cp, ep, ap = rect_stats(pred[j])
        cps, eps_, _ = rect_stats(pred[j] * scale)
        row = {"iou": M[i, j], "centre_err_m": float(np.linalg.norm(cp - cg)),
               "size_err_long": float(abs(ep[0] / eg[0] - 1)), "size_err_short": float(abs(ep[1] / eg[1] - 1)),
               "orient_err_deg": float(ap), "centre_err_m_scaled": float(np.linalg.norm(cps - cg)),
               "size_err_long_scaled": float(abs(eps_[0] / eg[0] - 1)),
               "size_err_short_scaled": float(abs(eps_[1] / eg[1] - 1))}
        if heights_gt is not None and heights_pred is not None:
            row["height_err_m"] = float(abs(heights_pred[j] - heights_gt[i]))
        rows.append(row)
        out["tp25"] += M[i, j] >= 0.25
    return out, rows


def plan_axes(ax, gt, v2, v3, box_c, title):
    for rect, col, lw, ls in [(r, "#222222", 2.0, "-") for r in gt] + [(r, "#d9534f", 1.4, "--") for r in v2] + \
                             [(r, "#2a78d6", 1.6, "-") for r in v3]:
        ax.add_patch(MplPolygon(rect[:, [0, 1]], closed=True, fill=False, ec=col, lw=lw, ls=ls))
    ax.add_patch(MplPolygon(box_c, closed=True, fill=False, ec="#999999", lw=1, ls=":"))
    ax.plot([0], [0], "k^", ms=7)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    tag = "" if args.limit is None else "_smoke"
    df = mb.rooms()
    if args.limit:
        df = df.groupby("split").head(args.limit)
    id2label = erb.id2label_ade()
    data = {}
    for r in tqdm(list(df.itertuples()), desc="load + boxes", unit="room"):
        d = mb.OUT / r.id
        g = mb.load_gt(d)
        dist = np.load(d / "dist.npy")
        cm3 = cv2.imread(str(d / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cm2 = cv2.imread(str(d / "seg_v2.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        inst = cv2.imread(str(d / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        insts = json.loads((d / "inst_v3.json").read_text())
        box = rb.fit_box(dist, cm3, id2label)
        R2, _ = rm.gravity_rotation(dist, cm2, id2label)
        plan2 = rm.plan_360(dist, cm2, id2label, R2)
        v2 = [(np.c_[f["rect"][:, 0], np.zeros(4), f["rect"][:, 1]] @ R2)[:, [0, 2]] for f in plan2["furniture"]]
        data[r.id] = {"g": g, "dist": dist, "inst": inst, "insts": insts, "box": box, "v2": v2, "cm3": cm3,
                      "v2_scale": 1.4 / plan2["frame"]["d"], "row": r}

    # --- dev sweep ----------------------------------------------------------------------------------------- #
    sweep = []
    keys = list(GRID)
    for vals in tqdm(list(itertools.product(*GRID.values())), desc="dev sweep", unit="config"):
        p = dict(zip(keys, vals))
        tp = ngt = npred = 0
        ious = []
        for rid, D in data.items():
            if D["row"].split != "dev":
                continue
            sol, _ = fu.build_solids(D["dist"], D["inst"], D["insts"], D["box"], id2label, p, D["cm3"])
            gt, _ = gt_rects(D["g"])
            o, rows = match(gt, fu.footprints_cam(sol, D["box"]))
            tp, ngt, npred = tp + o["tp25"], ngt + o["n_gt"], npred + o["n_pred"]
            ious += [x["iou"] for x in rows]
        pr, rc = tp / max(npred, 1), tp / max(ngt, 1)
        sweep.append({**p, "precision25": pr, "recall25": rc, "f1_25": 2 * pr * rc / max(pr + rc, 1e-9),
                      "mean_iou_matched": float(np.mean(ious)) if ious else 0.0, "n_pred": npred})
    sweep = pd.DataFrame(sweep).sort_values(["f1_25", "mean_iou_matched"], ascending=False)
    best = {k: sweep.iloc[0][k] for k in keys}
    best = {k: (bool(v) if k in ("priors", "leftovers") else float(v)) for k, v in best.items()}
    print(sweep.head(8).round(3).to_string(index=False))
    print("selected on dev:", best)

    # --- v2 vs v3 ------------------------------------------------------------------------------------------ #
    per_room, pairs, figs = [], [], []
    for rid, D in tqdm(data.items(), desc="scoring", unit="room"):
        r = D["row"]
        gt, gboxes = gt_rects(D["g"])
        sol, cnt = fu.build_solids(D["dist"], D["inst"], D["insts"], D["box"], id2label, best, D["cm3"])
        alt = {**best, "priors": not best["priors"]}
        sol_alt, cnt_alt = fu.build_solids(D["dist"], D["inst"], D["insts"], D["box"], id2label, alt, D["cm3"])
        v3 = fu.footprints_cam(sol, D["box"])
        sc3 = 1.4 / D["box"]["camera_height"]
        alt_name = "v3 priors on" if alt["priors"] else "v3 priors off"
        for name, pred, sc, hp, cn in (("v2", D["v2"], D["v2_scale"], None, None),
                                       ("v3", v3, sc3, [s["top"] for s in sol], cnt),
                                       (alt_name, fu.footprints_cam(sol_alt, D["box"]), sc3, [s["top"] for s in sol_alt], cnt_alt)):
            o, rows = match(gt, pred, [b["h"] for b in gboxes] if hp else None, hp, sc)
            per_room.append({"id": rid, "room": r.room, "split": r.split, "method": name, **o, **(cn or {})})
            pairs += [{"id": rid, "split": r.split, "method": name, **x} for x in rows]
        if r.split == "test" and r.layout_seed in (5, 6):
            box = D["box"]
            bc = (np.c_[rb.box_corners_A(box)[:, 0], np.zeros(4), rb.box_corners_A(box)[:, 1]] @ box["R"])[:, [0, 2]]
            shell = rb.build_shell(box, mb.rgb(mb.OUT / rid / "texture.png"), D["dist"])
            im_, _ = fu.add_leftovers(D["inst"], D["insts"], D["cm3"], id2label) if best["leftovers"] else (D["inst"], None)
            part = fu.solids_part(sol, box, D["dist"], im_, mb.rgb(mb.OUT / rid / "texture.png"))
            mesh = rb.combine([p for p in shell["parts"] if p["kind"] != "ceiling"] + ([part] if len(part["faces"]) else []))
            if r.layout_seed == 5:
                from src import textured_mesh as tm
                tm.export_glb(mesh, mb.OUT / rid / "v3_m3_dollhouse.glb")
            figs.append((r.prompt, gt, D["v2"], v3, bc, dollhouse_view(mesh)))
    pr_, pp = pd.DataFrame(per_room), pd.DataFrame(pairs)
    summ = []
    for (split, method), x in pr_.groupby(["split", "method"]):
        y = pp[(pp["split"] == split) & (pp["method"] == method)]
        P, R = x["tp25"].sum() / max(x["n_pred"].sum(), 1), x["tp25"].sum() / max(x["n_gt"].sum(), 1)
        row = {"split": split, "method": method, "gt": int(x["n_gt"].sum()), "pred": int(x["n_pred"].sum()),
               "precision25": P, "recall25": R, "f1_25": 2 * P * R / max(P + R, 1e-9), "matched": len(y),
               **{c: y[c].median() for c in y.columns if c not in ("id", "split", "method")}}
        for c in ("clamped", "rejected", "merged", "shrunk", "dropped_outside", "too_few_points", "instances"):
            if c in x:
                row[f"n_{c}"] = x[c].sum() if x[c].notna().any() else np.nan
        summ.append(row)
    summ = pd.DataFrame(summ)
    T, F = ROOT / "results/tables", ROOT / "results/figures"
    sweep.to_csv(T / f"furn_dev_sweep{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(T / f"furn_summary{tag}.csv", index=False, float_format="%.4f")
    pr_.to_csv(T / f"furn_per_room{tag}.csv", index=False, float_format="%.4f")
    pp.to_csv(T / f"furn_pairs{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())

    if figs:
        n = len(figs)
        cols = 5
        blocks = int(np.ceil(n / cols))
        fig, axes = plt.subplots(2 * blocks, cols, figsize=(4.4 * cols, 8.6 * blocks), dpi=90)
        axes = np.atleast_2d(axes)
        for a in axes.ravel():
            a.axis("off")
        for i, (title, gt, v2, v3, bc, dh) in enumerate(figs):
            row, col = 2 * (i // cols), i % cols
            axes[row, col].axis("on")
            plan_axes(axes[row, col], gt, v2, v3, bc, title)
            axes[row + 1, col].imshow(dh)
        axes[0, 0].legend(handles=[plt.Line2D([], [], color="#222222", lw=2, label="layout (truth)"),
                                   plt.Line2D([], [], color="#d9534f", ls="--", label="v2 plan rectangles"),
                                   plt.Line2D([], [], color="#2a78d6", label="v3 solids"),
                                   plt.Line2D([], [], color="#999999", ls=":", label="v3 room box")],
                          loc="upper left", fontsize=8)
        fig.suptitle("Held-out rooms: furniture footprints from above (camera = triangle) and v3 dollhouse "
                     "(box shell + solids, near walls culled)", fontsize=13)
        fig.tight_layout()
        fig.savefig(F / f"furn_plans{tag}.png", facecolor="white")
        plt.close(fig)


def dollhouse_view(mesh: dict, size: int = 320) -> np.ndarray:
    v = mesh["vertices"]
    lo, hi = v.min(0), v.max(0)                                 # the shell spans the room
    c = (lo + hi) / 2
    r = 1.25 * np.linalg.norm((hi - lo)[[0, 2]])
    el, az = np.radians(60), np.radians(200)
    eye = c + np.array([r * np.sin(az) * np.cos(el), -r * np.sin(el), r * np.cos(az) * np.cos(el)])
    f = mesh["faces"]
    a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    front = (np.cross(b - a, cc - a) * (eye - (a + b + cc) / 3)).sum(1) > 0     # back-face culling
    mesh = {**mesh, "faces": f[front]}
    hit, _, col = rm.Raycaster(mesh).cast(eye, rm.look_at_dirs(eye, c, 50, size), color=True)
    return col.reshape(size, size, 3)


if __name__ == "__main__":
    main()
