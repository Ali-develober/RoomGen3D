"""PLAN_V3 milestone 1: instance segmentation on panoramas, v2 vs v3, on the layout-first test bench.

v2: SegFormer on 14 views -> class map -> connected components per furniture class (src/instances.v2_instances)
v3: Mask2Former panoptic on 13 views -> cross-view merged instances (src/instances.merge_views); its class map
    for room surfaces is replaced by the SegFormer one (instances.hybrid_class_map, "v3"); the pure panoptic
    class map is reported as "v3_panoptic_only"

Ground truth: the layout rendered from the camera (scripts/make_bench.py): class per pixel and furniture box index.
The generated panorama follows the layout only approximately (milestone 6: furniture IoU 0.45), so absolute
numbers are a lower bound for both methods; the comparison is on identical panoramas.

Metrics (pixels weighted by cos(latitude) = solid angle):
    detection   GT furniture instances with >= 0.12% visible area; Hungarian matching on mask IoU;
                a match counts if IoU >= 0.5 ("any class") and, for "same family", if the furniture family agrees.
                Precision, recall, F1; recall per GT class; recall of objects that cross the panorama seam.
    semantics   IoU vs. the rendered layout labels: wall, floor, ceiling, furniture, door, window; mIoU of the first four.
Tuning: SCORE_MIN x MERGE_IOU grid on the DEV rooms (layout seeds 1-4), best F1 (same family);
report on the HELD-OUT rooms (seeds 5-8).
Outputs: outputs/bench/<id>/{m2f_views.npz (cache), class_v3.png, inst_v3.png}; results/tables/inst_*.csv;
results/figures/inst_examples.png
Usage: python scripts/eval_instances.py [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import instances as ins  # noqa: E402
from src.utils import set_seed  # noqa: E402
import make_bench as mb  # noqa: E402

W, H = 1024, 512
SCORES = [0.3, 0.5, 0.7]
MERGES = [0.3, 0.5, 0.7]
GROUPS = {"wall": [0], "floor": [3, 28], "ceiling": [5], "door": [14, 58], "window": [8]}
AREA = np.cos(np.pi / 2 - np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))


def cache_views(model, proc, pano, path: Path, device):
    """Raw per-view panoptic output, cached as npz (segment maps int16 + json info)."""
    if path.exists():
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return [(z[f"seg{k}"].astype(np.int32), info[k]) for k in range(len(info))]
    out = ins.run_views(model, proc, pano, device)
    np.savez_compressed(path, info=json.dumps([o[1] for o in out]), **{f"seg{k}": o[0].astype(np.int16) for k, o in enumerate(out)})
    return out


def detection(gt_inst, gt_fam, pred_inst, pred_fam, min_px):
    """Matching of GT and predicted instances. Returns per-GT rows and counts."""
    g_ids = [g for g in np.unique(gt_inst[gt_inst >= 0]) if (gt_inst == g).sum() >= min_px]
    p_ids = list(range(len(pred_fam)))
    rows = []
    if not g_ids:
        return rows, {"tp_any": 0, "tp_fam": 0, "n_gt": 0, "n_pred": len(p_ids)}
    iou = np.zeros((len(g_ids), len(p_ids)))
    for a, g in enumerate(g_ids):
        gm = gt_inst == g
        cand = np.unique(pred_inst[gm & (pred_inst >= 0)])
        for p in cand:
            pm = pred_inst == p
            iou[a, p] = AREA[gm & pm].sum() / AREA[gm | pm].sum()
    res = {}
    for name, ok in (("any", np.ones_like(iou, bool)),
                     ("fam", np.array([[gt_fam[g] == pred_fam[p] for p in p_ids] for g in g_ids]).reshape(iou.shape))):
        m = np.where(ok, iou, 0.0)
        r, c = linear_sum_assignment(-m) if m.size else ([], [])
        hit = {g_ids[i]: m[i, j] for i, j in zip(r, c) if m[i, j] >= 0.5}
        res[name] = hit
    for g in g_ids:
        rows.append({"gt": int(g), "family": gt_fam[g], "hit_any": g in res["any"], "hit_fam": g in res["fam"],
                     "best_iou": float(iou[g_ids.index(g)].max()) if len(p_ids) else 0.0,
                     "seam": bool((gt_inst[:, 0] == g).any() and (gt_inst[:, -1] == g).any())})
    return rows, {"tp_any": len(res["any"]), "tp_fam": len(res["fam"]), "n_gt": len(g_ids), "n_pred": len(p_ids)}


def semantics(pred_cls, gt_cls, fam):
    out = {}
    furn = np.array(sorted(fam))
    for k, ids in {**GROUPS, "furniture": furn}.items():
        a, b = np.isin(pred_cls, ids), np.isin(gt_cls, ids)
        u = AREA[a | b].sum()
        out[f"iou_{k}"] = float(AREA[a & b].sum() / u) if u > 0 else np.nan
    out["miou"] = float(np.nanmean([out[f"iou_{k}"] for k in ("wall", "floor", "ceiling", "furniture")]))
    return out


def prf(tp, n_gt, n_pred):
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def colorize(inst, pano):
    rng = np.random.default_rng(3)
    cols = rng.integers(40, 255, (256, 3))
    out = (pano.astype(np.float32) * 0.35).astype(np.uint8)
    m = inst >= 0
    out[m] = (0.3 * pano[m] + 0.7 * cols[inst[m] % 256]).astype(np.uint8)
    for i in np.unique(inst[m]):                                    # outlines
        cnt, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    tag = "" if args.limit is None else "_smoke"
    df = mb.rooms()
    if args.limit:
        df = df.groupby("split").head(args.limit)
    model, proc = ins.load_panoptic(device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    fam = ins.family_of(id2label)
    raw = {}
    for r in tqdm(list(df.itertuples()), desc="Mask2Former views", unit="room"):
        d = mb.OUT / r.id
        pano = cv2.resize(mb.rgb(d / "pano.png"), (W, H), interpolation=cv2.INTER_AREA)
        raw[r.id] = cache_views(model, proc, pano, d / "m2f_views.npz", device)
    del model
    ev.release_memory()

    min_px = int(ins.MIN_SHARE * W * H)
    gts = {r.id: mb.load_gt(mb.OUT / r.id) for r in df.itertuples()}

    def gt_fams(g):
        return {i: fam.get(b["ade"], "other") for i, b in enumerate(g["layout"]["boxes"])}

    def score(pred, g):
        pf = [p["family"] for p in pred["instances"]]
        rows, cnt = detection(g["inst"], gt_fams(g), pred["inst_map"], pf, min_px)
        return rows, cnt, semantics(pred["class_map"], g["labels"], fam)

    # --- dev sweep ------------------------------------------------------------------------------------- #
    sweep = []
    dev = df[df["split"] == "dev"]
    for s in SCORES:
        for mg in MERGES:
            tot = {"tp_any": 0, "tp_fam": 0, "n_gt": 0, "n_pred": 0}
            for r in dev.itertuples():
                pred = ins.merge_views(raw[r.id], id2label, W, H, score_min=s, merge_iou=mg)
                _, cnt, _ = score(pred, gts[r.id])
                for k in tot:
                    tot[k] += cnt[k]
            p, rc, f = prf(tot["tp_fam"], tot["n_gt"], tot["n_pred"])
            sweep.append({"score_min": s, "merge_iou": mg, "precision": p, "recall": rc, "f1": f,
                          "f1_any": prf(tot["tp_any"], tot["n_gt"], tot["n_pred"])[2], "n_pred": tot["n_pred"]})
    sweep = pd.DataFrame(sweep)
    best = sweep.sort_values(["f1", "f1_any"], ascending=False).iloc[0]
    S, M = float(best["score_min"]), float(best["merge_iou"])
    print(sweep.round(3).to_string(index=False))
    print("selected on dev: score_min", S, "merge_iou", M)

    # --- both methods on every room ---------------------------------------------------------------------- #
    per_room, per_gt, examples = [], [], []
    for r in tqdm(list(df.itertuples()), desc="scoring", unit="room"):
        d, g = mb.OUT / r.id, gts[r.id]
        v3 = ins.merge_views(raw[r.id], id2label, W, H, score_min=S, merge_iou=M)
        v3["class_map"] = ins.fill_unknown(v3["class_map"])
        seg2 = cv2.imread(str(d / "seg_v2.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        hyb = {**v3, "class_map": ins.hybrid_class_map(seg2, v3)}
        cv2.imwrite(str(d / "class_v3.png"), hyb["class_map"].astype(np.uint8))
        cv2.imwrite(str(d / "inst_v3.png"), (v3["inst_map"] + 1).astype(np.uint8))
        (d / "inst_v3.json").write_text(json.dumps(v3["instances"], indent=1))
        v2 = ins.v2_instances(seg2, id2label)
        for name, pred in (("v2", v2), ("v3_panoptic_only", v3), ("v3", hyb)):
            rows, cnt, sem = score(pred, g)
            per_room.append({"id": r.id, "room": r.room, "split": r.split, "method": name, **cnt, **sem})
            per_gt += [{"id": r.id, "room": r.room, "split": r.split, "method": name,
                        "gt_label": g["layout"]["boxes"][x["gt"]]["label"], **x} for x in rows]
        if r.split == "test" and r.layout_seed == 5:
            pano = cv2.resize(mb.rgb(d / "pano.png"), (W, H), interpolation=cv2.INTER_AREA)
            examples.append((r.prompt, pano, colorize(g["inst"], pano), colorize(v2["inst_map"], pano),
                             colorize(v3["inst_map"], pano)))
    pr, pg = pd.DataFrame(per_room), pd.DataFrame(per_gt)

    summ = []
    for (split, method), x in pr.groupby(["split", "method"]):
        p, rc, f = prf(x["tp_fam"].sum(), x["n_gt"].sum(), x["n_pred"].sum())
        pa, ra, fa = prf(x["tp_any"].sum(), x["n_gt"].sum(), x["n_pred"].sum())
        y = pg[(pg["split"] == split) & (pg["method"] == method)]
        summ.append({"split": split, "method": method, "rooms": len(x), "gt_instances": int(x["n_gt"].sum()),
                     "pred_instances": int(x["n_pred"].sum()), "precision_fam": p, "recall_fam": rc, "f1_fam": f,
                     "precision_any": pa, "recall_any": ra, "f1_any": fa,
                     "seam_objects": int(y["seam"].sum()), "seam_recall_any": y[y["seam"]]["hit_any"].mean(),
                     **{c: x[c].mean() for c in x.columns if c.startswith("iou_") or c == "miou"}})
    summ = pd.DataFrame(summ)
    by_class = (pg[pg["split"] == "test"].groupby(["gt_label", "method"])
                .agg(n=("hit_fam", "size"), recall_fam=("hit_fam", "mean"), recall_any=("hit_any", "mean"),
                     mean_best_iou=("best_iou", "mean")).reset_index()
                .pivot(index="gt_label", columns="method").sort_values(("n", "v2"), ascending=False))
    by_class.columns = [f"{a}_{b}" for a, b in by_class.columns]
    by_class = by_class[[c for c in by_class.columns if "panoptic_only" not in c]]
    by_class = by_class.drop(columns="n_v3").rename(columns={"n_v2": "n"})

    T, F = ROOT / "results/tables", ROOT / "results/figures"
    sweep.to_csv(T / f"inst_dev_sweep{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(T / f"inst_summary{tag}.csv", index=False, float_format="%.4f")
    by_class.to_csv(T / f"inst_by_class_test{tag}.csv", float_format="%.4f")
    pr.to_csv(T / f"inst_per_room{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())
    print(by_class.round(2).to_string())

    if examples:
        n = len(examples)
        fig, axes = plt.subplots(n, 4, figsize=(24, 3.3 * n), dpi=80)
        axes = np.atleast_2d(axes)
        for i, (title, *ims) in enumerate(examples):
            for j, im in enumerate(ims):
                axes[i, j].imshow(im)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(["Panorama (held-out)", "Ground truth (layout boxes)",
                                          "v2: SegFormer + connected components",
                                          "v3: Mask2Former panoptic, merged across views"][j], fontsize=13)
            axes[i, 0].text(0, -0.04, title, transform=axes[i, 0].transAxes, fontsize=11)
        fig.tight_layout()
        fig.savefig(F / f"inst_examples{tag}.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
