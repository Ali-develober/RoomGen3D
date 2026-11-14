"""PLAN_V4 milestone 1: readable 2D design shots - hero-view selection vs. a simple baseline.

A. Bench (40 layout-first rooms of PLAN_V3, outputs/bench/<id>): the v3 room box, class map and instances
   (class_v3.png, inst_v3.png) are what the pipeline sees; the layout's ground truth (gt_inst.png = furniture
   box index, gt_labels.png = ADE20K class, layout.json) is used ONLY to measure the chosen views.
   For every candidate yaw (every 5 degrees, 80 degree HFOV, level, frame B) both are measured:
       pred_*  on the predicted maps (what the hero score sees)
       gt_*    on the layout ground truth: furniture boxes >= 50 % in view (gt_inst), distinct classes among
               them (gt_fam), frame share of furniture (gt_furn) and wall (gt_wall)
   Weight grid (score_view) chosen on the DEVELOPMENT rooms by the mean of gt_inst (tie: gt_fam), then
   reported once on the HELD-OUT rooms against
       baseline   "always the first wall": the yaw of wall view A, same 80 degree FOV
       random     the mean over all candidate yaws (expected value of a random direction)
       default    hero with DEFAULT weights (fixed before the sweep)
       oracle     the yaw with the most GT furniture in view (upper bound)
B. The six demo rooms (outputs/room360/<room>, v3 maps): predicted measures (no GT), shots saved with the
   v4 pipeline outputs (outputs/room360/<room>/v4), one figure per room (hero + walls + plan with view
   cones) and an overview; before/after figure panorama vs. shots.
Outputs: results/tables/shots_scan_bench.csv (per room x yaw), shots_weights_dev.csv, shots_selection.csv
(summary), shots_selection_per_room.csv, shots_demo.csv; figures shots_<room>.png, shots_overview.png,
shots_before_after.png.
Usage: python scripts/eval_design_shots.py [--limit N]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import design_shots as ds  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v4 as p4  # noqa: E402
from src import room_box as rb  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
GRID = {"inst": [0.25, 0.5, 1.0], "fam": [0.5, 1.0, 2.0], "furn": [2.0, 4.0], "wall": [0.0, 1.0], "cf": [1.0, 3.0],
        "anchor": [0.0, 1.0, 2.0], "anchor_share": [0.0, 5.0, 10.0], "depth": [0.0, 0.5, 1.0]}


def load_v3(d: Path, cm_name="class_v3.png", inst_name="inst_v3.png", json_name="inst_v3.json"):
    cm = cv2.imread(str(d / cm_name), cv2.IMREAD_UNCHANGED).astype(np.int32)
    inst = cv2.imread(str(d / inst_name), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
    return cm, inst, json.loads((d / json_name).read_text())


def scan_room(d: Path, id2label: dict, sets: dict) -> tuple[pd.DataFrame, dict]:
    dist = np.load(d / "dist.npy")
    cm, inst, insts = load_v3(d)
    box = rb.fit_box(dist, cm, id2label)
    fam = {int(i["id"]): i.get("family", i["label"]) for i in insts}
    gt_inst = cv2.imread(str(d / "gt_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
    gt_lab = cv2.imread(str(d / "gt_labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    L = json.loads((d / "layout.json").read_text())
    gfam = {i: b["label"] for i, b in enumerate(L["boxes"])}
    rows = []
    for y in np.arange(0.0, 360.0, ds.HERO_STEP):
        M = ds.view_to_cam(box, y)
        p = ds.view_measures(cm, inst, fam, M, ds.HERO_FOV, sets, dist=dist)
        g = ds.view_measures(gt_lab, gt_inst, gfam, M, ds.HERO_FOV, sets)
        g["anchor"] = float(0 in g["inst_ids"])          # layout box 0 = the room's main piece
        rows.append({"yaw": float(y), **{f"pred_{k}": v for k, v in p.items() if k != "inst_ids"},
                     **{f"gt_{k}": v for k, v in g.items() if k != "inst_ids"}})
    return pd.DataFrame(rows), box


def pick(df: pd.DataFrame, weights: dict) -> int:
    """Row index of the hero yaw for these weights (same rule as design_shots.pick_heroes, first hero)."""
    m = {c[5:]: df[c].to_numpy() for c in df.columns if c.startswith("pred_")}
    return int(np.argmax(ds.score_view(m, weights)))


def nearest_yaw_row(df: pd.DataFrame, yaw: float) -> int:
    d = np.abs((df["yaw"].to_numpy() - yaw + 180) % 360 - 180)
    return int(np.argmin(d))


def bench(limit: int | None):
    id2label = erb.id2label_ade()
    sets = ds.class_sets(id2label)
    rooms = mb.rooms()
    if limit:
        rooms = pd.concat([rooms[rooms.split == "dev"].head(limit), rooms[rooms.split == "test"].head(limit)])
    scans, base_yaw = {}, {}
    for r in tqdm(list(rooms.itertuples()), desc="bench scans"):
        df, box = scan_room(mb.OUT / r.id, id2label, sets)
        df.insert(0, "id", r.id)
        df.insert(1, "split", r.split)
        scans[r.id] = df
        base_yaw[r.id] = ds.wall_yaws(box)["A"]
    pd.concat(scans.values()).to_csv(TAB / "shots_scan_bench.csv", index=False)

    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    dev = [i for i in scans if scans[i]["split"].iat[0] == "dev"]
    wrows = []
    for c in combos:
        sel = [scans[i].iloc[pick(scans[i], c)] for i in dev]
        wrows.append({**c, "gt_anchor": np.mean([s["gt_anchor"] for s in sel]),
                      "gt_inst": np.mean([s["gt_inst"] for s in sel]), "gt_fam": np.mean([s["gt_fam"] for s in sel]),
                      "gt_furn": np.mean([s["gt_furn"] for s in sel])})
    # objective: the main piece fully in view (GT layout box 0), then GT furniture pieces in view
    wdf = pd.DataFrame(wrows).sort_values(["gt_anchor", "gt_inst", "gt_fam"], ascending=False, kind="stable")
    wdf.to_csv(TAB / "shots_weights_dev.csv", index=False)
    best = {k: float(wdf.iloc[0][k]) for k in keys}
    print("selected weights (dev):", best, "dev gt_inst", wdf.iloc[0]["gt_inst"])

    per = []
    cols = ["pred_inst", "pred_fam", "pred_furn", "pred_wall", "pred_ceiling", "pred_floor", "pred_anchor",
            "pred_depth", "gt_anchor", "gt_inst", "gt_fam", "gt_furn", "gt_wall"]
    for i, df in scans.items():
        choices = {"baseline_first_wall": nearest_yaw_row(df, base_yaw[i]), "hero_default": pick(df, {}),
                   "hero_tuned": pick(df, best),
                   "oracle_gt": int(np.lexsort((-df["gt_inst"], -df["gt_anchor"]))[0])}
        for method, j in choices.items():
            per.append({"id": i, "split": df["split"].iat[0], "method": method, "yaw": df["yaw"].iat[j],
                        **{c: df[c].iat[j] for c in cols}})
        per.append({"id": i, "split": df["split"].iat[0], "method": "random_mean", "yaw": np.nan,
                    **{c: df[c].mean() for c in cols}})
    per = pd.DataFrame(per)
    per.to_csv(TAB / "shots_selection_per_room.csv", index=False)
    summ = per.groupby(["split", "method"])[cols].mean().reset_index()
    summ["rooms"] = per.groupby(["split", "method"]).size().values
    # paired hero - baseline (bootstrap 95% CI over rooms)
    prs = []
    for split in summ["split"].unique():
        for method in ("hero_default", "hero_tuned"):
            a = per[(per.split == split) & (per.method == method)].set_index("id")
            b = per[(per.split == split) & (per.method == "baseline_first_wall")].set_index("id")
            for col in ("gt_anchor", "gt_inst", "pred_inst"):
                dif = (a[col] - b[col]).to_numpy()
                rng = np.random.default_rng(0)
                boots = [rng.choice(dif, len(dif)).mean() for _ in range(2000)]
                prs.append({"split": split, "method": method, "measure": col, "diff_vs_baseline": dif.mean(),
                            "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                            "better": int((dif > 0).sum()), "equal": int((dif == 0).sum()),
                            "worse": int((dif < 0).sum())})
    prs = pd.DataFrame(prs)
    prs.to_csv(TAB / "shots_selection_paired.csv", index=False)
    print(prs.round(2).to_string())
    summ["furn_plus_wall_pred"] = summ["pred_furn"] + summ["pred_wall"]
    summ["furn_plus_wall_gt"] = summ["gt_furn"] + summ["gt_wall"]
    summ.to_csv(TAB / "shots_selection.csv", index=False)
    print(summ.round(3).to_string())
    return best


def demo(weights: dict):
    id2label = erb.id2label_ade()
    sets = ds.class_sets(id2label)
    rows, figs = [], []
    for prompt, seed in tqdm(pl.DEMO_PROMPTS, desc="demo rooms"):
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        dist = np.load(d / "distance.npy")
        cm, inst, insts = load_v3(d / "v3")
        tex = mb.rgb(d / "texture.png")
        pano = mb.rgb(d / "pano.png")
        ana = {"class_map": cm, "inst_map": inst, "instances": insts}
        t0 = time.perf_counter()
        res = p4.build(dist, tex, ana, id2label, shot_weights=weights)
        t_build = time.perf_counter() - t0
        p4.save(res, d / "v4", title=prompt, pano=tex)
        shots = res["shots"]["shots"]
        fam = {int(i["id"]): i.get("family", i["label"]) for i in insts}
        base = ds.view_measures(cm, inst, fam, ds.view_to_cam(res["box"], ds.wall_yaws(res["box"])["A"]), ds.HERO_FOV,
                                sets, dist=dist)
        hero = shots[0]["measures"]
        rows.append({"room": prompt, "hero_yaw": shots[0]["yaw"], "n_hero": sum(s["kind"] == "hero" for s in shots),
                     "baseline_inst": base["inst"], "hero_inst": hero["inst"], "baseline_fam": base["fam"],
                     "hero_fam": hero["fam"], "baseline_anchor": base["anchor"], "hero_anchor": hero["anchor"],
                     "baseline_depth_m": base["depth"], "hero_depth_m": hero["depth"], "baseline_furn_wall": base["furn"] + base["wall"],
                     "hero_furn_wall": hero["furn"] + hero["wall"], "baseline_ceil_floor": base["ceiling"] + base["floor"],
                     "hero_ceil_floor": hero["ceiling"] + hero["floor"], "shots_seconds": res["shots"]["seconds"],
                     "v4_build_seconds": t_build, "camera_height_m": res["box"]["camera_height"]})
        figs.append((prompt, pano, shots, res["plan_image"]))
        # per-room figure: hero(s) + four walls + plan with cones
        n = len(shots)
        fig = plt.figure(figsize=(4.2 * 4, 3.3 * 2 + 0.6), dpi=80)
        gsp = fig.add_gridspec(2, 4)
        for k, s in enumerate(shots[:-4]):
            ax = fig.add_subplot(gsp[0, k])
            ax.imshow(s["image"]); ax.set_title(ds.caption(s), fontsize=10); ax.axis("off")
        for k, s in enumerate(shots[-4:]):
            ax = fig.add_subplot(gsp[1, k])
            ax.imshow(s["image"]); ax.set_title(ds.caption(s), fontsize=10); ax.axis("off")
        ax = fig.add_subplot(gsp[0, n - 4:])
        ax.imshow(res["plan_image"]); ax.set_title("floor plan with camera and view cones", fontsize=10); ax.axis("off")
        fig.suptitle(prompt, fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG / f"shots_{pl.slug(prompt, seed)}.png", facecolor="white")
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "shots_demo.csv", index=False)
    print(df.round(2).to_string())

    fig, axes = plt.subplots(len(figs), 3, figsize=(18, 3.6 * len(figs)), dpi=70,
                             gridspec_kw={"width_ratios": [2, 1.33, 1.2]})
    for i, (title, pano, shots, plan) in enumerate(figs):
        axes[i, 0].imshow(pano); axes[i, 0].set_title(f"before (v3): raw panorama - {title}", fontsize=10)
        axes[i, 1].imshow(shots[0]["image"]); axes[i, 1].set_title("after (v4): hero shot H", fontsize=10)
        axes[i, 2].imshow(plan); axes[i, 2].set_title("plan with view cones", fontsize=10)
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "shots_before_after.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(len(figs), 5, figsize=(20, 3.1 * len(figs)), dpi=70)
    for i, (title, pano, shots, plan) in enumerate(figs):
        for k, s in enumerate([shots[0]] + shots[-4:]):
            axes[i, k].imshow(s["image"]); axes[i, k].axis("off")
            axes[i, k].set_title((title + "  -  " if k == 0 else "") + ds.caption(s), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "shots_overview.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-demo", action="store_true")
    a = ap.parse_args()
    best = bench(a.limit)
    (TAB / "shots_weights_selected.json").write_text(json.dumps(best, indent=1))
    if not a.skip_demo:
        demo(best)


if __name__ == "__main__":
    main()
