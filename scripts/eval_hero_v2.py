"""Polish round, step 1: hero view that shows the room's MAIN piece.

New rule (src/design_shots.py): the room type (from the prompt / the bench index) names the main classes (bed /
sofa / dining table / island or counter / bathtub or vanity); the main piece = the largest, least occluded instance
of those classes; the hero must show >= 50 % of it whenever any candidate yaw does, and a dev-tuned weight rewards
it being centred (1 at the frame centre, 0 at the border).
Before = the milestone-1 rule (results/tables/shots_weights_selected.json, no main-piece term).
Bench (40 layout rooms): ground truth main piece = layout box 0 (bed / sofa / counter / dining table / bathtub):
    gt_main_share     share of box 0 inside the hero frame (gt_main_in_view: >= 50 %)
    gt_main_centre    1 - |angle of box 0 from the frame centre| / (HFOV / 2) (0 if outside)
    gt_inst           layout boxes >= 50 % in view
The weight main_centre in {0, 1, 2, 4, 8} is chosen on the dev rooms (gt_main_share, then gt_main_centre, then
gt_inst); the held-out rooms are reported once. Demo rooms: hero before / after (the predicted main piece) and a
figure with both hero shots.
Outputs: results/tables/hero_v2_bench.csv, hero_v2_per_room.csv, hero_v2_demo.csv, hero_v2_weights.json;
figure results/figures/hero_v2_demo.png
Usage: python scripts/eval_hero_v2.py
"""

import json
import sys
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
from src import equirect as eq  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import room_box as rb  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
GRID = [0.0, 1.0, 2.0, 4.0, 8.0]


def gt_main_yaw(gt_inst: np.ndarray, box: dict) -> float | None:
    m = gt_inst == 0
    if m.sum() < 20:
        return None
    H, W = gt_inst.shape
    sa = np.cos(np.pi / 2 - np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    d = (eq.pixel_dirs(W, H)[m] * sa[m][:, None]).sum(0) @ box["R"].T @ box["Q"]
    return float(np.degrees(np.arctan2(d[0], d[2])) % 360)


def scan(d: Path, room: str, id2label, sets, ana_dir: Path, dist_name: str, gt: bool):
    dist = np.load(d / dist_name)
    cm = cv2.imread(str(ana_dir / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    inst = cv2.imread(str(ana_dir / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
    insts = json.loads((ana_dir / "inst_v3.json").read_text())
    box = rb.fit_box(dist, cm, id2label)
    fam = {int(i["id"]): i.get("family", i["label"]) for i in insts}
    main = ds.main_piece(inst, insts, dist, box, room)
    rows = ds.scan_yaws(box, cm, inst, fam, sets, dist=dist, main=main)
    if gt:
        gi = cv2.imread(str(d / "gt_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        gl = cv2.imread(str(d / "gt_labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        L = json.loads((d / "layout.json").read_text())
        gfam = {i: b["label"] for i, b in enumerate(L["boxes"])}
        gy = gt_main_yaw(gi, box)
        for r in rows:
            g = ds.view_measures(gl, gi, gfam, ds.view_to_cam(box, r["yaw"]), ds.HERO_FOV, sets, main_id=0)
            r["gt_main_share"] = float(g["main_complete"])
            r["gt_main_in_view"] = float(g["main_complete"] >= 0.5)
            r["gt_main_centre"] = 0.0 if gy is None or r["gt_main_share"] < 0.2 else \
                max(0.0, 1 - ds._gap(r["yaw"], gy) / (ds.HERO_FOV / 2))
            r["gt_inst"] = g["inst"]
    return rows, box, main, dist, cm, inst, insts


def choose(rows: list[dict], weights: dict, require: bool) -> dict:
    return ds.pick_heroes(rows, weights, 1, require_main=require)[0]


def main():
    id2label = erb.id2label_ade()
    sets = ds.class_sets(id2label)
    w1 = {**ds.DEFAULT_WEIGHTS, **json.loads((TAB / "shots_weights_selected.json").read_text())}
    bench = []
    for r in tqdm(list(mb.rooms().itertuples()), desc="bench"):
        d = mb.OUT / r.id
        rows, box, main_, *_ = scan(d, r.room.replace("_", " "), id2label, sets, d, "dist.npy", True)
        bench.append({"id": r.id, "split": r.split, "rows": rows, "main": main_})
    cols = ["gt_main_share", "gt_main_in_view", "gt_main_centre", "gt_inst", "main_complete"]
    sweep = []
    for wc in GRID:
        w = {**w1, "main_centre": wc}
        sel = [choose(b["rows"], w, True) for b in bench if b["split"] == "dev"]
        sweep.append({"main_centre": wc, **{c: float(np.mean([s.get(c, 0) for s in sel])) for c in cols}})
    sweep = pd.DataFrame(sweep)
    print(sweep.round(3).to_string())
    best = sweep.sort_values(["gt_main_share", "gt_main_centre", "gt_inst"], ascending=False, kind="stable").iloc[0]
    w2 = {**w1, "main_centre": float(best["main_centre"])}
    (TAB / "hero_v2_weights.json").write_text(json.dumps({k: float(v) for k, v in w2.items()}, indent=1))
    per = []
    for b in bench:
        before, after = choose(b["rows"], w1, False), choose(b["rows"], w2, True)
        for name, s in (("before (milestone 1)", before), ("after (main piece)", after)):
            per.append({"id": b["id"], "split": b["split"], "rule": name, "yaw": s["yaw"],
                        "main_found": b["main"] is not None, **{c: s.get(c, 0) for c in cols}})
    per = pd.DataFrame(per)
    per.to_csv(TAB / "hero_v2_per_room.csv", index=False)
    summ = per.groupby(["split", "rule"])[cols + ["main_found"]].mean().reset_index()
    summ = pd.concat([summ, sweep.assign(split="dev sweep", rule=lambda x: "main_centre=" + x["main_centre"].astype(str))
                      .drop(columns="main_centre")])
    summ.to_csv(TAB / "hero_v2_bench.csv", index=False)
    print(summ.round(3).to_string())
    # demo rooms
    demo, figs = [], []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        room = ds.room_type_of(prompt)
        rows, box, main_, dist, cm, inst, insts = scan(d, room, id2label, sets, d / "v3", "distance.npy", False)
        before, after = choose(rows, w1, False), choose(rows, w2, True)
        hires = mb.rgb(d / "v4/pano_4096_detail.png")
        ims = [ds.render_view(hires, ds.view_to_cam(box, s["yaw"]), 1024, 768, ds.HERO_FOV) for s in (before, after)]
        demo.append({"room": prompt, "main_piece": None if main_ is None else main_["label"],
                     "main_occlusion": None if main_ is None else main_["occlusion"],
                     "yaw_before": before["yaw"], "yaw_after": after["yaw"],
                     "main_in_view_before": before["main_complete"], "main_in_view_after": after["main_complete"],
                     "main_centre_before": before["main_centre"], "main_centre_after": after["main_centre"],
                     "inst_before": before["inst"], "inst_after": after["inst"]})
        figs.append((prompt, *ims))
    dm = pd.DataFrame(demo)
    dm.to_csv(TAB / "hero_v2_demo.csv", index=False)
    print(dm.round(2).to_string())
    fig, axes = plt.subplots(len(figs), 2, figsize=(12, 4.4 * len(figs)), dpi=70)
    for i, (t, a, b) in enumerate(figs):
        axes[i, 0].imshow(a); axes[i, 0].set_title(f"hero before (milestone 1) - {t}", fontsize=10)
        axes[i, 1].imshow(b); axes[i, 1].set_title(f"hero after (main piece: {demo[i]['main_piece']})", fontsize=10)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "hero_v2_demo.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
