"""PLAN_V3 milestone 4: floor plan drawing, v2 vs v3.

A. The six demo rooms of milestone 4/5 (outputs/room360/<room>, v2 SegFormer classes and DAP distance already
   there): Mask2Former instances, v3 room (src/pipeline_v3.py), plan PNG / SVG / DXF in outputs/room360/<room>/v3/.
   Figure results/figures/plan_v2_v3.png: the v2 plan (floorplan.png) next to the v3 plan, same panorama.
B. The 20 held-out bench rooms: plan quality numbers for both
       rotation_deg         how far the drawn walls are from the page axes (v2: the plan's dominant direction)
       inferred_sides       sides without a measured wall (hatched)
       furniture_overlaps   pairs of furniture footprints that overlap (IoU > 0.02)
       outside_share        share of furniture footprint area outside the room outline
       label_failures       labels that found no free position (v3 only; v2 places no labels with checks)
       openings             v3 doors / windows vs. the layout's: matched if same wall, same kind and centre within
                            0.5 m -> precision / recall
Outputs: results/tables/plan_quality.csv, plan_quality_per_room.csv, plan_demo_v3.csv; figure plan_v2_v3.png
Usage: python scripts/plan_v3_eval.py
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import evaluate as ev  # noqa: E402
from src import furniture as fu  # noqa: E402
from src import instances as ins  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import plan_v3 as pv  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import room_mesh as rm  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402


def overlaps_outside(rects: list[np.ndarray], room: np.ndarray) -> tuple[int, float]:
    n = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i].astype(np.float32), rects[j].astype(np.float32)
            inter, _ = cv2.intersectConvexConvex(a, b)
            if inter > 0 and inter / (cv2.contourArea(a) + cv2.contourArea(b) - inter) > 0.02:
                n += 1
    tot = sum(cv2.contourArea(r.astype(np.float32)) for r in rects)
    inside = sum(cv2.intersectConvexConvex(r.astype(np.float32), room.astype(np.float32))[0] for r in rects)
    return n, float(1 - inside / tot) if tot > 0 else np.nan


def gt_openings(L: dict) -> list[dict]:
    C = np.array(L["camera"])
    wall = {"N": "z1", "S": "z0", "E": "x1", "W": "x0"}
    return [{"kind": o["kind"], "wall": wall[o["wall"]],
             "c": o["c"] - (C[0] if o["wall"] in ("N", "S") else C[2])} for o in L["openings"]]


def main():
    device = torch.device("cuda")
    id2label = erb.id2label_ade()
    panoptic = ins.load_panoptic(device)

    # --- A: demo rooms ------------------------------------------------------------------------------------ #
    demo, figs = [], []
    for prompt, seed in tqdm(pl.DEMO_PROMPTS, desc="demo rooms"):
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        out = d / "v3"
        out.mkdir(exist_ok=True)
        dist = np.load(d / "distance.npy")
        seg = cv2.imread(str(d / "class_map.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        if (out / "inst_v3.png").exists():
            ana = {"class_map": cv2.imread(str(out / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
                   "inst_map": cv2.imread(str(out / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
                   "instances": json.loads((out / "inst_v3.json").read_text())}
        else:
            ana = p3v.analyse(panoptic, mb.rgb(d / "pano.png"), seg, id2label, device)
            cv2.imwrite(str(out / "class_v3.png"), ana["class_map"].astype(np.uint8))
            cv2.imwrite(str(out / "inst_v3.png"), (ana["inst_map"] + 1).astype(np.uint8))
            (out / "inst_v3.json").write_text(json.dumps(ana["instances"], indent=1))
        res = p3v.build(dist, mb.rgb(d / "texture.png"), ana, id2label)
        p3v.save(res, out, title=prompt)
        b = res["box"]
        demo.append({"prompt": prompt, "area_m2": b["area"], "width_m": b["width"], "depth_m": b["depth"],
                     "sides_detected": b["n_detected"], "solids": len(res["solids"]),
                     "openings": len(res["openings"]), **{f"plan_{k}": v for k, v in res["plan_stats"].items()}})
        figs.append((prompt, mb.rgb(d / "floorplan.png"), res["plan_image"]))
    del panoptic
    ev.release_memory()

    fig, axes = plt.subplots(len(figs), 2, figsize=(16, 7.2 * len(figs)), dpi=70)
    for i, (title, a, b) in enumerate(figs):
        for j, im in enumerate((a, b)):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
        axes[i, 0].set_title(f"v2 plan: {title}", fontsize=14)
        axes[i, 1].set_title("v3 plan (same panorama)", fontsize=14)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/plan_v2_v3.png", facecolor="white")
    plt.close(fig)

    # --- B: held-out bench rooms ------------------------------------------------------------------------- #
    rows = []
    for r in tqdm([r for r in mb.rooms().itertuples() if r.split == "test"], desc="bench plans"):
        d = mb.OUT / r.id
        g = mb.load_gt(d)
        dist = np.load(d / "dist.npy")
        cm2 = cv2.imread(str(d / "seg_v2.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cm3 = cv2.imread(str(d / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        inst = cv2.imread(str(d / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        insts = json.loads((d / "inst_v3.json").read_text())
        R2, _ = rm.gravity_rotation(dist, cm2, id2label)
        p2 = rm.plan_360(dist, cm2, id2label, R2)
        rects2 = [f["rect"] for f in p2["furniture"]]
        n2, o2 = overlaps_outside(rects2, p2["room"]["corners"])
        dom = abs(float(p2["manhattan"].get("dominant_deg", 0.0))) % 90
        rows.append({"id": r.id, "method": "v2", "rotation_deg": min(dom, 90 - dom),
                     "inferred_sides": 4 - p2["room"]["n_observed_sides"], "furniture": len(rects2),
                     "furniture_overlaps": n2, "outside_share": o2, "openings_pred": len(p2["openings"])})
        res = p3v.build(dist, mb.rgb(d / "texture.png"), {"class_map": cm3, "inst_map": inst, "instances": insts}, id2label)
        box = res["box"]
        rects3 = [fu._corners(s) for s in res["solids"]]
        room3 = np.array([[box["x"][0], box["z"][0]], [box["x"][1], box["z"][0]], [box["x"][1], box["z"][1]],
                          [box["x"][0], box["z"][1]]])
        n3, o3 = overlaps_outside(rects3, room3)
        _, st = pv.draw_plan(box, res["solids"], res["openings"])
        gto = gt_openings(g["layout"])
        used, tp = set(), 0
        for o in res["openings"]:
            c = (o["s0"] + o["s1"]) / 2
            for k, q in enumerate(gto):
                if k not in used and q["wall"] == o["wall"] and q["kind"] == o["kind"] and abs(q["c"] - c) < 0.5:
                    used.add(k)
                    tp += 1
                    break
        rows.append({"id": r.id, "method": "v3", "rotation_deg": 0.0, "inferred_sides": 4 - box["n_detected"],
                     "furniture": len(rects3), "furniture_overlaps": n3, "outside_share": o3,
                     "label_failures": st["label_failures"], "openings_pred": len(res["openings"]),
                     "openings_gt": len(gto), "openings_tp": tp})
    df = pd.DataFrame(rows)
    summ = df.groupby("method").agg(rooms=("id", "size"), rotation_deg=("rotation_deg", "mean"),
                                    inferred_sides=("inferred_sides", "mean"), furniture=("furniture", "mean"),
                                    furniture_overlaps=("furniture_overlaps", "mean"),
                                    outside_share=("outside_share", "mean"), openings_pred=("openings_pred", "sum")).reset_index()
    v3 = df[df["method"] == "v3"]
    summ.loc[summ["method"] == "v3", "label_failures"] = v3["label_failures"].sum()
    summ.loc[summ["method"] == "v3", "openings_precision"] = v3["openings_tp"].sum() / max(v3["openings_pred"].sum(), 1)
    summ.loc[summ["method"] == "v3", "openings_recall"] = v3["openings_tp"].sum() / max(v3["openings_gt"].sum(), 1)
    T = ROOT / "results/tables"
    df.to_csv(T / "plan_quality_per_room.csv", index=False, float_format="%.4f")
    summ.to_csv(T / "plan_quality.csv", index=False, float_format="%.4f")
    pd.DataFrame(demo).to_csv(T / "plan_demo_v3.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(pd.DataFrame(demo).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
