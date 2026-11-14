"""Milestone 6 (research experiment): layout first, then the image, then exact 3D.

Rooms: the six demo prompts x layout seeds 1 and 2 = 12 layouts (src/layout.py), generation seed =
the prompt's demo seed. Per room:
  layout            JSON, ground-truth floor plan, equirectangular distance + ADE20K labels from the camera
  panoramas         ControlNet depth (scale 1) + panorama LoRA ("layout"), and the same pipeline and seed
                    with the ControlNet switched off (scale 0, "free"): the baseline for adherence
  adherence         DAP (+ wrap fix) distance of the generated panorama vs. the layout distance:
                    AbsRel and delta < 1.25, as predicted and after one median scale; SegFormer
                    (14 views) labels vs. layout labels: IoU of wall / floor / ceiling / furniture, mIoU
  plan check        the 360 floor plan pipeline (milestones 3 + 5: DAP + SegFormer + floorplan) run on the
                    layout panorama, compared with the true layout: area and side-length error.
                    (This is the only place where the plan pipeline meets ground truth.)
  3D                the panorama texture projected onto the EXACT layout geometry (room_mesh on the
                    layout distance and labels); empty share inside (centre, 1 m) and dollhouse, compared
                    with the milestone-4 mesh of the same prompt; CLIP score of crops of the layout,
                    free and milestone-4 panoramas
Outputs: outputs/layout/<room>_L<seed>/..., results/tables/layout_*.csv, results/figures/layout_*.png
Usage: python scripts/eval_layout.py [--limit 2]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import analysis as an  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import layout as ly  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import room_mesh as rm  # noqa: E402
from src import textured_mesh as tm  # noqa: E402
from src.utils import set_seed  # noqa: E402
import eval_room3d as er  # noqa: E402

OUT = ROOT / "outputs" / "layout"
LAYOUT_SEEDS = [1, 2]
GROUPS = {"wall": [0], "floor": [3, 28], "ceiling": [5]}


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def save_rgb(p, a):
    cv2.imwrite(str(p), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def groups(lab: np.ndarray, id2label: dict) -> dict:
    furn = fp.class_ids(id2label, set(fp.FURNITURE))
    g = {k: np.isin(lab, v) for k, v in GROUPS.items()}
    g["furniture"] = np.isin(lab, furn)
    return g


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def depth_adherence(pred, gt):
    m = np.isfinite(pred) & (gt > 0.1)
    p, g = pred[m], gt[m]
    out = {}
    for name, x in (("raw", p), ("scaled", p * np.median(g / p))):
        r = np.maximum(x / g, g / x)
        out[f"absrel_{name}"] = float(np.mean(np.abs(x - g) / g))
        out[f"delta1_{name}"] = float(np.mean(r < 1.25))
    out["scale_gt_over_pred"] = float(np.median(g / p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    tag = "" if args.limit is None else "_smoke"
    prompts = pl.DEMO_PROMPTS[: args.limit] if args.limit else pl.DEMO_PROMPTS
    items = []
    for prompt, seed in prompts:
        for ls in LAYOUT_SEEDS:
            d = OUT / f"{pl.slug(prompt, seed)}_L{ls}"
            d.mkdir(parents=True, exist_ok=True)
            L = ly.generate_layout(ly.room_type(prompt), ls)
            ly.save_json(L, d / "layout.json")
            dist, lab = ly.render(L)
            np.save(d / "layout_dist.npy", dist)
            cv2.imwrite(str(d / "layout_labels.png"), lab.astype(np.uint8))
            items.append((prompt, seed, ls, d, L))

    # --- GPU stages, one model at a time ------------------------------------------------------- #
    timing = {}
    todo = [it for it in items if not (it[3] / "pano_free.png").exists()]
    if todo:
        ev.release_memory()
        torch.cuda.reset_peak_memory_stats()
        pipe = gp.load_controlnet_pipeline(device)
        for prompt, seed, ls, d, L in tqdm(todo, desc="panoramas", unit="room"):
            c = ly.control_image(np.load(d / "layout_dist.npy"))
            img, t = gp.timed(gp.generate_pano_controlled, pipe, prompt, seed, c, 1.0)
            save_rgb(d / "pano_layout.png", img)
            save_rgb(d / "pano_free.png", gp.generate_pano_controlled(pipe, prompt, seed, c, 0.0))
            (d / "timing.json").write_text(json.dumps({"generate_s": t}))
        timing["vram_controlnet_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        del pipe
        ev.release_memory()
    todo = [it for it in items if not (it[3] / "texture.png").exists()]
    if todo:
        net = gp.load_esrgan(device)
        for prompt, seed, ls, d, L in todo:
            save_rgb(d / "texture.png", gp.esrgan_upscale(net, rgb(d / "pano_layout.png")))
        del net
        ev.release_memory()
    todo = [it for it in items if not (it[3] / "dist_free.npy").exists()]
    if todo:
        dap, _ = pdp.load_dap(device)
        for prompt, seed, ls, d, L in tqdm(todo, desc="DAP", unit="room"):
            for k in ("layout", "free"):
                np.save(d / f"dist_{k}.npy", pdp.dap_depth_wrap(dap, rgb(d / f"pano_{k}.png"), device))
        del dap
        ev.release_memory()
    seg = an.load_segmenter(device)
    id2label = seg[0].config.id2label
    for prompt, seed, ls, d, L in tqdm(items, desc="segmentation", unit="room"):
        for k in ("layout", "free"):
            f = d / f"seg_{k}.png"
            if not f.exists():
                cv2.imwrite(str(f), pdp.segment_pano(*seg, rgb(d / f"pano_{k}.png"), device).astype(np.uint8))
    clip_model, clip_proc = ev.load_clip(device)

    def clip_of(pano, caption):
        crops = eq.horizontal_views(pano)
        paths = []
        for i, c in enumerate(crops):
            p = OUT / f"_clip_{i}.png"
            save_rgb(p, c)
            paths.append(str(p))
        return float(ev.clip_scores(clip_model, clip_proc, paths, [caption] * 4, device).mean())

    # --- metrics ------------------------------------------------------------------------------- #
    rows, mesh_rows, figs = [], [], []
    for prompt, seed, ls, d, L in tqdm(items, desc="metrics", unit="room"):
        gt_d = np.load(d / "layout_dist.npy")
        gt_l = cv2.imread(str(d / "layout_labels.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        gg = groups(gt_l, id2label)
        r = {"prompt": prompt, "seed": seed, "layout_seed": ls, "room_w": L["W"], "room_d": L["D"],
             "area_true_m2": L["W"] * L["D"], "boxes": len(L["boxes"])}
        r.update(json.loads((d / "timing.json").read_text()) if (d / "timing.json").exists() else {})
        for k in ("layout", "free"):
            pred = np.load(d / f"dist_{k}.npy")
            for m, v in depth_adherence(pred, gt_d).items():
                r[f"{k}_{m}"] = v
            pg = groups(cv2.imread(str(d / f"seg_{k}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32), id2label)
            ious = {g: iou(pg[g], gg[g]) for g in gg}
            for g, v in ious.items():
                r[f"{k}_iou_{g}"] = v
            r[f"{k}_miou"] = float(np.nanmean(list(ious.values())))
            r[f"{k}_clip"] = clip_of(rgb(d / f"pano_{k}.png"), prompt)
        # plan pipeline on the layout panorama vs. the true layout
        dist = np.load(d / "dist_layout.npy")
        cm = cv2.imread(str(d / "seg_layout.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        R, _ = rm.gravity_rotation(dist, cm, id2label)
        plan = rm.plan_360(dist, cm, id2label, R)
        room = plan["room"]
        sides_p, sides_t = sorted([room["width"], room["depth"]]), sorted([L["W"], L["D"]])
        r.update({"plan_area_m2": room["area"], "plan_area_rel_err": room["area"] / (L["W"] * L["D"]) - 1,
                  "plan_side_rel_err": float(np.mean([abs(a / b - 1) for a, b in zip(sides_p, sides_t)])),
                  "plan_inferred_sides": 4 - room["n_observed_sides"],
                  "camera_height_pred_m": plan["frame"]["d"], "camera_height_true_m": ly.CAMERA_HEIGHT})
        rows.append(r)
        # exact-geometry mesh: panorama texture on the layout distance + labels
        t0 = time.perf_counter()
        mesh = rm.build_room_mesh(gt_d, rgb(d / "texture.png"), gt_l, id2label)
        r["mesh_s"] = time.perf_counter() - t0
        tm.export_glb(mesh, d / "room.glb")
        tm.export_glb(rm.dollhouse(mesh), d / "dollhouse.glb")
        for x in er.inside_metrics(mesh, prompt, "layout") + er.dollhouse_metrics(mesh, prompt, "layout")[0]:
            mesh_rows.append({**x, "layout_seed": ls})
        if ls == LAYOUT_SEEDS[0]:
            gt_plan = fp.draw_floorplan(ly.plan_dict(L), title=f"Layout: {prompt}", note="Ground-truth layout")
            pred_plan = fp.draw_floorplan(plan, title="Plan recovered from the generated panorama")
            top = None
            try:
                import floorplan_360 as f3
                top = f3.dollhouse_topview(rm.dollhouse(mesh), 400)
            except Exception:
                pass
            figs.append((prompt, gt_plan, ly.control_image(gt_d), rgb(d / "pano_layout.png"), rgb(d / "pano_free.png"),
                         pred_plan, top))
    del seg, clip_model
    ev.release_memory()

    df = pd.DataFrame(rows)
    # milestone-4 comparison on the same prompts: CLIP of the SD 2.1 panoramas and their empty shares
    m4 = pd.read_csv(ROOT / "results/tables/room3d_empty_per_view.csv")
    m4 = m4[m4["mode"] == "room360"]
    mesh = pd.DataFrame(mesh_rows)
    summ = []
    for name, g in (("layout (ControlNet)", "layout"), ("free (same pipeline, no ControlNet)", "free")):
        summ.append({"panorama": name, "depth_absrel_raw": df[f"{g}_absrel_raw"].mean(),
                     "depth_absrel_scaled": df[f"{g}_absrel_scaled"].mean(),
                     "delta1_scaled": df[f"{g}_delta1_scaled"].mean(), "miou": df[f"{g}_miou"].mean(),
                     **{f"iou_{k}": df[f"{g}_iou_{k}"].mean() for k in ("wall", "floor", "ceiling", "furniture")},
                     "clip": df[f"{g}_clip"].mean()})
    summ = pd.DataFrame(summ)
    mesh_summary = []
    for mode, g in (("layout-first (exact geometry)", mesh), ("milestone 4 (estimated geometry)", m4)):
        ins = g[g["view"] == "inside"]
        mesh_summary.append({"mesh": mode, "empty_inside_centre": ins[ins["offset_m"] == 0]["empty"].mean(),
                             "empty_inside_1m": ins[ins["offset_m"] == 1.0]["empty"].mean(),
                             "empty_dollhouse": g[g["view"] == "dollhouse"]["empty"].mean()})
    mesh_summary = pd.DataFrame(mesh_summary)
    plan_summary = pd.DataFrame([{"rooms": len(df), "area_rel_err_mean": df["plan_area_rel_err"].mean(),
                                  "area_abs_rel_err_mean": df["plan_area_rel_err"].abs().mean(),
                                  "area_abs_rel_err_median": df["plan_area_rel_err"].abs().median(),
                                  "side_rel_err_mean": df["plan_side_rel_err"].mean(),
                                  "inferred_sides_mean": df["plan_inferred_sides"].mean(),
                                  "camera_height_pred_median": df["camera_height_pred_m"].median(),
                                  "dap_scale_gt_over_pred_median": df["layout_scale_gt_over_pred"].median()}])
    tables, figd = ROOT / "results/tables", ROOT / "results/figures"
    df.to_csv(tables / f"layout_per_room{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(tables / f"layout_adherence{tag}.csv", index=False, float_format="%.4f")
    mesh_summary.to_csv(tables / f"layout_mesh{tag}.csv", index=False, float_format="%.4f")
    plan_summary.to_csv(tables / f"layout_plan_check{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())
    print(mesh_summary.round(4).to_string(index=False))
    print(plan_summary.round(3).T.to_string())
    print("timing", timing, "generate_s mean", df.get("generate_s", pd.Series(dtype=float)).mean(), "mesh_s", df["mesh_s"].mean())

    cols = ["Layout (ground truth)", "Depth condition", "Panorama, layout-conditioned", "Same seed, no ControlNet",
            "Plan recovered from the panorama", "Dollhouse on exact geometry"]
    fig, axes = plt.subplots(len(figs), 6, figsize=(30, 4.2 * len(figs)), dpi=80,
                             gridspec_kw={"width_ratios": [1, 1.6, 1.6, 1.6, 1, 1]})
    axes = np.atleast_2d(axes)
    for i, row in enumerate(figs):
        for j, im in enumerate(row[1:]):
            if im is not None:
                axes[i, j].imshow(im)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=12)
        axes[i, 0].text(0, -0.05, row[0], transform=axes[i, 0].transAxes, fontsize=11)
    fig.tight_layout()
    fig.savefig(figd / f"layout_first{tag}.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
