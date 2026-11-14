"""PLAN_V4 final: one summary figure per demo room and one before / after table.

Per demo room (outputs/room360/<room>): the final v4 room (pipeline_v4.build with the milestone-2 4096 panorama,
up to 4 TripoSR objects from the milestone-4 cache, relief, open sides, patch fill) is rebuilt and saved to
outputs/room360/<room>/v4; the v3 room is rebuilt from the same inputs.
Figure results/figures/v4_summary_<room>.png:
    row 1  hero shot H | wall shots A-D
    row 2  floor plan with view cones | v3 dollhouse | v4 dollhouse | v3 inside render | v4 inside render
           (inside renders: exact ray casting from the panorama centre at the hero pose)
Overview: results/figures/v4_summary_overview.png. Table results/tables/v4_summary.csv: every before / after
metric of PLAN_V4, read from the milestone tables (held-out / demo as stated in the 'data' column).
Usage: python scripts/v4_summary.py
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
from src import design_shots as ds  # noqa: E402
from src import objects3d as o3  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import pipeline_v4 as p4v  # noqa: E402
from src import room_v4 as r4  # noqa: E402
import eval_objects as eo  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEVICE = torch.device("cuda")


def figures():
    id2label = erb.id2label_ade()
    model = o3.load_triposr(DEVICE)
    over = []
    for prompt, seed in tqdm(pl.DEMO_PROMPTS, desc="demo rooms"):
        slug = pl.slug(prompt, seed)
        d = ROOT / "outputs/room360" / slug
        dist = np.load(d / "distance.npy")
        ana = {"class_map": cv2.imread(str(d / "v3/class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
               "inst_map": cv2.imread(str(d / "v3/inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((d / "v3/inst_v3.json").read_text())}
        tex, hires = mb.rgb(d / "texture.png"), mb.rgb(d / "v4/pano_4096_detail.png")
        r4_ = p4v.build(dist, tex, ana, id2label, hires=hires, triposr=model, device=DEVICE, max_objects=4,
                        cache_dir=ROOT / "outputs/objects_eval" / f"demo_{slug}")
        p4v.save(r4_, d / "v4", title=prompt, pano=hires)
        r3_ = p3v.build(dist, tex, ana, id2label)
        b3, b4 = r3_["box"], r4_["box"]
        objs = r4_["objects"]["objects"] if r4_.get("objects") else []
        shots = r4_["shots"]["shots"]
        hero = shots[0]
        d3 = eo.dollhouse_render(r3_["dollhouse"], [], b3)
        d4 = eo.dollhouse_render(r4_["dollhouse"], objs, b4)
        i3 = r4.render(r3_["mesh"], b3, hero["yaw"], hero["fov"])[0]
        i4 = r4.render(r4_["mesh"], b4, hero["yaw"], hero["fov"])[0]
        fig = plt.figure(figsize=(25, 9.6), dpi=70)
        gs = fig.add_gridspec(2, 5)
        walls = [s for s in shots if s["kind"] == "wall"]
        for j, s in enumerate([hero] + walls[:4]):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(s["image"])
            ax.set_title(ds.caption(s), fontsize=12)
            ax.axis("off")
        n_obj = len(objs)
        panels = [(r4_["plan_image"] if "plan_image" in r4_ else None, "v4 floor plan with view cones"),
                  (d3, "v3 dollhouse (blocks)"),
                  (d4, f"v4 dollhouse ({n_obj} generated object(s), blocks, relief)"),
                  (i3, "v3 inside render (hero pose)"), (i4, "v4 inside render (hero pose)")]
        for j, (im, t) in enumerate(panels):
            ax = fig.add_subplot(gs[1, j])
            if im is not None:
                ax.imshow(im)
            ax.set_title(t, fontsize=12)
            ax.axis("off")
        fig.suptitle(f"{prompt} - PLAN_V4 summary", fontsize=15)
        fig.tight_layout()
        fig.savefig(FIG / f"v4_summary_{slug}.png", facecolor="white")
        plt.close(fig)
        over.append((prompt, hero["image"], r4_["plan_image"], d4, i4))
    fig, axes = plt.subplots(len(over), 4, figsize=(22, 4.3 * len(over)), dpi=60)
    for i, (t, h, p, d4, i4) in enumerate(over):
        for j, (im, tt) in enumerate(((h, "hero shot"), (p, "plan + view cones"), (d4, "v4 dollhouse"), (i4, "v4 inside"))):
            axes[i, j].imshow(im)
            axes[i, j].set_title(tt + (f" - {t}" if j == 0 else ""), fontsize=11)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "v4_summary_overview.png", facecolor="white")
    plt.close(fig)


def table() -> pd.DataFrame:
    rows = []
    add = lambda m, metric, data, v3, v4, better: rows.append(  # noqa: E731
        {"milestone": m, "metric": metric, "data": data, "v3": v3, "v4": v4, "v4_better": better})
    s = pd.read_csv(TAB / "shots_selection.csv").set_index(["split", "method"])
    for c, name in (("gt_inst", "layout furniture >= 50% in the main 2D view"), ("gt_anchor", "main piece in the main 2D view"),
                    ("pred_inst", "predicted instances in the main 2D view")):
        a, b = s.loc[("test", "baseline_first_wall"), c], s.loc[("test", "hero_tuned"), c]
        add("1 design shots", name + " (v3: first wall)", "held-out bench", a, b, b > a)
    p = pd.read_csv(TAB / "pano_refine_report.csv").set_index("config")
    sel = json.loads((TAB / "pano_refine_decision.json").read_text())["selected"]
    for c, name, hi in (("sharpness", "shot sharpness (Laplacian var.)", True), ("brisque", "shot BRISQUE", False),
                        ("kid_x1e3", "shot KID x1e3 vs MMIS", False), ("clip", "shot CLIP", True)):
        a, b = p.loc["v3_texture", c], p.loc[sel, c]
        add("2 sharper panorama", name + f" (v4: {sel})", "held-out bench + demo", a, b, (b > a) if hi else (b < a))
    a, b = p.loc["v3_texture", "sharpness"], p.loc["esrgan", "sharpness"]
    add("2 sharper panorama", "shot sharpness (v4 app default: ESRGAN views)", "held-out bench + demo", a, b, b > a)
    r = pd.read_csv(TAB / "room_v4_report.csv").set_index("variant")
    g = json.loads((TAB / "room_v4_selection.json").read_text())["geometry"]
    for c, name, hi in (("psnr", "inside render vs shot PSNR", True), ("ssim", "inside render vs shot SSIM", True),
                        ("lpips", "inside render vs shot LPIPS", False), ("jaggedness", "furniture silhouette jaggedness", False),
                        ("glb_mb", "inside GLB size (MB)", False)):
        a, b = r.loc["v3", c], r.loc[g, c]
        add("3 sharp 3D", name, "held-out bench + demo (offline)", a, b, (b > a) if hi else (b < a))
    v = pd.read_csv(TAB / "viewer_v4_summary.csv")
    v3_ = v[(v.version == "v3")].iloc[0]
    v4_ = v[(v.version == "v4") & (v.tone_mapping == "neutral")].iloc[0]
    for c, name, hi in (("psnr", "browser render vs shot PSNR", True), ("ssim", "browser SSIM", True),
                        ("lpips", "browser LPIPS", False), ("delta_e", "browser colour error dE2000", False)):
        add("3 sharp 3D", name, "demo rooms, headed Chromium", v3_[c], v4_[c], (v4_[c] > v3_[c]) if hi else (v4_[c] < v3_[c]))
    o = pd.read_csv(TAB / "objects_summary.csv").set_index("set").loc["test"]
    add("4 furniture", "silhouette IoU from the centre", "held-out bench", o["sil_iou_v3"], o["sil_iou_v4"],
        o["sil_iou_v4"] > o["sil_iou_v3"] + 0.005)
    for c, name, hi in (("psnr", "full-room render vs panorama PSNR", True), ("ssim", "full-room render SSIM", True),
                        ("lpips", "full-room render LPIPS", False)):
        a, b = o[f"{c}_v3"], o[f"{c}_v4"]
        add("4 furniture", name, "held-out bench", a, b, (b > a) if hi else (b < a))
    add("4 furniture", "furniture found per room (IoU >= 0.25), with relief", "held-out bench", o["tp25"], o["tp25_with_relief"],
        o["tp25_with_relief"] > o["tp25"])
    add("4 furniture", "share of pieces as generated objects", "held-out bench", 0.0,
        o["total_n_object"] / max(o["total_n_solids"], 1), None)
    f = pd.read_csv(TAB / "fill_summary.csv").set_index(["set", "method"])
    fm = json.loads((TAB / "fill_selection.json").read_text())["method"]
    for c, name, hi in (("energy_err", "hidden-area fill |texture energy ratio - 1|", False), ("darkness", "hidden-area darkness", False),
                        ("brisque", "hidden-area BRISQUE", False), ("ssim", "synthetic-hole SSIM", True)):
        a, b = f.loc[("test", "ns"), c], f.loc[("test", fm), c]
        add("5 weaknesses", name + f" (v4: {fm})", "held-out bench", a, b, (b > a) if hi else (b < a))
    dd = pd.read_csv(TAB / "doors_v4.csv").set_index(["set", "config"])
    dsel = json.loads((TAB / "doors_v4_selection.json").read_text())["config"]
    add("5 weaknesses", "door / window recall (refined views)", "held-out bench", dd.loc[("test", "v3"), "recall"],
        dd.loc[("test", dsel), "recall"], dd.loc[("test", dsel), "recall"] > dd.loc[("test", "v3"), "recall"])
    od = pd.read_csv(TAB / "open_sides_demo.csv")
    add("5 weaknesses", "demo rooms with open sides marked", "demo rooms", 0, int((od["open_sides"] > 0).sum()), None)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "v4_summary.csv", index=False, float_format="%.4f")
    return t


def main():
    figures()
    t = table()
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
