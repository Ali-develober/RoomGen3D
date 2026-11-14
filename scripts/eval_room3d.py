"""Milestone 4 evaluation: full-room (360 degree) mesh vs. single-view mesh on the six demo prompts.

Both meshes are gravity-aligned (floor plane -> up = -Y) with their camera at the origin and
rendered by exact ray casting (src/room_mesh.Raycaster). "Empty" = share of rays that hit nothing.

  inside      camera at the centre and moved 0.5 m and 1.0 m horizontally (4 directions); 8 views
              (yaw 0, 45, ..., 315), pitch 0, 90 deg FOV, 128 x 128 rays each. Reported over all
              8 yaws and over the 3 "front" yaws (-45, 0, +45: the single view's own direction)
  dollhouse   8 views from outside and above (azimuth 0, 45, ..., 315; elevation 55 deg; 50 deg FOV)
              looking at the room centre; empty = share of the rays that pass through the room's
              bounding box but hit no surface. 360 mode: dollhouse mesh (ceiling removed);
              single view: its mesh with the same ceiling cut

Single-view meshes are rebuilt from outputs/pipeline/<room>/ (image, refined metric depth) with the
same code as the Quick mode; 360 meshes from outputs/room360/<room>/ (scripts/run_room360.py).
Outputs: results/tables/room3d_empty.csv, room3d_empty_per_view.csv;
         results/figures/room3d_views.png, room3d_dollhouse.png
Usage: python scripts/eval_room3d.py
"""

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
from src import analysis as an  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402
from src import room_mesh as rm  # noqa: E402
from src import textured_mesh as tm  # noqa: E402

SIZE = 128
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
FRONT = {315, 0, 45}
OFFSETS = [0.0, 0.5, 1.0]
DIRS = [0, 90, 180, 270]


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def single_view_mesh(d: Path, seg, device) -> dict:
    """Quick-mode mesh of a demo room, gravity-aligned like the 360 mesh."""
    img = rgb(d / "image.png")
    depth = np.load(d / "depth_m.npy")
    ana = pl.analyse(img, *seg, device)
    mesh = tm.build_textured_mesh(depth, img, ana["edges"], ana["masks"])
    h, w = depth.shape
    k = r3.intrinsics(w, h)
    pts = fp.backproject(depth, k)
    sel = lambda names: np.isin(ana["class_map"], fp.class_ids(ana["id2label"], names))  # noqa: E731
    fr = fp.gravity_frame(pts, depth > 0.05, sel(fp.FLOOR_CLASSES), sel(fp.CEILING_CLASSES))
    R = np.stack([fr["right"], -fr["up"], fr["fwd"]])
    mesh = dict(mesh)
    mesh["vertices"] = (mesh["vertices"] @ R.T).astype(np.float32)
    mesh["heights"] = (fr["d"] - mesh["vertices"][:, 1]).astype(np.float32)
    mesh["box"] = None
    return mesh


def aabb(v: np.ndarray):
    return np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)


def hits_box(o, dirs, lo, hi):
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    tmin, tmax = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
    return tmax >= np.maximum(tmin, 0)


def inside_metrics(mesh, room, mode):
    rc = rm.Raycaster(mesh)
    rows = []
    for r in OFFSETS:
        for th in (DIRS if r > 0 else [0]):
            o = np.array([r * np.sin(np.radians(th)), 0.0, r * np.cos(np.radians(th))])
            for yaw in YAWS:
                hit, _, _ = rc.cast(o, rm.view_dirs(yaw, 0, 90, SIZE))
                rows.append({"room": room, "mode": mode, "view": "inside", "offset_m": r, "dir": th, "yaw": yaw,
                             "front": yaw in FRONT, "empty": 1 - hit.mean()})
    return rows


def dollhouse_metrics(mesh, room, mode):
    dh = rm.dollhouse(mesh)
    rc = rm.Raycaster(dh)
    lo, hi = aabb(dh["vertices"])
    c = (lo + hi) / 2
    r = 1.6 * np.linalg.norm((hi - lo)[[0, 2]])
    rows, imgs = [], []
    for az in YAWS:
        a, el = np.radians(az), np.radians(55)
        eye = c + np.array([r * np.sin(a) * np.cos(el), -r * np.sin(el), r * np.cos(a) * np.cos(el)])
        dirs = rm.look_at_dirs(eye, c, 50, SIZE)
        inb = hits_box(eye, dirs, lo, hi)
        hit, _, col = rc.cast(eye, dirs, color=az in (0, 180))
        rows.append({"room": room, "mode": mode, "view": "dollhouse", "offset_m": np.nan, "dir": np.nan, "yaw": az,
                     "front": az in FRONT, "empty": 1 - hit[inb].mean() if inb.any() else np.nan})
        if col is not None:
            imgs.append(col.reshape(SIZE, SIZE, 3))
    return rows, imgs


def render(mesh, origin, yaw, size=256, pitch=0.0):
    _, _, col = rm.Raycaster(mesh).cast(np.asarray(origin, float), rm.view_dirs(yaw, pitch, 90, size), color=True)
    return col.reshape(size, size, 3)


def main():
    device = torch.device("cuda")
    seg = an.load_segmenter(device)
    from transformers import AutoConfig
    id2label = AutoConfig.from_pretrained(an.SEG_MODEL).id2label
    rows, figs, dolls = [], [], []
    for prompt, seed in tqdm(pl.DEMO_PROMPTS, desc="rooms"):
        s = pl.slug(prompt, seed)
        single = single_view_mesh(ROOT / "outputs/pipeline" / s, seg, device)
        d = ROOT / "outputs/room360" / s
        res = p3.build(np.load(d / "distance.npy"), rgb(d / "texture.png"),
                       cv2.imread(str(d / "class_map.png"), cv2.IMREAD_UNCHANGED).astype(np.int32), id2label)
        full = res["mesh"]
        for mesh, mode in ((single, "single_view"), (full, "room360")):
            rows += inside_metrics(mesh, prompt, mode)
            r_, im = dollhouse_metrics(mesh, prompt, mode)
            rows += r_
            dolls.append((prompt, mode, im))
        figs.append((prompt, [render(single, (0, 0, 0), 0), render(single, (0, 0, 0.5), 180),
                              render(full, (0, 0, 0), 0), render(full, (0, 0, 0), 180),
                              render(full, (0.0, 0, 1.0), 90)]))
    del seg
    ev.release_memory()
    per = pd.DataFrame(rows)
    out = []
    for (mode, view), g in per.groupby(["mode", "view"], sort=False):
        if view == "inside":
            for r in OFFSETS:
                gg = g[g["offset_m"] == r]
                out.append({"mode": mode, "view": f"inside, {r:.1f} m from centre", "empty_all_yaws": gg["empty"].mean(),
                            "empty_front_yaws": gg[gg["front"]]["empty"].mean()})
        else:
            out.append({"mode": mode, "view": "dollhouse (outside, above)", "empty_all_yaws": g["empty"].mean(),
                        "empty_front_yaws": g[g["front"]]["empty"].mean()})
    summ = pd.DataFrame(out)
    tables, fdir = ROOT / "results/tables", ROOT / "results/figures"
    per.to_csv(tables / "room3d_empty_per_view.csv", index=False, float_format="%.4f")
    summ.to_csv(tables / "room3d_empty.csv", index=False, float_format="%.4f")
    print(summ.round(4).to_string(index=False))

    titles = ["Single view: camera", "Single view: 0.5 m back, turned 180°", "360°: centre, front",
              "360°: centre, turned 180°", "360°: 1 m forward, looking right"]
    fig, axes = plt.subplots(len(figs), 5, figsize=(17, 3.5 * len(figs)), dpi=90)
    for i, (prompt, ims) in enumerate(figs):
        for j, im in enumerate(ims):
            axes[i, j].imshow(im)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
        axes[i, 0].set_ylabel(prompt, fontsize=9)
    fig.suptitle("Ray-cast renders (white = no surface)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fdir / "room3d_views.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(len(figs), 4, figsize=(12, 3.1 * len(figs)), dpi=90)
    rooms = [p for p, _ in figs]
    for i, prompt in enumerate(rooms):
        ims = {m: im for p, m, im in dolls if p == prompt}
        for j, (m, k) in enumerate((("single_view", 0), ("single_view", 1), ("room360", 0), ("room360", 1))):
            axes[i, j].imshow(ims[m][k])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"{'Single view' if m == 'single_view' else '360° dollhouse'}, azimuth {180 * k}°", fontsize=10)
        axes[i, 0].set_ylabel(prompt, fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "room3d_dollhouse.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
