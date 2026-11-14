"""PLAN_V3 milestone 2: room box regularization, v2 vs v3 on the layout-first test bench.

Same inputs for both: DAP distance of the generated panorama + Real-ESRGAN texture (outputs/bench).
    v2  room_mesh.build_room_mesh with the v2 SegFormer class map (the current 360 pipeline)
    v3  room_box.build_room_v3 with the v3 panoptic class map (milestone 1): box shell + observed furniture
Ground truth: the layout box (camera frame: x in [-Cx, W-Cx], z in [-Cz, D-Cz], floor 1.4 m below the camera).

Metrics per room
    box         area error, side (width / depth) error, corner error (mean distance GT corner -> nearest
                predicted corner, m), sides detected (v2: sides backed by a wall line; v3: density peak)
    walls       rays from the camera through pixels whose GT label is wall (1/16 of them), cast against the
                mesh: |hit - GT distance| median (m); per GT wall, a plane is fitted to the hit points:
                planarity = RMS residual (cm), normal error = angle to the true wall normal (deg)
    outside     share of mesh surface area more than 10 cm outside the GT box, after scaling the mesh so that
                the camera is 1.4 m above its floor (removes DAP's global scale error; "_scaled" metrics)
    flat        share of rendered pixels in flat single-colour patches (7x7 grey std < 1.5); inside views
                (camera at the centre, 8 yaws): ADDED flatness = render minus the same view cut from the panorama
                (plain generated walls are flat anyway); dollhouse views (ceiling removed, 4 azimuths): raw
    empty       share of inside-view rays that hit nothing (holes)
Inpainting of hidden shell texels (Telea / Navier-Stokes / patch-based): synthetic furniture-shaped holes
(bottom-anchored rectangles) in the VISIBLE part of each wall texture; PSNR / SSIM inside the hole, flat
share, time. The method is chosen on the DEV rooms (best mean SSIM), reported on the HELD-OUT rooms.
Outputs: results/tables/box_*.csv, results/figures/box_*.png
Usage: python scripts/eval_room_box.py [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import equirect as eq  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import room_mesh as rm  # noqa: E402
from src.utils import set_seed  # noqa: E402
import make_bench as mb  # noqa: E402

YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
SIZE = 128


def id2label_ade():
    from transformers import AutoConfig
    return {int(k): v for k, v in AutoConfig.from_pretrained(str(ROOT / "checkpoints/mask2former_ade_panoptic")).id2label.items()}


def gt_box(L: dict) -> dict:
    """GT room box in the camera frame (Y down)."""
    C = np.array(L["camera"])
    return {"x": (-C[0], L["W"] - C[0]), "z": (-C[2], L["D"] - C[2]), "floor": C[1], "ceil": C[1] - L["H"],
            "corners": np.array([[-C[0], -C[2]], [L["W"] - C[0], -C[2]], [L["W"] - C[0], L["D"] - C[2]],
                                 [-C[0], L["D"] - C[2]]])}


def box_metrics(corners_pred: np.ndarray, n_det: int, G: dict, L: dict) -> dict:
    """corners_pred: 4 x 2 plan corners in the camera frame (x, z)."""
    e = np.linalg.norm(corners_pred[[1, 3]] - corners_pred[0], axis=1)
    area = float(cv2.contourArea(corners_pred.astype(np.float32)))
    sides_p, sides_t = sorted(e), sorted([L["W"], L["D"]])
    corner = np.mean([np.min(np.linalg.norm(corners_pred - g, axis=1)) for g in G["corners"]])
    return {"area_m2": area, "area_rel_err": area / (L["W"] * L["D"]) - 1,
            "side_rel_err": float(np.mean([abs(a / b - 1) for a, b in zip(sides_p, sides_t)])),
            "corner_err_m": float(corner), "sides_detected": n_det}


def wall_metrics(mesh: dict, R: np.ndarray, gt_dist: np.ndarray, gt_lab: np.ndarray, G: dict) -> dict:
    """Ray casts through GT wall pixels; R maps camera -> mesh frame."""
    H, W = gt_dist.shape
    sub = np.zeros((H, W), bool)
    sub[::4, ::4] = True
    m = sub & (gt_lab == 0)
    dirs_cam = eq.pixel_dirs(W, H)[m]
    hit, t, _ = rm.Raycaster(mesh).cast(np.zeros(3), (dirs_cam @ R.T).astype(np.float32))
    out = {"wall_hit_share": float(hit.mean()),
           "wall_abs_err_m": float(np.median(np.abs(t[hit] - gt_dist[m][hit]))) if hit.any() else np.nan}
    P = dirs_cam[hit] * t[hit, None]                               # hit points, camera frame
    Pg = dirs_cam[hit] * gt_dist[m][hit, None]                     # where they should be
    # which GT wall: the one whose plane the GT point lies on
    sides = [(0, G["x"][0], np.array([1, 0, 0])), (0, G["x"][1], np.array([-1, 0, 0])),
             (2, G["z"][0], np.array([0, 0, 1])), (2, G["z"][1], np.array([0, 0, -1]))]
    rms, ang = [], []
    for ax, pos, nrm in sides:
        s = np.abs(Pg[:, ax] - pos) < 0.02
        if s.sum() < 50:
            continue
        q = P[s]
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        rms.append(float(np.sqrt(np.mean(((q - c) @ n) ** 2))))
        ang.append(float(np.degrees(np.arccos(min(1.0, abs(n @ nrm))))))
    out["planarity_rms_cm"] = 100 * float(np.mean(rms)) if rms else np.nan
    out["wall_normal_err_deg"] = float(np.mean(ang)) if ang else np.nan
    return out


def outside_share(mesh: dict, R: np.ndarray, G: dict, margin: float = 0.10, scale: float = 1.0) -> float:
    v = scale * mesh["vertices"].astype(np.float64) @ R             # mesh frame -> camera frame
    f = mesh["faces"]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ctr = (a + b + c) / 3
    out = ((ctr[:, 0] < G["x"][0] - margin) | (ctr[:, 0] > G["x"][1] + margin) |
           (ctr[:, 2] < G["z"][0] - margin) | (ctr[:, 2] > G["z"][1] + margin) |
           (ctr[:, 1] > G["floor"] + margin) | (ctr[:, 1] < G["ceil"] - margin))
    return float(area[out].sum() / max(area.sum(), 1e-9))


def flat_share(img: np.ndarray, hit: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
    h = cv2.erode(hit.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((sd[h] < 1.5).mean()) if h.any() else np.nan


def views(mesh: dict, dollhouse: dict, texture: np.ndarray, R: np.ndarray, size: int = SIZE, keep: bool = False):
    """Inside views at the centre and dollhouse views; returns metrics and (optionally) images.
    flat_added_inside = flat share of the render minus that of the same view cut from the panorama."""
    rc = rm.Raycaster(mesh)
    flat, empty, imgs = [], [], {}
    for yaw in YAWS:
        dA = rm.view_dirs(yaw, 0, 90, size)
        hit, _, col = rc.cast(np.zeros(3), dA, color=True)
        img = col.reshape(size, size, 3)
        ref = rb._sample_pano(texture, (dA @ R).reshape(size, size, 3))
        flat.append(flat_share(img, hit.reshape(size, size)) - flat_share(ref, np.ones((size, size), bool)))
        empty.append(1 - hit.mean())
        if keep and yaw == 0:
            imgs["inside"] = img
    rd = rm.Raycaster(dollhouse)
    v = dollhouse["vertices"]
    lo, hi = np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)
    c = (lo + hi) / 2
    r = 1.6 * np.linalg.norm((hi - lo)[[0, 2]])
    dflat = []
    for az in (45, 135, 225, 315):
        a, el = np.radians(az), np.radians(55)
        eye = c + np.array([r * np.sin(a) * np.cos(el), -r * np.sin(el), r * np.cos(a) * np.cos(el)])
        hit, _, col = rd.cast(eye, rm.look_at_dirs(eye, c, 50, size * 2), color=True)
        img = col.reshape(size * 2, size * 2, 3)
        dflat.append(flat_share(img, hit.reshape(size * 2, size * 2)))
        if keep and az == 225:
            imgs["dollhouse"] = img
    return {"flat_added_inside": float(np.nanmean(flat)), "empty_inside": float(np.mean(empty)),
            "flat_dollhouse": float(np.nanmean(dflat))}, imgs


def v3_dollhouse(mesh: dict) -> dict:
    keep = mesh["face_kind"] != "ceiling"
    out = dict(mesh)
    out["faces"] = mesh["faces"][keep]
    return out


def inpaint_trials(box: dict, texture: np.ndarray, dist: np.ndarray, rng) -> list[dict]:
    """Synthetic furniture-shaped holes in the visible part of each wall texture."""
    rows = []
    for pl in rb.shell_planes(box)[:4]:
        img, hidden, beyond = rb.plane_texture(pl, box, texture, dist, "telea")
        vis = ~hidden & ~beyond
        h, w = vis.shape
        for _ in range(2):
            hw, hh = int(w * rng.uniform(0.15, 0.35)), int(h * rng.uniform(0.25, 0.45))
            x0 = int(rng.integers(0, max(w - hw, 1)))
            y0 = h - hh - int(rng.integers(0, max(h // 10, 1)))
            hole = np.zeros_like(vis)
            hole[y0:y0 + hh, x0:x0 + hw] = True
            if vis[hole].mean() < 0.98 or vis.mean() < 0.3:
                continue
            for meth in rb.INPAINT_METHODS:
                t0 = time.perf_counter()
                out = rb.inpaint(img, hole, meth)
                dt = time.perf_counter() - t0
                ys, xs = np.nonzero(hole)
                a, b = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
                mu = cv2.blur(g, (7, 7))
                sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - mu * mu, 0))
                rows.append({"plane": pl["kind"], "method": meth, "hole_px": int(hole.sum()), "time_s": dt,
                             "psnr": peak_signal_noise_ratio(b, a, data_range=255),
                             "ssim": structural_similarity(b, a, channel_axis=2, data_range=255),
                             "flat_share": float((sd[hole] < 1.5).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    tag = "" if args.limit is None else "_smoke"
    df = mb.rooms()
    if args.limit:
        df = df.groupby("split").head(args.limit)
    id2label = id2label_ade()
    rng = np.random.default_rng(0)

    # --- inpainting: choose on dev, report on test --------------------------------------------------- #
    inp = []
    for r in tqdm(list(df.itertuples()), desc="inpainting trials", unit="room"):
        d = mb.OUT / r.id
        dist, tex = np.load(d / "dist.npy"), mb.rgb(d / "texture.png")
        cm3 = cv2.imread(str(d / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        box = rb.fit_box(dist, cm3, id2label)
        inp += [{"id": r.id, "split": r.split, **x} for x in inpaint_trials(box, tex, dist, rng)]
    inp = pd.DataFrame(inp)
    inp_s = inp.groupby(["split", "method"])[["psnr", "ssim", "flat_share", "time_s"]].mean().reset_index()
    inp_s["n"] = inp.groupby(["split", "method"]).size().values
    dev_s = inp_s[inp_s["split"] == "dev"]
    method = str(dev_s.sort_values("ssim", ascending=False).iloc[0]["method"])
    print(inp_s.round(3).to_string(index=False))
    print("inpainting method selected on dev:", method)

    rows, figs = [], []
    for r in tqdm(list(df.itertuples()), desc="meshes", unit="room"):
        d = mb.OUT / r.id
        g = mb.load_gt(d)
        L, G = g["layout"], gt_box(g["layout"])
        dist, tex = np.load(d / "dist.npy"), mb.rgb(d / "texture.png")
        cm2 = cv2.imread(str(d / "seg_v2.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cm3 = cv2.imread(str(d / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        t0 = time.perf_counter()
        m2 = rm.build_room_mesh(dist, tex, cm2, id2label)
        t2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        m3 = rb.build_room_v3(dist, tex, cm3, id2label, method=method)
        t3 = time.perf_counter() - t0
        room2 = m2["plan"]["room"]
        c2 = (m2["box"]["corners"] if m2["box"] else room2["corners"])          # plan coords of frame A (x, z)
        R2, R3 = m2["R"], m3["box"]["R"]
        to_cam = lambda c, R: (np.c_[c[:, 0], np.zeros(len(c)), c[:, 1]] @ R)[:, [0, 2]]  # noqa: E731
        res = {}
        for name, mesh, R, corners, ndet, dh, t in (
                ("v2", m2, R2, to_cam(c2, R2), room2["n_observed_sides"] if room2 else 0, rm.dollhouse(m2), t2),
                ("v3", m3, R3, to_cam(rb.box_corners_A(m3["box"]), R3), m3["box"]["n_detected"], v3_dollhouse(m3), t3)):
            sc = 1.4 / mesh["camera_height"]                         # remove DAP's global scale error
            met = {**box_metrics(corners, ndet, G, L), **wall_metrics(mesh, R, g["dist"], g["labels"], G),
                   **{f"{k}_scaled": v for k, v in box_metrics(corners * sc, ndet, G, L).items()
                      if k in ("area_rel_err", "side_rel_err", "corner_err_m")},
                   "camera_height_pred": mesh["camera_height"], "outside_share_scaled": outside_share(mesh, R, G, scale=sc),
                   "build_s": t}
            vm, imgs = views(mesh, dh, tex, R, keep=(r.split == "test" and r.layout_seed == 5))
            met.update(vm)
            res[name] = imgs
            rows.append({"id": r.id, "room": r.room, "split": r.split, "method": name, **met})
        if res["v3"]:
            figs.append((r.prompt, mb.rgb(d / "pano.png"), res["v2"]["dollhouse"], res["v3"]["dollhouse"],
                         res["v2"]["inside"], res["v3"]["inside"]))
    pr = pd.DataFrame(rows)
    cols = [c for c in pr.columns if c not in ("id", "room", "split", "method")]
    summ = pr.groupby(["split", "method"])[cols].mean().reset_index()
    for c in ("area_rel_err", "area_rel_err_scaled"):
        summ[c.replace("rel_err", "abs_rel_err")] = pr.groupby(["split", "method"])[c].agg(lambda s: s.abs().mean()).values

    T, F = ROOT / "results/tables", ROOT / "results/figures"
    inp.to_csv(T / f"box_inpaint_trials{tag}.csv", index=False, float_format="%.4f")
    inp_s.to_csv(T / f"box_inpaint{tag}.csv", index=False, float_format="%.4f")
    pr.to_csv(T / f"box_per_room{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(T / f"box_summary{tag}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())

    if figs:
        n = len(figs)
        fig, axes = plt.subplots(n, 5, figsize=(26, 3.6 * n), dpi=80, gridspec_kw={"width_ratios": [2, 1, 1, 1, 1]})
        axes = np.atleast_2d(axes)
        titles = ["Panorama (held-out)", "v2 dollhouse", "v3 dollhouse (box shell)", "v2 inside", "v3 inside"]
        for i, (title, *ims) in enumerate(figs):
            for j, im in enumerate(ims):
                axes[i, j].imshow(im)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(titles[j], fontsize=13)
            axes[i, 0].text(0, -0.06, title, transform=axes[i, 0].transAxes, fontsize=11)
        fig.tight_layout()
        fig.savefig(F / f"box_v2_v3{tag}.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
