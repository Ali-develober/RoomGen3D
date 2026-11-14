"""Milestone 3 evaluation: panoramic depth, method A (multi-view Depth Anything + global alignment)
vs. method B (DAP; B_dap_wrap = DAP run twice, on the panorama and on it rolled by half a width, cross-faded so the seam disappears), on the generated panoramas of the configuration selected in milestone 2
(first seed: 30 panoramas, 5 rooms x 6 styles).

Per panorama and method:
  overlap disagreement    (A only) mean relative difference of overlapping views, before / after
                          the scale-only and the scale+shift alignment
  seam ratio of depth     |difference across the wrap| / |difference between neighbouring columns|
  wall planarity          wall pixels (SegFormer on the 14 views, projected back) -> 3D -> sequential
                          RANSAC vertical planes (up to 6, inlier band 2% of the median distance);
                          RMS point-to-nearest-plane distance (capped at 10%) as % of the median distance,
                          and the share of wall points on a plane
  floor flatness          RANSAC plane on the floor pixels; RMS distance of the floor points (% of the
                          camera height) and the share of inliers
  wall angles             |angle - 90| between plane normals of wall pairs more than 45 deg apart
  room height             camera-to-floor + camera-to-ceiling plane distance; plausible if 2.2-3.5 m
  time, peak VRAM
Wrap-aware post-processing (gated denoising of the panorama + depth refinement) is applied to both
methods and the planarity metrics are reported before and after it.

Ground truth: no real RGB-D panorama set could be obtained within the plan's rules (Stanford2D3D and
Matterport3D need a signed licence request; see README), so only geometry checks are reported.

Outputs: outputs/pano_depth/<method>/<id>.npy, results/tables/pano_depth_*.csv,
         results/figures/pano_depth_*.png
Usage: python scripts/eval_pano_depth.py [--limit 3]
"""

import argparse
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
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import pano_depth as pd_  # noqa: E402
from src.textured_mesh import _ransac_plane  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "pano_depth"
PANO = ROOT / "outputs" / "pano_eval"
METHODS = ["A_scale", "A_scale_shift", "B_dap", "B_dap_wrap"]


def load(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def selected_config() -> str:
    t = pd.read_csv(ROOT / "results/tables/pano_configs.csv")
    return t.loc[t["selected"], "config"].iloc[0]


def segment_pano(seg, pano, device) -> np.ndarray:
    """ADE20K class map of a panorama: SegFormer on the 14 perspective views, labels projected back
    (segmenting the equirectangular image directly labels most of the distorted ceiling as wall)."""
    return pd_.segment_pano(*seg, pano, device)


def geometry(dist: np.ndarray, cm: np.ndarray, id2label: dict) -> dict:
    """Wall planarity, floor flatness, wall angles and room height of one distance map."""
    pts = pd_.to_points(dist)
    sel = lambda names: np.isin(cm, [i for i, l in id2label.items() if l.strip() in names])  # noqa: E731
    med = float(np.median(dist))
    rng = np.random.default_rng(0)
    out = {}

    def sample(mask, n=40_000):
        m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        p = pts[m]
        return p if len(p) <= n else p[rng.choice(len(p), n, replace=False)]

    up = np.array([0.0, -1.0, 0.0])
    floor_d = ceil_d = None
    for name, names in (("floor", {"floor", "rug"}), ("ceiling", {"ceiling"})):
        p = sample(sel(names))
        if len(p) < 500:
            continue
        n, d, inl = _ransac_plane(p, 0.02 * med)
        res = np.abs(p @ n + d)
        if name == "floor":
            floor_d = d
            out["floor_rms_pct_cam_h"] = 100 * float(np.sqrt(np.mean(np.minimum(res, 0.1 * med) ** 2))) / max(d, 0.1)
            out["floor_inlier_share"] = len(inl) / len(p)
            out["floor_tilt_deg"] = float(np.degrees(np.arccos(min(1, abs(n @ up)))))
            up = n if n @ up > 0 else -n
        else:
            ceil_d = d
    if floor_d is not None and ceil_d is not None:
        out["room_height_m"] = floor_d + ceil_d
        out["camera_height_m"] = floor_d

    p = sample(sel({"wall"}), 60_000)
    if len(p) >= 1000:
        thr = 0.02 * med
        rest, normals, best = p, [], np.full(len(p), np.inf)
        for _ in range(8):
            if len(rest) < 500 or len(normals) >= 6:
                break
            n, d, inl = _ransac_plane(rest, thr)
            if len(inl) < 0.05 * len(p):                             # too small: stop
                break
            if abs(n @ up) <= 0.3:                                      # vertical: a wall plane
                normals.append(n)
                best = np.minimum(best, np.abs(p @ n + d))
            rest = rest[np.abs(rest @ n + d) > thr]                     # (non-vertical: skipped)
        if normals:
            out["walls_found"] = len(normals)
            out["wall_rms_pct"] = 100 * float(np.sqrt(np.mean(np.minimum(best, 0.1 * med) ** 2))) / med
            out["wall_inlier_share"] = float(np.mean(best < thr))
            dev = []
            for i in range(len(normals)):
                for j in range(i + 1, len(normals)):
                    a = np.degrees(np.arccos(np.clip(abs(normals[i] @ normals[j]), 0, 1)))
                    if a > 45:
                        dev.append(abs(90 - a))
            if dev:
                out["wall_angle_err_deg"] = float(np.mean(dev))
    out["depth_seam_ratio"] = eq.seam_ratio(dist)
    out["median_dist_m"] = med
    return out


def run_method_a(ids, cfg, device, limit_tag) -> pd.DataFrame:
    rows = []
    model, proc = dp.load_depth_model("depth_anything_metric", device)
    for pid in tqdm(ids, desc="method A", unit="pano"):
        pano = load(PANO / cfg / f"{pid}.png")
        for shift in (False, True):
            name = "A_scale_shift" if shift else "A_scale"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = pd_.method_a(pano, model, proc, device, shift=shift)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            (OUT / name).mkdir(parents=True, exist_ok=True)
            np.save(OUT / name / f"{pid}.npy", r["distance"].astype(np.float16))
            if not shift:
                (OUT / "A_unaligned").mkdir(parents=True, exist_ok=True)
                np.save(OUT / "A_unaligned" / f"{pid}.npy", r["distance_unaligned"].astype(np.float16))
            rows.append({"id": pid, "method": name, "seconds": sec,
                         "vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                         "disagree_before": r["disagree_before"], "disagree_after": r["disagree_after"],
                         "scale_min": float(r["scale"].min()), "scale_max": float(r["scale"].max())})
    del model, proc
    ev.release_memory()
    return pd.DataFrame(rows)


def run_method_b(ids, cfg, device) -> pd.DataFrame:
    rows = []
    model, info = pd_.load_dap(device)
    print("DAP weights:", info)
    for pid in tqdm(ids, desc="method B (DAP)", unit="pano"):
        pano = load(PANO / cfg / f"{pid}.png")
        for name, fn in (("B_dap", pd_.dap_depth), ("B_dap_wrap", pd_.dap_depth_wrap)):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = fn(model, pano, device)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            (OUT / name).mkdir(parents=True, exist_ok=True)
            np.save(OUT / name / f"{pid}.npy", d.astype(np.float16))
            rows.append({"id": pid, "method": name, "seconds": sec,
                         "vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
    del model
    ev.release_memory()
    return pd.DataFrame(rows)


def fig_examples(ids, cfg, path, n=3):
    pick = ids[:: max(1, len(ids) // n)][:n]
    cols = ["image", "A_unaligned", "A_scale", "B_dap"]
    fig, axes = plt.subplots(len(pick), 4, figsize=(20, 2.9 * len(pick)), dpi=100)
    axes = np.atleast_2d(axes)
    for i, pid in enumerate(pick):
        pano = load(PANO / cfg / f"{pid}.png")
        ds = [np.load(OUT / c / f"{pid}.npy").astype(np.float32) for c in cols[1:]]
        hi = np.percentile(np.concatenate([d.ravel() for d in ds]), 98)
        axes[i, 0].imshow(pano)
        for j, d in enumerate(ds):
            axes[i, j + 1].imshow(d, cmap="gray_r", vmin=0, vmax=hi)
        for j, t in enumerate(["Panorama", "A: views blended, no alignment", "A: aligned (scale)", "B: DAP"]):
            axes[i, j].set_title(t if i == 0 else "", fontsize=10)
            axes[i, j].axis("off")
    fig.suptitle("Equirectangular distance maps (white = near, black = far; same scale per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    tag = "" if args.limit is None else "_smoke"
    cfg = selected_config()
    prompts = pd.read_csv(ROOT / "results/tables/pano_per_image.csv", dtype={"id": str})
    prompts = prompts[prompts["config"] == cfg].drop_duplicates("id")
    ids = sorted(prompts["id"])[::2]                         # first seed of every prompt (ids alternate seeds)
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]
    print(f"config {cfg}, {len(ids)} panoramas")

    ta = run_method_a(ids, cfg, device, tag)
    tb = run_method_b(ids, cfg, device)
    timing = pd.concat([ta, tb], ignore_index=True)

    seg = an.load_segmenter(device)
    id2label = seg[0].config.id2label
    rows = []
    for pid in tqdm(ids, desc="geometry", unit="pano"):
        pano = load(PANO / cfg / f"{pid}.png")
        cm = segment_pano(seg, pano, device)
        pre = pd_.gated_preprocess_pano(pano)
        for m in ["A_unaligned"] + METHODS:
            d = np.load(OUT / m / f"{pid}.npy").astype(np.float32)
            rows.append({"id": pid, "method": m, "post": "none", **geometry(d, cm, id2label)})
            if m != "A_unaligned":
                ref = pd_.refine_pano_distance(d, pre)
                rows.append({"id": pid, "method": m, "post": "gated+refine", **geometry(ref, cm, id2label)})
    del seg
    ev.release_memory()
    geo = pd.DataFrame(rows)
    cols = ["wall_rms_pct", "wall_inlier_share", "walls_found", "wall_angle_err_deg", "floor_rms_pct_cam_h",
            "floor_inlier_share", "floor_tilt_deg", "room_height_m", "camera_height_m", "depth_seam_ratio", "median_dist_m"]
    summ = geo.groupby(["method", "post"], sort=False)[cols].median().reset_index()
    plaus = geo.groupby(["method", "post"], sort=False)["room_height_m"].apply(
        lambda h: float(((h >= 2.2) & (h <= 3.5)).sum() / max(h.notna().sum(), 1))).rename("height_plausible_share")
    summ = summ.merge(plaus.reset_index(), on=["method", "post"])
    tsum = timing.groupby("method", sort=False)[[c for c in ["seconds", "vram_gb", "disagree_before", "disagree_after"]
                                                 if c in timing]].mean().reset_index()
    summ = summ.merge(tsum, on="method", how="left")
    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    geo.to_csv(tables / f"pano_depth_per_image{tag}.csv", index=False, float_format="%.4f")
    timing.to_csv(tables / f"pano_depth_timing{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(tables / f"pano_depth{tag}.csv", index=False, float_format="%.4f")
    fig_examples(ids, cfg, figs / f"pano_depth_examples{tag}.png")
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())


if __name__ == "__main__":
    main()
