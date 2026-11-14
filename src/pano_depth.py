"""Panoramic (equirectangular) metric depth.

Method A (ours): equirectangular -> overlapping perspective views -> Depth Anything V2 Metric Indoor
per view -> z-depth converted to radial distance -> global least-squares alignment of the per-view
scale (and shift) over the overlaps -> feathered blending back to an equirectangular DISTANCE map.

    views        6 cube faces (yaw 0/90/180/270 at pitch 0, up, down) + 8 offset views
                 (yaw 45/135/225/315 at pitch +-35), all 100 deg FOV, 518 px
    per view     z(u, v) from the depth model; radial distance r = z * |ray(u, v)| with ray z = 1
    to panorama  for every panorama pixel and view: project its direction into the view; if it
                 falls inside, sample r there (bilinear) -> D_i (equirect) + weight W_i
    weights      feathering: w = (cos of the angle between the pixel direction and the view axis)^4,
                 times a ramp that goes to 0 within 6% of the view border
    alignment    unknowns: per-view scale s_i (and shift t_i). Every panorama pixel seen by views i
                 and j gives one equation  s_i D_i + t_i = s_j D_j + t_j  (scale-only: in log space,
                 log s_i - log s_j = log D_j - log D_i). Gauge: mean log s_i = 0 (and mean t_i = 0),
                 so the overall metric scale of the depth model is kept. Robust: 3 rounds of
                 iteratively re-weighted least squares (Cauchy weights on the residuals).
    blending     D = sum_i W_i s_i D_i / sum_i W_i

Method B: DAP "Depth Any Panoramas" (Insta360), a metric panoramic depth network (load_dap / dap_depth).

Post-processing (wrap-aware): noise-gated denoising of the panorama and depth refinement, with the
panorama padded horizontally by its own opposite border before filtering (cv2.BORDER_WRAP), so the
filters see continuous content across lon = +-180 deg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from . import depth as dp
from . import equirect as eq

ROOT = Path(__file__).resolve().parents[1]
DAP_DIR = ROOT / "checkpoints" / "dap"
VIEW_FOV = 100.0
VIEW_SIZE = 518
VIEWS = ([(y, 0.0) for y in (0, 90, 180, 270)] + [(0, 90.0), (0, -90.0)] +
         [(y, p) for y in (45, 135, 225, 315) for p in (35.0, -35.0)])
FEATHER_POWER = 4
BORDER_RAMP = 0.06
MAX_DIST = 20.0


# --------------------------------------------------------------------------- #
# Method A
# --------------------------------------------------------------------------- #
def view_images(pano: np.ndarray, views=VIEWS, fov: float = VIEW_FOV, size: int = VIEW_SIZE) -> list[np.ndarray]:
    return [eq.e2p(pano, fov, y, p, size) for y, p in views]


def predict_views(model, proc, images, device) -> list[np.ndarray]:
    """Metric z-depth (m) of every perspective view."""
    return [dp.predict_depth(model, proc, im, device) for im in images]


def z_to_radial(z: np.ndarray, fov: float = VIEW_FOV) -> np.ndarray:
    """Planar z-depth of a square pinhole view -> distance along each pixel ray."""
    r = eq.perspective_rays(z.shape[0], fov)
    return z * np.linalg.norm(r, axis=-1)


def project_views(dists: list[np.ndarray], w: int, h: int, views=VIEWS, fov: float = VIEW_FOV):
    """Resample every view's radial distance onto the equirectangular grid.

    Returns D (n, h, w) with NaN where a view does not see the pixel, and feather weights W (n, h, w).
    """
    dirs = eq.pixel_dirs(w, h)
    n = len(views)
    D = np.full((n, h, w), np.nan, np.float32)
    W = np.zeros((n, h, w), np.float32)
    size = dists[0].shape[0]
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    c = (size - 1) / 2
    for i, ((yaw, pitch), dist) in enumerate(zip(views, dists)):
        R = eq.rotation(yaw, pitch)
        dc = dirs @ R                     # world -> camera (R is orthonormal: R^T applied to row vectors)
        z = dc[..., 2]
        ok = z > 1e-3
        u = np.where(ok, dc[..., 0] / np.maximum(z, 1e-3) * f + c, -1)
        v = np.where(ok, dc[..., 1] / np.maximum(z, 1e-3) * f + c, -1)
        inside = ok & (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
        samp = cv2.remap(dist.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        edge = np.minimum.reduce([u, v, size - 1 - u, size - 1 - v]) / (BORDER_RAMP * size)
        wgt = np.clip(z, 0, 1) ** FEATHER_POWER * np.clip(edge, 0, 1)
        D[i][inside] = samp[inside]
        W[i][inside] = wgt[inside]
    return D, W


def _pairs(D: np.ndarray, W: np.ndarray, max_per_pair: int = 4000, seed: int = 0):
    """Overlap samples: list of (i, j, D_i, D_j) arrays for every view pair with enough overlap."""
    rng = np.random.default_rng(seed)
    n = len(D)
    out = []
    valid = np.isfinite(D) & (W > 0.02)
    for i in range(n):
        for j in range(i + 1, n):
            m = valid[i] & valid[j]
            k = np.flatnonzero(m)
            if len(k) < 200:
                continue
            if len(k) > max_per_pair:
                k = rng.choice(k, max_per_pair, replace=False)
            out.append((i, j, D[i].ravel()[k], D[j].ravel()[k]))
    return out


def overlap_disagreement(D: np.ndarray, W: np.ndarray, scale=None, shift=None) -> float:
    """Mean relative disagreement |a - b| / ((a + b) / 2) over all overlap samples (after scale/shift)."""
    n = len(D)
    s = np.ones(n) if scale is None else scale
    t = np.zeros(n) if shift is None else shift
    errs = []
    for i, j, a, b in _pairs(D, W):
        a, b = s[i] * a + t[i], s[j] * b + t[j]
        errs.append(np.abs(a - b) / np.maximum((a + b) / 2, 1e-3))
    return float(np.mean(np.concatenate(errs))) if errs else float("nan")


def align_views(D: np.ndarray, W: np.ndarray, shift: bool = False, irls: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Global least-squares per-view scale (and shift) from the overlaps (see module docstring)."""
    n = len(D)
    pairs = _pairs(D, W)
    if not shift:
        rows, rhs = [], []
        for i, j, a, b in pairs:
            la, lb = np.log(np.maximum(a, 1e-3)), np.log(np.maximum(b, 1e-3))
            rows.append((i, j, len(a)))
            rhs.append(lb - la)                        # log s_i - log s_j = log b - log a
        A = np.zeros((sum(r[2] for r in rows) + 1, n))
        y = np.zeros(A.shape[0])
        k = 0
        for (i, j, m), r in zip(rows, rhs):
            A[k:k + m, i], A[k:k + m, j], y[k:k + m] = 1, -1, r
            k += m
        A[-1, :] = 100.0                                # gauge: mean log s = 0 (heavily weighted)
        wts = np.ones(len(y))
        for _ in range(irls):
            sw = np.sqrt(wts)
            x = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
            res = A @ x - y
            c = 2.385 * max(np.median(np.abs(res[:-1])), 1e-4) / 0.6745
            wts = 1.0 / (1.0 + (res / c) ** 2)
            wts[-1] = 1.0
        return np.exp(x), np.zeros(n)
    # scale + shift, in two steps. A joint solve is ill-posed: every view can shrink its scale and
    # take the same large shift, which makes all views agree on a constant map. So the scales come
    # from the log-space solve above and stay fixed; then only the shifts are solved:
    #     t_i - t_j = s_j b - s_i a      with the gauge  mean t = 0  (hard, by elimination of t_0).
    s, _ = align_views(D, W, shift=False, irls=irls)
    m_tot = sum(len(a) for _, _, a, _ in pairs)
    A = np.zeros((m_tot, n))
    y = np.zeros(m_tot)
    k = 0
    for i, j, a, b in pairs:
        m = len(a)
        A[k:k + m, i], A[k:k + m, j], y[k:k + m] = 1, -1, s[j] * b - s[i] * a
        k += m
    # eliminate t_0 = -(t_1 + ... + t_{n-1}): its column is subtracted from all the others
    A = A[:, 1:] - A[:, :1]
    wts = np.ones(len(y))
    for _ in range(irls):
        sw = np.sqrt(wts)
        x = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        res = A @ x - y
        c = 2.385 * max(np.median(np.abs(res)), 1e-4) / 0.6745
        wts = 1.0 / (1.0 + (res / c) ** 2)
    t = np.r_[-x.sum(), x]
    return s, t


def blend(D: np.ndarray, W: np.ndarray, scale=None, shift=None) -> np.ndarray:
    n = len(D)
    s = np.ones(n) if scale is None else scale
    t = np.zeros(n) if shift is None else shift
    Dz = np.where(np.isfinite(D), D * s[:, None, None] + t[:, None, None], 0.0)
    num = (W * Dz).sum(0)
    den = W.sum(0)
    out = num / np.maximum(den, 1e-6)
    return np.clip(out, 0.05, MAX_DIST).astype(np.float32)


def method_a(pano: np.ndarray, model, proc, device, shift: bool = False, out_size: tuple | None = None) -> dict:
    """Full method A. Returns distance map, raw (unaligned) blend, per-view scales/shifts and the
    overlap disagreement before / after alignment."""
    h, w = out_size or pano.shape[:2]
    ims = view_images(pano)
    z = predict_views(model, proc, ims, device)
    r = [z_to_radial(zz) for zz in z]
    D, W = project_views(r, w, h)
    s, t = align_views(D, W, shift=shift)
    return {"distance": blend(D, W, s, t), "distance_unaligned": blend(D, W), "scale": s, "shift": t,
            "disagree_before": overlap_disagreement(D, W), "disagree_after": overlap_disagreement(D, W, s, t),
            "views": ims, "D": D, "W": W}


def project_labels(labels: list[np.ndarray], w: int, h: int, views=VIEWS, fov: float = VIEW_FOV) -> np.ndarray:
    """Per-view label maps (e.g. ADE20K class ids) -> one equirectangular label map. Each panorama
    pixel takes the label of the view that sees it most centrally (highest feather weight)."""
    dirs = eq.pixel_dirs(w, h)
    size = labels[0].shape[0]
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    c = (size - 1) / 2
    best_w = np.full((h, w), -1.0, np.float32)
    out = np.zeros((h, w), np.int32)
    for (yaw, pitch), lab in zip(views, labels):
        dc = dirs @ eq.rotation(yaw, pitch)
        z = dc[..., 2]
        ok = z > 1e-3
        u = np.where(ok, dc[..., 0] / np.maximum(z, 1e-3) * f + c, -1)
        v = np.where(ok, dc[..., 1] / np.maximum(z, 1e-3) * f + c, -1)
        inside = ok & (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
        wgt = np.where(inside, z, -1.0)
        samp = cv2.remap(lab.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_NEAREST)
        better = wgt > best_w
        out[better] = samp[better].astype(np.int32)
        best_w[better] = wgt[better]
    return out


def segment_pano(seg_model, seg_proc, pano: np.ndarray, device, views=VIEWS) -> np.ndarray:
    """ADE20K class map of a panorama: SegFormer on the perspective views, labels projected back."""
    from .analysis import segment_classes
    ims = view_images(pano, views)
    labs = [segment_classes(seg_model, seg_proc, im, device) for im in ims]
    return project_labels(labs, pano.shape[1], pano.shape[0], views)


# --------------------------------------------------------------------------- #
# Method B: DAP
# --------------------------------------------------------------------------- #
def load_dap(device):
    """DAP (Depth Any Panoramas, ViT-L) in fp32 (2.3 GB peak). Output of the network is distance / 100 m."""
    sys.path.insert(0, str(DAP_DIR))
    import os
    cwd = os.getcwd()
    os.chdir(DAP_DIR)                                   # the model resolves its DINOv3 code by relative path
    try:
        from networks.models import make
        import networks.dap  # noqa: F401  (registers "dap")
        m = make({"name": "dap", "args": {"midas_model_type": "vitl", "fine_tune_type": "hypersim",
                                          "min_depth": 0.01, "max_depth": 1.0, "train_decoder": True}})
    finally:
        os.chdir(cwd)
    state = torch.load(DAP_DIR / "model.pth", map_location="cpu", weights_only=True)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = m.load_state_dict(state, strict=False)
    m = m.float().to(device).eval()                    # fp16 overflows (all-NaN output): run in fp32
    return m, {"missing": len(missing), "unexpected": len(unexpected)}


@torch.inference_mode()
def dap_depth(model, pano: np.ndarray, device) -> np.ndarray:
    """Distance map (m) of an equirectangular RGB uint8 image, at 512 x 1024 then resized to the input."""
    h, w = pano.shape[:2]
    x = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    out = model(t)
    d = out["pred_depth"][0, 0].float().cpu().numpy() * 100.0
    if "pred_mask" in out:                              # sky / invalid mask predicted by the network
        bad = (1 - out["pred_mask"][0, 0].float().cpu().numpy()) <= 0.5
        d[bad] = np.nan
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.isnan(d).any():
        d = np.where(np.isnan(d), np.nanmedian(d), d)
    return np.clip(d, 0.05, MAX_DIST).astype(np.float32)


def dap_depth_wrap(model, pano: np.ndarray, device) -> np.ndarray:
    """DAP made seamless: DAP does not treat the left and right borders as neighbours, so its
    prediction jumps at lon = +-180 deg. It is run a second time on the panorama rolled by half a
    width (seam moved to the centre), the second result is rolled back, and the two are cross-faded
    with weights cos^2(lon / 2) and sin^2(lon / 2): each prediction is used where it is far from its
    own image border, and the weights sum to 1 everywhere."""
    w = pano.shape[1]
    d1 = dap_depth(model, pano, device)
    d2 = np.roll(dap_depth(model, np.roll(pano, w // 2, axis=1), device), -(w // 2), axis=1)
    lon = 2 * np.pi * (np.arange(w) + 0.5) / w - np.pi
    a = np.cos(lon / 2) ** 2
    return (a * d1 + (1 - a) * d2).astype(np.float32)


# --------------------------------------------------------------------------- #
# Wrap-aware post-processing
# --------------------------------------------------------------------------- #
def wrap_apply(fn, img: np.ndarray, pad: int = 32):
    """Apply an image filter on a horizontally wrap-padded copy and cut the padding off again."""
    ext = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_WRAP)
    out = fn(ext)
    return out[:, pad:-pad]


def refine_pano_distance(dist: np.ndarray, pano: np.ndarray) -> np.ndarray:
    """Depth refinement of README section 8 (median + bilateral, edge-aware, hole filling), wrap-aware."""
    pad = 32
    ext_rgb = cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP)
    ext_d = cv2.copyMakeBorder(dist, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return dp.refine_metric_depth(ext_d, ext_rgb)[:, pad:-pad]


def gated_preprocess_pano(pano: np.ndarray) -> np.ndarray:
    """Noise-gated denoising of the panorama (wrap-aware)."""
    from .preprocessing import preprocess_gated
    return wrap_apply(preprocess_gated, pano)


# --------------------------------------------------------------------------- #
# Geometry checks (for the comparison of methods)
# --------------------------------------------------------------------------- #
def to_points(dist: np.ndarray) -> np.ndarray:
    """Equirect distance map -> H x W x 3 points (camera axes, centre of the panorama at the origin)."""
    return eq.pixel_dirs(dist.shape[1], dist.shape[0]) * dist[..., None]
