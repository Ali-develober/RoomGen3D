"""Room box regularization (PLAN_V3 milestone 2): the room becomes a clean box with straight,
fully textured walls.

    distance map (DAP) + class map (v3 panoptic) of a panorama
      -> gravity frame A (room_mesh.gravity_rotation: floor plane -> "up" = -Y, camera at the origin)
      -> per-pixel surface normals (wrap-aware central differences of the point map)
      -> dominant wall directions: the horizontal normal angle of wall pixels, taken modulo 90 deg
         (4-fold symmetry of a rectangular room), histogram weighted by surface area, peak + local
         circular mean -> yaw; frame B = A rotated so that the walls are axis-aligned
      -> box fit per side (+x, -x, +z, -z): wall pixels whose normal faces that side, their position
         along the axis is histogrammed (2 cm bins, area-weighted), the PEAK is the wall. A side is
         "detected" if the peak holds >= DETECT_AREA m^2 of wall spanning >= DETECT_SPAN m; otherwise it is
         "inferred" from the floor extent (97th percentile of floor points) and furniture (99th).
         Floor = floor plane; ceiling = median height of ceiling pixels (fallback 2.7 m)
      -> shell: walls, floor and ceiling are exact planes (inward-facing quads). Each gets its own
         texture: every texel is projected into the panorama and coloured from it, so whatever is
         seen beyond a wall (doorway, window, opening) is FLATTENED onto the wall as texture.
         Texels hidden behind furniture (observed distance clearly shorter than the plane's) are
         filled by inpainting in the plane's own texture space (Telea / Navier-Stokes / patch-based)
      -> furniture: the observed surface (grid mesh with depth cuts) of furniture pixels in front of
         the shell (milestone 3 replaces this with solids)
All vertices are returned in frame A (the frame of the v2 mesh), so v2 and v3 can be compared directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import equirect as eq
from . import room_mesh as rm
from . import textured_mesh as tm

DETECT_AREA = 0.6           # m^2 of wall surface within +-PEAK_BAND of the peak
DETECT_SPAN = 0.8           # m: the supporting wall points must span at least this along the wall
PEAK_BAND = 0.08            # m
BIN = 0.02                  # m, histogram bin of the side fit
PEAK_RULE = "farthest"      # "max" or "farthest" (see _side_peak); chosen on the dev rooms (README, milestone 4)
FAR_SHARE = 0.3
NORMAL_FACING = 0.8         # |n . axis| above this: the pixel's surface faces that axis
TEXEL = 0.015               # m per texel of the shell textures (capped at MAX_TEX px per side)
MAX_TEX = 1024
HIDDEN_REL, HIDDEN_ABS = 0.04, 0.10   # texel hidden if observed distance < plane distance * (1 - rel) - abs
DEFAULT_CEILING = 2.7
INPAINT_METHODS = ("telea", "ns", "patch")
DEFAULT_INPAINT = "ns"      # chosen on the dev rooms (best SSIM on synthetic holes), scripts/eval_room_box.py


# --------------------------------------------------------------------------- #
# Normals and dominant directions
# --------------------------------------------------------------------------- #
def pixel_area(dist: np.ndarray) -> np.ndarray:
    """Surface area (m^2) seen by each equirectangular pixel, ignoring incidence: r^2 * solid angle."""
    h, w = dist.shape
    lat = np.pi / 2 - np.pi * (np.arange(h) + 0.5) / h
    return dist ** 2 * (2 * np.pi / w) * (np.pi / h) * np.cos(lat)[:, None]


def normals(P: np.ndarray, step: int = 2) -> np.ndarray:
    """Unit surface normals of an H x W x 3 point map (wrap-aware), oriented towards the camera."""
    dx = np.roll(P, -step, 1) - np.roll(P, step, 1)
    dy = np.zeros_like(P)
    dy[step:-step] = P[2 * step:] - P[:-2 * step]
    n = np.cross(dy, dx)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1
    return n


def dominant_yaw(n: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Yaw (rad, in [-pi/4, pi/4)) of the wall directions modulo 90 deg, and its peak strength."""
    nh = np.hypot(n[..., 0], n[..., 2])
    m = mask & (nh > 0.9)
    phi = np.arctan2(n[..., 2], n[..., 0])[m]
    w = (weight * nh ** 2)[m]
    if len(phi) < 100:
        return 0.0, 0.0
    a = np.degrees(phi) % 90.0
    hist, _ = np.histogram(a, bins=180, range=(0, 90), weights=w)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(np.r_[hist[-6:], hist, hist[:6]], k, "same")[6:-6]
    peak = (np.argmax(sm) + 0.5) * 0.5
    d = (a - peak + 45) % 90 - 45                         # circular offset from the peak
    near = np.abs(d) < 8
    yaw = peak + np.average(d[near], weights=w[near])
    yaw = (yaw + 45) % 90 - 45
    return float(np.radians(yaw)), float(sm.max() / max(sm.sum(), 1e-9))


def yaw_matrix(yaw: float) -> np.ndarray:
    """Q with P_B = P_A @ Q: rotates frame A about its vertical axis by -yaw (walls -> axis-aligned)."""
    c, s = np.cos(yaw), np.sin(yaw)
    # a wall normal at angle yaw (atan2(z, x)) must map to angle 0
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


# --------------------------------------------------------------------------- #
# Box fit
# --------------------------------------------------------------------------- #
def _side_peak(coord: np.ndarray, w: np.ndarray, along: np.ndarray, rule: str = None):
    """Peak of an area-weighted histogram of wall positions. Returns (position, area, span).
    rule "max": the strongest peak; "farthest": the farthest peak holding at least FAR_SHARE of the
    strongest one's area (and DETECT_AREA) - a large furniture front labelled "wall" is nearer than the wall."""
    rule = rule or PEAK_RULE
    if len(coord) < 50:
        return None
    lo, hi = coord.min(), coord.max()
    nb = max(int((hi - lo) / BIN) + 1, 16)
    hist, edges = np.histogram(coord, bins=nb, range=(lo, lo + nb * BIN), weights=w)
    sm = np.convolve(hist, np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2), "same")
    centres = (edges[:-1] + edges[1:]) / 2
    pk = centres[np.argmax(sm)]
    if rule == "farthest":
        k = int(round(PEAK_BAND / BIN))
        is_max = np.array([sm[i] > 0 and sm[i] == sm[max(0, i - k):i + k + 1].max() for i in range(nb)])
        area = np.array([w[np.abs(coord - c) < PEAK_BAND].sum() for c in centres[is_max]])
        cand = centres[is_max][area >= max(DETECT_AREA, FAR_SHARE * area.max())] if len(area) else []
        if len(cand):
            pk = float(np.max(cand))
    band = np.abs(coord - pk) < PEAK_BAND
    pos = float(np.average(coord[band], weights=w[band])) if band.any() else float(pk)
    span = float(np.subtract(*np.percentile(along[band], [97, 3]))) if band.sum() > 20 else 0.0
    return pos, float(w[band].sum()), span


def fit_box(dist: np.ndarray, class_map: np.ndarray, id2label: dict) -> dict:
    """Axis-aligned room box in frame B (walls along X and Z). See the module docstring."""
    from .floorplan import CEILING_CLASSES, FLOOR_CLASSES, FURNITURE, WALL_CLASSES, class_ids
    R, frame = rm.gravity_rotation(dist, class_map, id2label)
    PA = rm.aligned_points(dist, R)
    nA = normals(PA)
    area = pixel_area(dist)
    sel = lambda names: np.isin(class_map, class_ids(id2label, names))  # noqa: E731
    wall, floor, ceil = sel(WALL_CLASSES), sel(FLOOR_CLASSES), sel(CEILING_CLASSES)
    furn = sel(set(FURNITURE))
    yaw, strength = dominant_yaw(nA, wall, area)
    Q = yaw_matrix(yaw)
    P, n = PA @ Q, nA @ Q
    cam_h = frame["d"]
    height = cam_h - P[..., 1]                                    # Y is down
    wall_ok = wall & (height > 0.3) & (height < 2.2)
    inc = np.maximum(np.abs((n * P).sum(-1)) / np.maximum(np.linalg.norm(P, axis=-1), 1e-6), 0.2)
    w_area = area / inc                                            # true surface area of the pixel

    sides = {}
    for ax, other in ((0, 2), (2, 0)):
        for sgn in (1, -1):
            name = f"{'+' if sgn > 0 else '-'}{'xz'[ax // 2]}"
            m = wall_ok & (sgn * n[..., ax] < -NORMAL_FACING) & (sgn * P[..., ax] > 0.3)
            pk = _side_peak(sgn * P[..., ax][m], w_area[m], P[..., other][m])
            fl = sgn * P[..., ax][floor & (sgn * P[..., ax] > 0)]
            fu = sgn * P[..., ax][furn & (sgn * P[..., ax] > 0) & (height > 0.02)]
            infer = max(np.percentile(fl, 97) if len(fl) > 50 else 0.0,
                        np.percentile(fu, 99) + 0.02 if len(fu) > 50 else 0.0, 1.0)
            if pk is not None and pk[1] >= DETECT_AREA and pk[2] >= DETECT_SPAN:
                sides[name] = {"pos": pk[0], "detected": True, "area": pk[1], "span": pk[2]}
            else:
                sides[name] = {"pos": float(infer), "detected": False, "area": pk[1] if pk else 0.0,
                               "span": pk[2] if pk else 0.0}
    hc = height[ceil & (height > cam_h + 0.2)]
    ceil_h = float(np.median(hc)) if len(hc) > 200 else max(DEFAULT_CEILING, cam_h + 0.5)
    x0, x1 = -sides["-x"]["pos"], sides["+x"]["pos"]
    z0, z1 = -sides["-z"]["pos"], sides["+z"]["pos"]
    return {"R": R, "frame": frame, "yaw": yaw, "yaw_strength": strength, "Q": Q, "x": (x0, x1), "z": (z0, z1),
            "floor_y": cam_h, "ceil_y": cam_h - ceil_h, "camera_height": cam_h, "ceiling_height": ceil_h,
            "sides": sides, "n_detected": int(sum(s["detected"] for s in sides.values())),
            "width": x1 - x0, "depth": z1 - z0, "area": (x1 - x0) * (z1 - z0)}


def box_corners_A(box: dict) -> np.ndarray:
    """4 floor-plan corners (x, z) of the box in frame A (counter-clockwise from (x0, z0))."""
    (x0, x1), (z0, z1) = box["x"], box["z"]
    cB = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], float)
    cA = cB @ box["Q"].T
    return cA[:, [0, 2]]


# --------------------------------------------------------------------------- #
# Inpainting in texture space
# --------------------------------------------------------------------------- #
def patch_inpaint(img: np.ndarray, mask: np.ndarray, patch: int = 9, work: int = 192, n_cand: int = 800,
                  seed: int = 0) -> np.ndarray:
    """Exemplar-based fill (Criminisi-style, simplified): at a working resolution of `work` px, the hole
    is filled from its boundary inwards; each step copies the known source patch with the smallest
    SSD to the known part of the target patch (candidates: n_cand random fully-known patches).
    The low-resolution fill is upsampled and used only inside the hole."""
    h, w = mask.shape
    if not mask.any() or mask.all():
        return cv2.inpaint(img, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA) if mask.any() else img
    s = min(1.0, work / max(h, w))
    sw, sh = max(int(w * s), patch + 2), max(int(h * s), patch + 2)
    im = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    hole = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    r = patch // 2
    pad = lambda a, v: np.pad(a, ((r, r), (r, r)) + ((0, 0),) * (a.ndim - 2), constant_values=v)  # noqa: E731
    imp, holep = pad(im, 0), pad(hole, True)
    known_full = ~holep
    win = np.lib.stride_tricks.sliding_window_view
    kp = win(known_full, (patch, patch)).all((-1, -2))           # fully known source patches (centre coords)
    ys, xs = np.nonzero(kp)
    if len(ys) == 0:
        return cv2.inpaint(img, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ys), min(n_cand, len(ys)), replace=False)
    src = np.stack([imp[y:y + patch, x:x + patch] for y, x in zip(ys[pick], xs[pick])])   # (n, p, p, 3)
    conf = known_full.astype(np.float32)
    for _ in range(int(hole.sum()) + 10):
        if not holep[r:-r, r:-r].any():
            break
        front = holep & (cv2.dilate((~holep).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        front[:r], front[-r:], front[:, :r], front[:, -r:] = False, False, False, False
        fy, fx = np.nonzero(front)
        if len(fy) == 0:
            break
        cs = cv2.boxFilter(conf, -1, (patch, patch), normalize=True)[fy, fx]
        i = int(np.argmax(cs))
        y, x = fy[i], fx[i]
        tgt = imp[y - r:y + r + 1, x - r:x + r + 1]
        kn = ~holep[y - r:y + r + 1, x - r:x + r + 1]
        d = (((src - tgt[None]) ** 2).sum(-1) * kn[None]).sum((1, 2))
        best = src[int(np.argmin(d))]
        fill = ~kn
        tgt[fill] = best[fill]
        conf[y - r:y + r + 1, x - r:x + r + 1][fill] = cs[i]
        holep[y - r:y + r + 1, x - r:x + r + 1] = False
    low = imp[r:-r, r:-r]
    up = cv2.resize(np.clip(low, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[mask] = up[mask]
    return out


def _low_base(img: np.ndarray, mask: np.ndarray, sigma: float = 6.0, work: int = 512) -> np.ndarray:
    """Smooth shading base inside the hole: NS inpainting of the blurred texture at a working resolution."""
    h, w = mask.shape
    s_ = min(1.0, work / max(h, w))
    sw, sh = max(int(w * s_), 8), max(int(h * s_), 8)
    small = cv2.resize(cv2.GaussianBlur(img, (0, 0), sigma), (sw, sh), interpolation=cv2.INTER_AREA)
    hm = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    hm = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    base = cv2.inpaint(small, hm, 5, cv2.INPAINT_NS)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def with_base(fill: np.ndarray, img: np.ndarray, mask: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """PLAN_V4 milestone 5: keep only the detail band of a fill (fill - G*fill) on top of a smooth NS base, so
    copied texture brings structure but no colour blotches."""
    f = fill.astype(np.float32)
    detail = f - cv2.GaussianBlur(f, (0, 0), sigma)
    out = img.astype(np.float32).copy()
    out[mask] = (_low_base(img, mask, sigma) + detail)[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def find_period(img: np.ndarray, valid: np.ndarray, min_shift: int = 6, work: int = 512) -> tuple | None:
    """Dominant repetition (dy, dx) of a texture from its masked, normalised autocorrelation (FFT) - floor tiles,
    planks, wallpaper. Returns (dy, dx, correlation) at full resolution or None."""
    h, w = valid.shape
    s_ = min(1.0, work / max(h, w))
    sw, sh = max(int(w * s_), 16), max(int(h * s_), 16)
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float64)
    v = cv2.resize(valid.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.float64)
    if v.sum() < 0.2 * v.size:
        return None
    g = g - cv2.GaussianBlur(g, (0, 0), 8)                           # remove shading, keep the pattern
    g = (g - (g * v).sum() / v.sum()) * v
    H2, W2 = 2 * sh, 2 * sw
    Fg, Fv = np.fft.rfft2(g, (H2, W2)), np.fft.rfft2(v, (H2, W2))
    num = np.fft.irfft2(Fg * np.conj(Fg), (H2, W2))
    den = np.fft.irfft2(Fv * np.conj(Fv), (H2, W2))
    var = (g * g).sum() / v.sum()
    corr = num / np.maximum(den * var, 1e-9)
    corr[den < 0.3 * v.sum()] = -1
    ms = max(2, int(round(min_shift * s_)))
    yy, xx = np.meshgrid(np.fft.fftfreq(H2) * H2, np.fft.fftfreq(W2) * W2, indexing="ij")
    corr[(np.abs(yy) < ms) & (np.abs(xx) < ms)] = -1
    corr[(np.abs(yy) > sh * 0.6) | (np.abs(xx) > sw * 0.6)] = -1
    k = int(np.argmax(corr))
    c = float(corr.flat[k])
    dy, dx = int(round(yy.flat[k] / s_)), int(round(xx.flat[k] / s_))
    return (dy, dx, c)


def tile_inpaint(img: np.ndarray, mask: np.ndarray, thr: float = 0.5, kmax: int = 40) -> np.ndarray:
    """Periodic fill: every hole pixel is copied from the nearest visible pixel along the texture's dominant
    period (find_period); falls back to patch_inpaint without a clear period (correlation < thr)."""
    per = find_period(img, ~mask)
    if per is None or per[2] < thr:
        return patch_inpaint(img, mask, work=384)
    dy, dx, _ = per
    h, w = mask.shape
    out = img.copy()
    todo = mask.copy()
    ys, xs = np.nonzero(todo)
    for k in range(1, kmax + 1):
        if len(ys) == 0:
            break
        for sgn in (1, -1):
            sy, sx = ys + sgn * k * dy, xs + sgn * k * dx
            ok = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            ok[ok] &= ~mask[sy[ok], sx[ok]]
            out[ys[ok], xs[ok]] = img[sy[ok], sx[ok]]
            todo[ys[ok], xs[ok]] = False
            keep = ~ok
            ys, xs = ys[keep], xs[keep]
    if todo.any():
        out = patch_inpaint(out, todo, work=384)
    return out


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = DEFAULT_INPAINT) -> np.ndarray:
    """Fill `mask` (bool) of an RGB uint8 texture with one of INPAINT_METHODS (+ PLAN_V4: "tile", "patch_base",
    "tile_base" - the *_base variants keep only the fill's detail band on a smooth NS base)."""
    if not mask.any() or method == "none":
        return img
    if method == "patch":
        return patch_inpaint(img, mask)
    if method == "tile":
        return tile_inpaint(img, mask)
    if method == "patch_base":
        return with_base(patch_inpaint(img, mask, work=384), img, mask)
    if method == "tile_base":
        return with_base(tile_inpaint(img, mask), img, mask)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask.astype(np.uint8), 5, flag)


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
def _sample_pano(img: np.ndarray, dirs_cam: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Sample an equirectangular image (any channels) along camera-frame directions (..., 3)."""
    h, w = img.shape[:2]
    u, v = eq.dirs_to_uv(dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True), w, h)
    ext = cv2.copyMakeBorder(img, 0, 0, 4, 4, cv2.BORDER_WRAP)
    return cv2.remap(ext, (u + 4).astype(np.float32), v.astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)


def shell_planes(box: dict) -> list[dict]:
    """The six planes of the box in frame B as rectangles: origin o, edge vectors a (texture u) and
    b (texture v, downwards / across), kind."""
    (x0, x1), (z0, z1) = box["x"], box["z"]
    yc, yf = box["ceil_y"], box["floor_y"]
    P = lambda x, y, z: np.array([x, y, z], float)  # noqa: E731
    return [  # walls seen from inside: u runs left -> right, v from the ceiling down
        {"kind": "wall+z", "o": P(x0, yc, z1), "a": P(x1 - x0, 0, 0), "b": P(0, yf - yc, 0)},
        {"kind": "wall+x", "o": P(x1, yc, z1), "a": P(0, 0, z0 - z1), "b": P(0, yf - yc, 0)},
        {"kind": "wall-z", "o": P(x1, yc, z0), "a": P(x0 - x1, 0, 0), "b": P(0, yf - yc, 0)},
        {"kind": "wall-x", "o": P(x0, yc, z0), "a": P(0, 0, z1 - z0), "b": P(0, yf - yc, 0)},
        {"kind": "floor", "o": P(x0, yf, z1), "a": P(x1 - x0, 0, 0), "b": P(0, 0, z0 - z1)},
        {"kind": "ceiling", "o": P(x0, yc, z0), "a": P(x1 - x0, 0, 0), "b": P(0, 0, z1 - z0)},
    ]


def plane_texture(pl: dict, box: dict, texture: np.ndarray, dist: np.ndarray, method: str = DEFAULT_INPAINT,
                  texel_m: float = TEXEL, max_tex: int = MAX_TEX, band_m: float = 0.0):
    """Texture of one shell plane from the panorama, hidden texels inpainted.
    PLAN_V4: texel_m / max_tex set the resolution; band_m > 0 keeps the projected panorama colour (no
    inpainting) in a band of that width inside the hidden region along its border - from the panorama centre
    that band is exactly what is seen past a furniture silhouette.
    Returns (rgb, hidden mask, beyond mask)."""
    la, lb = np.linalg.norm(pl["a"]), np.linalg.norm(pl["b"])
    texel = max(texel_m, max(la, lb) / max_tex)
    tw, th = max(int(np.ceil(la / texel)), 8), max(int(np.ceil(lb / texel)), 8)
    s = (np.arange(tw) + 0.5) / tw
    t = (np.arange(th) + 0.5) / th
    PB = pl["o"] + s[None, :, None] * pl["a"] + t[:, None, None] * pl["b"]          # (th, tw, 3)
    Pcam = (PB @ box["Q"].T) @ box["R"]                                              # B -> A -> camera
    rgb = _sample_pano(texture, Pcam)
    obs = _sample_pano(dist.astype(np.float32), Pcam)
    r = np.linalg.norm(Pcam, axis=-1)
    hidden = obs < r * (1 - HIDDEN_REL) - HIDDEN_ABS
    beyond = obs > r * (1 + HIDDEN_REL) + HIDDEN_ABS
    hidden = cv2.morphologyEx(hidden.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    hidden = cv2.dilate(hidden, np.ones((5, 5), np.uint8)) > 0                       # include mixed edge texels
    fill = hidden
    if band_m > 0:
        k = 2 * int(round(band_m / texel)) + 1
        fill = cv2.erode(hidden.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return inpaint(rgb, fill, method), hidden, beyond


def build_shell(box: dict, texture: np.ndarray, dist: np.ndarray, method: str = DEFAULT_INPAINT,
                texel_m: float = TEXEL, max_tex: int = MAX_TEX, band_m: float = 0.0) -> dict:
    """Textured shell (6 inward-facing quads) in frame A. Returns a part dict for combine()."""
    parts = []
    stats = {}
    for pl in shell_planes(box):
        img, hid, bey = plane_texture(pl, box, texture, dist, method, texel_m, max_tex, band_m)
        cB = np.stack([pl["o"], pl["o"] + pl["a"], pl["o"] + pl["a"] + pl["b"], pl["o"] + pl["b"]])
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        f = np.array([[0, 1, 2], [0, 2, 3]])
        # inward-facing: the normal must point to the box centre
        ctr = np.array([np.mean(box["x"]), (box["ceil_y"] + box["floor_y"]) / 2, np.mean(box["z"])])
        nrm = np.cross(cB[1] - cB[0], cB[2] - cB[0])
        if nrm @ (ctr - cB.mean(0)) < 0:
            f = f[:, ::-1]
        parts.append({"vertices": cB @ box["Q"].T, "uvs": uv, "faces": f, "texture": img, "kind": pl["kind"]})
        stats[pl["kind"]] = {"hidden_share": float(hid.mean()), "beyond_share": float(bey.mean())}
    return {"parts": parts, "stats": stats}


# --------------------------------------------------------------------------- #
# Observed surface of selected pixels (furniture), and mesh assembly
# --------------------------------------------------------------------------- #
def observed_part(dist: np.ndarray, texture: np.ndarray, R: np.ndarray, keep: np.ndarray, stride: int = 2) -> dict:
    """Grid mesh of the pixels in `keep` (H x W bool) with depth cuts, in frame A; wraps at the seam."""
    H, W = dist.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    pick = np.ix_(ys, xs)
    d = rm._wrap_filter(lambda e: tm.grow_foreground(tm.sharpen_depth_edges(e)), dist.astype(np.float32), 8)[pick]
    k = keep[pick]
    ext = lambda a: np.concatenate([a, a[:, :1]], 1)  # noqa: E731
    de, ke = ext(d).astype(np.float64), ext(k)
    ch, cv = tm.depth_cuts(de, None, tm.CUT_STRONG)
    faces = tm.grid_faces(ke, ch, cv)
    dirs = ext((eq.pixel_dirs(W, H) @ R.T)[pick])
    v = (dirs * de[..., None]).reshape(-1, 3)
    gx = ext(np.broadcast_to(xs[None].astype(float), (len(ys), len(xs))).copy())
    gx[:, -1] = W
    gy = np.broadcast_to(ys[:, None].astype(float), gx.shape)
    uv = np.stack([(gx + 0.5) / W, (gy + 0.5) / H], -1).reshape(-1, 2)
    used = np.unique(faces)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return {"vertices": v[used], "uvs": uv[used], "faces": remap[faces] if len(faces) else np.zeros((0, 3), int),
            "texture": texture, "kind": "observed"}


def combine(parts: list[dict]) -> dict:
    """Merge parts with separate textures into one mesh with a vertically stacked texture atlas."""
    Wa = max(p["texture"].shape[1] for p in parts)
    Ha = sum(p["texture"].shape[0] for p in parts)
    atlas = np.full((Ha, Wa, 3), 127, np.uint8)
    verts, uvs, faces, kinds = [], [], [], []
    y0, base = 0, 0
    for i, p in enumerate(parts):
        h, w = p["texture"].shape[:2]
        atlas[y0:y0 + h, :w] = p["texture"]
        uv = p["uvs"] * np.array([w / Wa, h / Ha]) + np.array([0, y0 / Ha])
        verts.append(p["vertices"])
        uvs.append(uv)
        faces.append(np.asarray(p["faces"]) + base)
        kinds += [p["kind"]] * len(p["faces"])
        base += len(p["vertices"])
        y0 += h
    return {"vertices": np.concatenate(verts).astype(np.float32), "uvs": np.concatenate(uvs).astype(np.float32),
            "faces": np.concatenate(faces).astype(np.uint32), "texture": atlas, "face_kind": np.array(kinds)}


def furniture_keep(dist: np.ndarray, class_map: np.ndarray, id2label: dict, box: dict) -> np.ndarray:
    """Furniture pixels that lie inside the box (at least 3 cm in front of every shell plane)."""
    from .floorplan import FURNITURE, class_ids
    furn = np.isin(class_map, class_ids(id2label, set(FURNITURE)))
    P = rm.aligned_points(dist, box["R"]) @ box["Q"]
    (x0, x1), (z0, z1) = box["x"], box["z"]
    m = 0.03
    inside = ((P[..., 0] > x0 + m) & (P[..., 0] < x1 - m) & (P[..., 2] > z0 + m) & (P[..., 2] < z1 - m)
              & (P[..., 1] < box["floor_y"] - 0.01) & (P[..., 1] > box["ceil_y"] + 0.01))
    return furn & inside


def build_room_v3(dist: np.ndarray, texture: np.ndarray, class_map: np.ndarray, id2label: dict,
                  method: str = DEFAULT_INPAINT, furniture: bool = True) -> dict:
    """Milestone-2 room: box shell + observed furniture surfaces, one mesh in frame A."""
    box = fit_box(dist, class_map, id2label)
    shell = build_shell(box, texture, dist, method)
    parts = list(shell["parts"])
    if furniture:
        keep = furniture_keep(dist, class_map, id2label, box)
        parts.append(observed_part(dist, texture, box["R"], keep))
    mesh = combine(parts)
    mesh.update({"box": box, "shell_stats": shell["stats"], "camera_height": box["camera_height"]})
    return mesh


# --------------------------------------------------------------------------- #
# PLAN_V4 milestone 5: open sides
# --------------------------------------------------------------------------- #
OPEN_THR = 0.15             # development choice (results/tables/open_sides_selection.json)


def side_beyond_shares(box: dict, dist: np.ndarray, rel: float = 0.10, abs_: float = 0.10) -> dict:
    """For each box side: the share (solid angle) of the rays that leave the box through that side's wall area
    whose observed distance lies clearly BEYOND it (> exit * (1 + rel) + abs_): an opening into another space."""
    from .boxiness import box_exit
    H, W = dist.shape
    dB = (eq.pixel_dirs(W, H) @ box["R"].T) @ box["Q"]
    r_exit = box_exit(dB, box)
    P = dB * r_exit[..., None]
    (x0, x1), (z0, z1) = box["x"], box["z"]
    sa = np.cos(np.pi / 2 - np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    beyond = dist > r_exit * (1 + rel) + abs_
    out = {}
    for name, ax, val in (("+x", 0, x1), ("-x", 0, x0), ("+z", 2, z1), ("-z", 2, z0)):
        on = np.abs(P[..., ax] - val) < 1e-3
        out[name] = float(sa[on & beyond].sum() / max(sa[on].sum(), 1e-9)) if on.any() else 0.0
    return out


def mark_open_sides(box: dict, dist: np.ndarray, class_map: np.ndarray, id2label: dict, thr: float | None = None,
                    min_gap: float = 0.5) -> dict:
    """Open plan / corridor / next room: v3 places each side at the FARTHEST supported wall peak (or, without a
    measured wall, at the 97th floor percentile), so a wall of the adjoining space can become the room's wall.
    Here the box is also fitted with the main-wall rule (largest wall peak, PEAK_RULE "max"); a side whose main
    wall lies >= min_gap closer is OPEN if, through that main wall, a share >= thr of the view goes clearly
    beyond it (side_beyond_shares): the side is drawn as OPEN and placed at the main wall or, if furniture is seen
    beyond it (furniture of the adjoining space cannot be told apart), at the furniture extent (99th percentile
    + 0.2 m) - never cutting furniture and never further than v3.
    Returns a new box (v3 boxes unchanged)."""
    global PEAK_RULE
    thr = OPEN_THR if thr is None else thr
    old = PEAK_RULE
    try:
        PEAK_RULE = "max"
        bm = fit_box(dist, class_map, id2label)
    finally:
        PEAK_RULE = old
    bm["R"], bm["Q"] = box["R"], box["Q"]
    sh = side_beyond_shares(bm, dist)
    from .floorplan import FURNITURE, class_ids
    P = rm.aligned_points(dist, box["R"]) @ box["Q"]
    furn = np.isin(class_map, class_ids(id2label, set(FURNITURE))) & (box["camera_height"] - P[..., 1] > 0.02)
    b = {**box, "sides": {k: dict(v) for k, v in box["sides"].items()}}
    for name, s_ in b["sides"].items():
        near = bm["sides"][name]["pos"]
        ax, sgn = (0 if name[1] == "x" else 2), (1.0 if name[0] == "+" else -1.0)
        fx = sgn * P[..., ax][furn & (sgn * P[..., ax] > 0)]
        f_ext = float(np.percentile(fx, 99)) + 0.2 if len(fx) > 50 else 0.0      # guard: keep the furniture inside
        s_["beyond_share"] = sh[name]
        s_["open"] = bool(near <= s_["pos"] - min_gap and sh[name] >= thr)
        if s_["open"]:
            s_["pos_v3"], s_["pos"] = s_["pos"], float(min(s_["pos"], max(near, f_ext)))
    x0, x1 = -b["sides"]["-x"]["pos"], b["sides"]["+x"]["pos"]
    z0, z1 = -b["sides"]["-z"]["pos"], b["sides"]["+z"]["pos"]
    b.update({"x": (x0, x1), "z": (z0, z1), "width": x1 - x0, "depth": z1 - z0, "area": (x1 - x0) * (z1 - z0),
              "n_open": int(sum(v["open"] for v in b["sides"].values()))})
    return b
