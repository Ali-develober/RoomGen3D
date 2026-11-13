"""Architect-style floor plan from a metric depth map + semantic segmentation.

    depth (m) + camera intrinsics + ADE20K class map
      -> back-projection to 3D points                                    (pinhole model)
      -> gravity alignment with the fitted floor plane                   (gravity_frame)
      -> orthographic top-down projection to a 2D grid in metres         (to_plan)
      -> walls: occupancy grid of wall points, morphological closing /   (extract_walls)
         opening, sequential RANSAC line fitting, Manhattan snapping,
         merging of collinear segments, corner snapping
      -> furniture: connected components per class, minimum-area         (extract_furniture)
         rectangle of each projected instance, labelled with its class
      -> doors / windows: projected onto the nearest wall line           (extract_openings)
      -> drawing: walls, door / window symbols, furniture, dimension      (draw_floorplan)
         lines, scale bar, camera marker, floor area; PNG + SVG

Plan coordinates: x to the right of the camera, y forward (both horizontal, in metres), camera
at the origin. Heights are measured above the fitted floor plane.

All dimensions are APPROXIMATE: Depth Anything V2 Metric Indoor is about 22% off in absolute
scale on NYU Depth V2 (README section 8), and a generated image has an assumed camera.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Parameters (fixed before the NYU validation run)
# --------------------------------------------------------------------------- #
CELL = 0.02                 # m per cell of the top-down occupancy grid
WALL_MIN_HEIGHT = 0.30      # wall points lower than this (skirting, floor bleed) are ignored
WALL_RANSAC_TOL = 0.06      # m: a grid cell is an inlier of a wall line within this distance ...
WALL_TOL_REL = 0.02         # ... or within 2% of its distance from the camera (depth noise grows with range)
WALL_RANSAC_ITERS = 400
WALL_MIN_LENGTH = 0.40      # m: shorter wall segments are dropped
WALL_MIN_CELLS = 30         # a wall line needs at least this many occupied cells
WALL_SPLIT_GAP = 0.35       # m: a gap this long along a line splits it into two segments
MAX_WALLS = 10
MANHATTAN_TOL = 12.0        # deg: segments this close to a dominant direction are snapped to it
MANHATTAN_MIN_SUPPORT = 0.6 # share of the wall length that must agree before snapping is applied
MERGE_ANGLE = 6.0           # deg   } collinear segments are merged if their directions, offsets
MERGE_OFFSET = 0.15         # m     } and the gap between them are all below these (offset: or 4% of the
MERGE_OFFSET_REL = 0.04     #       }   wall's distance, whichever is larger)
MERGE_GAP = 0.50            # m     }
FREE_SPACE_REL = 0.05       # } a scene point is "behind" a wall if it is 5% + 10 cm further along its ray
FREE_SPACE_ABS = 0.10       # }
FREE_SPACE_MAX_BEHIND = 0.15  # a wall line with more than 15% of its viewing wedge behind it is rejected
CORNER_REACH = 0.60         # m: perpendicular walls are extended to their corner if it is this close
WALL_THICKNESS = 0.10       # m, drawing only (drawn outwards, so the inner face is the measured line)
FLOOR_MAX_TILT = 45.0       # deg: a fitted floor / ceiling plane further from the image vertical is rejected
FLOOR_MIN_INLIERS = 0.5
INSTANCE_MIN_SHARE = 0.0012 # an object instance must cover at least 0.12% of the image
INSTANCE_ERODE = 3          # px: instance masks are eroded so boundary pixels (often background) are not used
OPENING_MAX_DIST = 0.35     # m: a door / window is attached to a wall line within this distance
OPENING_MIN_WIDTH = 0.30

# ADE20K class name -> (label on the plan, typical minimum footprint depth in m).
# The depth prior completes footprints whose back is hidden (a single view sees only the front face):
# if the observed short side is thinner than the prior, the rectangle is extended AWAY from the camera.
FURNITURE = {
    "bed": ("bed", 1.00), "sofa": ("sofa", 0.80), "armchair": ("armchair", 0.70), "chair": ("chair", 0.45),
    "swivel chair": ("chair", 0.50), "seat": ("seat", 0.45), "stool": ("stool", 0.35), "bench": ("bench", 0.40),
    "ottoman": ("ottoman", 0.45), "table": ("table", 0.60), "coffee table": ("coffee table", 0.45),
    "desk": ("desk", 0.55), "cabinet": ("cabinet", 0.40), "wardrobe": ("wardrobe", 0.55),
    "chest of drawers": ("drawers", 0.45), "shelf": ("shelf", 0.30), "bookcase": ("bookcase", 0.30),
    "buffet": ("sideboard", 0.45), "counter": ("counter", 0.60), "countertop": ("counter", 0.60),
    "kitchen island": ("island", 0.80), "bathtub": ("bathtub", 0.70), "toilet": ("WC", 0.60),
    "sink": ("sink", 0.45), "refrigerator": ("fridge", 0.65), "stove": ("stove", 0.60), "oven": ("oven", 0.55),
    "dishwasher": ("dishwasher", 0.55), "washer": ("washer", 0.55), "television receiver": ("TV", 0.10),
    "lamp": ("lamp", 0.30), "plant": ("plant", 0.35), "pot": ("plant", 0.30), "fireplace": ("fireplace", 0.40),
    "pool table": ("pool table", 1.20), "cradle": ("cradle", 0.50), "shower": ("shower", 0.80),
}
OPENINGS = {"door": "door", "screen door": "door", "windowpane": "window"}
FLOOR_CLASSES = {"floor", "rug"}
WALL_CLASSES = {"wall"}
CEILING_CLASSES = {"ceiling"}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def backproject(depth: np.ndarray, k: tuple) -> np.ndarray:
    """HxW depth (m) -> HxWx3 points in camera coordinates (X right, Y down, Z forward)."""
    fx, fy, cx, cy = k
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)


def class_ids(id2label: dict, names) -> list[int]:
    """ADE20K ids whose label (comma-separated synonyms) contains one of `names`."""
    names = set(names)
    out = []
    for i, label in id2label.items():
        if {n.strip().lower() for n in str(label).split(",")} & names:
            out.append(int(i))
    return out


def _ransac_plane(points: np.ndarray, thresh: float, iters: int = 300, seed: int = 0):
    """Plane n.x + d = 0 (unit n, d > 0 so the camera is on the positive side) + inlier mask."""
    rng = np.random.default_rng(seed)
    best, best_inl = None, None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        inl = np.abs(points @ n - n @ a) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = (n, -n @ a), inl
    if best is None:
        return None
    p = points[best_inl]                             # least-squares refit on the inliers
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, float(d), np.abs(points @ n + d) < thresh


def gravity_frame(pts: np.ndarray, valid: np.ndarray, floor: np.ndarray, ceiling: np.ndarray) -> dict:
    """Up direction, floor offset and horizontal axes of the plan.

    Order of preference: (1) RANSAC plane on the floor pixels, (2) RANSAC plane on the ceiling
    pixels (floor height then taken as the 1st percentile of the point heights), (3) camera assumed
    upright (up = -Y) with the floor at the 1st percentile. Returns up (unit, camera coords),
    d (height of the camera above the floor), right / fwd (plan axes) and the source used.
    """
    p_all = pts[valid]
    scale = float(np.median(p_all[:, 2]))
    thresh = max(0.02, 0.015 * scale)
    cam_up = np.array([0.0, -1.0, 0.0])
    up, d, src = None, None, "camera upright (no floor / ceiling plane)"
    for name, mask, sign in (("floor", floor, 1.0), ("ceiling", ceiling, -1.0)):
        m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        p = pts[m & valid]
        if len(p) < 500:
            continue
        if len(p) > 40_000:
            p = p[np.random.default_rng(0).choice(len(p), 40_000, replace=False)]
        fit = _ransac_plane(p, thresh)
        if fit is None:
            continue
        n, dd, inl = fit
        n_up = n if name == "floor" else -n          # floor: n points up (to the camera); ceiling: down
        if inl.mean() < FLOOR_MIN_INLIERS or n_up @ cam_up < np.cos(np.radians(FLOOR_MAX_TILT)):
            continue
        up = n_up
        d = dd if name == "floor" else None
        src = f"{name} plane"
        break
    if up is None:
        up = cam_up
    if d is None:                                    # floor height from the lowest points
        d = float(-np.percentile(p_all @ up, 1))
    right = np.array([1.0, 0.0, 0.0]) - up[0] * up    # camera X projected onto the horizontal plane
    right /= np.linalg.norm(right)
    fwd = np.cross(up, right)
    return {"up": up, "d": float(d), "right": right, "fwd": fwd, "source": src}


def to_plan(pts: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) camera points -> ((N,2) plan coordinates x right / y forward, (N,) height above floor)."""
    return np.stack([pts @ frame["right"], pts @ frame["fwd"]], -1), pts @ frame["up"] + frame["d"]


# --------------------------------------------------------------------------- #
# Occupancy grid
# --------------------------------------------------------------------------- #
class Grid:
    """Top-down grid (CELL m per cell) covering a set of plan points."""

    def __init__(self, xy: np.ndarray, margin: float = 0.5, cell: float = CELL):
        self.cell = cell
        self.x0, self.y0 = xy.min(0) - margin
        x1, y1 = xy.max(0) + margin
        self.w = int(np.ceil((x1 - self.x0) / cell)) + 1
        self.h = int(np.ceil((y1 - self.y0) / cell)) + 1

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.clip(((xy[:, 0] - self.x0) / self.cell).astype(int), 0, self.w - 1)
        r = np.clip(((xy[:, 1] - self.y0) / self.cell).astype(int), 0, self.h - 1)
        return r, c

    def occupancy(self, xy: np.ndarray) -> np.ndarray:
        occ = np.zeros((self.h, self.w), np.uint8)
        r, c = self.index(xy)
        occ[r, c] = 1
        return occ

    def centres(self, occ: np.ndarray) -> np.ndarray:
        r, c = np.nonzero(occ)
        return np.stack([self.x0 + (c + 0.5) * self.cell, self.y0 + (r + 0.5) * self.cell], -1)


def clean_occupancy(occ: np.ndarray, close: int = 5, open_: int = 3) -> np.ndarray:
    """Morphological closing (fills gaps between projected points) then opening (removes specks)."""
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_, open_))
    return cv2.morphologyEx(cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k1), cv2.MORPH_OPEN, k2)


# --------------------------------------------------------------------------- #
# Walls
# --------------------------------------------------------------------------- #
def _segment(direction: np.ndarray, offset: float, t0: float, t1: float, cells: int, inl: np.ndarray | None = None) -> dict:
    """Wall segment on the line {p : n.p = offset}, n = direction rotated by -90 deg, from t0 to t1 along it."""
    direction = direction / np.linalg.norm(direction)
    n = np.array([direction[1], -direction[0]])
    if offset < 0:                                     # normal points away from the camera (origin)
        n, offset = -n, -offset
    base = n * offset
    return {"dir": direction, "n": n, "offset": float(offset), "p0": base + t0 * direction,
            "p1": base + t1 * direction, "length": float(t1 - t0), "cells": int(cells), "inliers": inl,
            "snapped": False}


def _line_runs(t: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """Split sorted positions t into runs separated by more than `gap`; returns (start, end) index pairs."""
    breaks = np.nonzero(np.diff(t) > gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(t)]
    return list(zip(starts, ends))


def ransac_lines(pts: np.ndarray, tol: float = WALL_RANSAC_TOL, iters: int = WALL_RANSAC_ITERS,
                 min_cells: int = WALL_MIN_CELLS, max_lines: int = MAX_WALLS, seed: int = 0) -> list[dict]:
    """Sequential RANSAC line fitting on 2D points (occupied cell centres of the wall map).

    Each round: sample point pairs, keep the line with most inliers, refit it by total least squares
    (PCA), split its inliers at gaps longer than WALL_SPLIT_GAP, keep the runs of at least
    WALL_MIN_LENGTH, and remove all inliers before the next round.
    """
    rng = np.random.default_rng(seed)
    segs, pool = [], pts.copy()
    ptol = np.maximum(tol, WALL_TOL_REL * np.linalg.norm(pool, axis=1))     # per-point inlier distance
    for _ in range(4 * max_lines):
        if len(pool) < min_cells or len(segs) >= max_lines:
            break
        i = rng.integers(0, len(pool), (iters, 2))
        a, b = pool[i[:, 0]], pool[i[:, 1]]
        d = b - a
        norm = np.linalg.norm(d, axis=1)
        ok = norm > 5 * CELL
        if not ok.any():
            break
        a, d = a[ok], d[ok] / norm[ok, None]
        n = np.stack([d[:, 1], -d[:, 0]], 1)
        dist = np.abs((pool[None] - a[:, None]) @ n[..., None])[..., 0] if len(pool) * len(a) < 4e7 else None
        if dist is None:                                 # memory-safe fallback for huge pools
            counts = np.array([(np.abs((pool - aa) @ nn) < ptol).sum() for aa, nn in zip(a, n)])
            best = int(np.argmax(counts))
            inl = np.abs((pool - a[best]) @ n[best]) < ptol
        else:
            best = int(np.argmax((dist < ptol[None]).sum(1)))
            inl = dist[best] < ptol
        if inl.sum() < min_cells:
            break
        p = pool[inl]
        c = p.mean(0)
        direction = np.linalg.svd(p - c, full_matrices=False)[2][0]
        nn = np.array([direction[1], -direction[0]])
        inl = np.abs((pool - c) @ nn) < ptol           # inliers of the refined line
        p = pool[inl]
        t = (p - c) @ direction
        order = np.argsort(t)
        t, p = t[order], p[order]
        for s, e in _line_runs(t, WALL_SPLIT_GAP):
            if t[e - 1] - t[s] >= WALL_MIN_LENGTH and e - s >= min_cells // 2:
                off = float(c @ nn)
                tc = float(c @ direction)
                segs.append(_segment(direction, off, tc + t[s], tc + t[e - 1], e - s, p[s:e]))
        pool, ptol = pool[~inl], ptol[~inl]
    return segs


def _angle(seg: dict) -> float:
    """Direction angle of a segment in degrees, in [0, 180)."""
    return float(np.degrees(np.arctan2(seg["dir"][1], seg["dir"][0])) % 180.0)


def dominant_direction(segs: list[dict]) -> tuple[float, float]:
    """Length-weighted dominant direction modulo 90 deg (mean of 4*angle on the circle) and its support."""
    if not segs:
        return 0.0, 0.0
    ang = np.radians([_angle(s) for s in segs])
    w = np.array([s["length"] for s in segs])
    dom = (np.degrees(np.arctan2((w * np.sin(4 * ang)).sum(), (w * np.cos(4 * ang)).sum())) / 4.0) % 90.0
    dev = np.abs((np.degrees(ang) - dom + 45.0) % 90.0 - 45.0)
    return float(dom), float(w[dev <= MANHATTAN_TOL].sum() / w.sum())


def _refit(seg: dict, direction: np.ndarray) -> dict:
    """Segment with a new direction; offset and extent recomputed from its inlier points."""
    p = seg["inliers"]
    n = np.array([direction[1], -direction[0]])
    off = float(np.mean(p @ n))
    t = p @ direction
    return _segment(direction, off, float(t.min()), float(t.max()), seg["cells"], p)


def manhattan_snap(segs: list[dict]) -> tuple[list[dict], dict]:
    """Snap segments within MANHATTAN_TOL of the dominant direction (or its perpendicular) to it,
    if at least MANHATTAN_MIN_SUPPORT of the total wall length agrees with that direction."""
    dom, support = dominant_direction(segs)
    info = {"dominant_deg": dom, "support": support, "manhattan": support >= MANHATTAN_MIN_SUPPORT}
    if not info["manhattan"]:
        return segs, info
    out = []
    for s in segs:
        a = _angle(s)
        dev = (a - dom + 45.0) % 90.0 - 45.0
        if abs(dev) <= MANHATTAN_TOL:
            t = np.radians(a - dev)
            ns = _refit(s, np.array([np.cos(t), np.sin(t)]))
            ns["snapped"] = True
            out.append(ns)
        else:
            out.append(s)
    return out, info


def merge_collinear(segs: list[dict]) -> list[dict]:
    """Merge segments that lie on (nearly) the same line and nearly touch; repeat until stable."""
    segs = list(segs)
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                da = abs((_angle(a) - _angle(b) + 90.0) % 180.0 - 90.0)
                if da > MERGE_ANGLE:
                    continue
                d = a["dir"] if a["length"] >= b["length"] else b["dir"]
                n = np.array([d[1], -d[0]])
                tol = max(MERGE_OFFSET, MERGE_OFFSET_REL * max(a["offset"], b["offset"]))
                if abs(np.mean(a["inliers"] @ n) - np.mean(b["inliers"] @ n)) > tol:
                    continue
                ta, tb = sorted([a["p0"] @ d, a["p1"] @ d]), sorted([b["p0"] @ d, b["p1"] @ d])
                if max(ta[0], tb[0]) - min(ta[1], tb[1]) > MERGE_GAP:
                    continue
                p = np.concatenate([a["inliers"], b["inliers"]])
                m = _refit({"inliers": p, "cells": a["cells"] + b["cells"]}, d)
                m["snapped"] = a["snapped"] and b["snapped"]
                segs[i] = m
                del segs[j]
                changed = True
                break
            if changed:
                break
    return segs


def snap_corners(segs: list[dict], reach: float = CORNER_REACH) -> list[dict]:
    """Extend / trim pairs of roughly perpendicular walls to their intersection when it lies within
    `reach` of an endpoint of both (turns two loose wall ends into a clean corner)."""
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            da = abs((_angle(a) - _angle(b) + 90.0) % 180.0 - 90.0)
            if da < 60.0:
                continue
            m = np.array([a["dir"], -b["dir"]]).T
            try:
                ta, tb = np.linalg.solve(m, b["p0"] - a["p0"])
            except np.linalg.LinAlgError:
                continue
            x = a["p0"] + ta * a["dir"]
            ea = 0 if np.linalg.norm(x - a["p0"]) < np.linalg.norm(x - a["p1"]) else 1
            eb = 0 if np.linalg.norm(x - b["p0"]) < np.linalg.norm(x - b["p1"]) else 1
            if np.linalg.norm(x - a[f"p{ea}"]) <= reach and np.linalg.norm(x - b[f"p{eb}"]) <= reach:
                a[f"p{ea}"], b[f"p{eb}"] = x.copy(), x.copy()
    for s in segs:
        s["length"] = float(np.linalg.norm(s["p1"] - s["p0"]))
    return segs


def behind_share(seg: dict, evidence: np.ndarray) -> float:
    """Share of the evidence points inside the wall's viewing wedge that lie clearly BEHIND it.

    The camera sees a room boundary as the last surface along each ray, so a real wall cannot have
    floor, wall or furniture points behind it. Rays from the camera (origin) through each point are
    intersected with the wall line; a point is behind if it is more than 5% + 10 cm further away.
    """
    a0, a1 = _azimuth_interval(seg)
    az = np.degrees(np.arctan2(evidence[:, 0], evidence[:, 1]))
    e = evidence[(az >= a0) & (az <= a1)]
    if len(e) < 30:
        return 0.0
    r = np.linalg.norm(e, axis=1)
    den = (e / r[:, None]) @ seg["n"]
    rw = np.where(den > 1e-6, seg["offset"] / np.maximum(den, 1e-6), np.inf)
    return float(np.mean(r > rw * (1 + FREE_SPACE_REL) + FREE_SPACE_ABS))


def free_space_filter(segs: list[dict], evidence: np.ndarray | None) -> list[dict]:
    """Drop wall lines that have more than FREE_SPACE_MAX_BEHIND of the visible scene behind them
    (spurious lines inside the room: furniture sides, noisy bands of a wall seen at a grazing angle)."""
    if evidence is None or not len(evidence):
        return segs
    return [s for s in segs if behind_share(s, evidence) <= FREE_SPACE_MAX_BEHIND]


def drop_shadowed(segs: list[dict], max_angle: float = 15.0, min_cover: float = 0.5) -> list[dict]:
    """Of two nearly parallel lines that cover the same viewing directions (one is a noisy copy of the
    other, e.g. the two sides of a thick band of wall points), keep only the better supported one."""
    keep = sorted(segs, key=lambda s: -s["cells"])
    out = []
    for s in keep:
        sa = _azimuth_interval(s)
        dup = False
        for o in out:
            oa = _azimuth_interval(o)
            inter = min(sa[1], oa[1]) - max(sa[0], oa[0])
            dang = abs((_angle(s) - _angle(o) + 90.0) % 180.0 - 90.0)
            if dang <= max_angle and inter > min_cover * min(sa[1] - sa[0], oa[1] - oa[0]):
                dup = True
                break
        if not dup:
            out.append(s)
    return out


def extract_walls(xy: np.ndarray, grid: Grid, evidence: np.ndarray | None = None) -> tuple[list[dict], dict, np.ndarray]:
    """Wall segments from projected wall points: occupancy -> morphology -> RANSAC lines -> free-space
    check -> Manhattan snap -> merge collinear -> corners. Returns (segments, info, cleaned occupancy)."""
    occ = clean_occupancy(grid.occupancy(xy))
    segs = ransac_lines(grid.centres(occ))
    segs = free_space_filter(segs, evidence)
    segs = drop_shadowed(segs)
    segs, info = manhattan_snap(segs)
    segs = merge_collinear(segs)
    segs = [s for s in segs if s["length"] >= WALL_MIN_LENGTH]
    segs = snap_corners(segs)
    return segs, info, occ


def closed_room(segs: list[dict], info: dict) -> dict | None:
    """Closed rectangular room if (Manhattan) walls are seen on all four sides of the camera.

    Sides are found in the dominant frame; the longest wall on each side is taken as its boundary.
    Returns {"corners" (4x2), "area", "width", "depth"} or None if any side is missing.
    """
    if not info.get("manhattan") or not segs:
        return None
    t = np.radians(info["dominant_deg"])
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    sides = {}
    for s in segs:
        if not s["snapped"]:
            continue
        mid = (s["p0"] + s["p1"]) / 2
        axis = "u" if abs(s["n"] @ u) > abs(s["n"] @ v) else "v"
        key = (axis, np.sign(mid @ (u if axis == "u" else v)))
        if key not in sides or s["length"] > sides[key]["length"]:
            sides[key] = s
    if len(sides) < 4:
        return None
    pos = {k: float(((s["p0"] + s["p1"]) / 2) @ (u if k[0] == "u" else v)) for k, s in sides.items()}
    u0, u1, v0, v1 = pos[("u", -1.0)], pos[("u", 1.0)], pos[("v", -1.0)], pos[("v", 1.0)]
    corners = np.array([u0 * u + v0 * v, u1 * u + v0 * v, u1 * u + v1 * v, u0 * u + v1 * v])
    return {"corners": corners, "width": u1 - u0, "depth": v1 - v0, "area": (u1 - u0) * (v1 - v0)}


# --------------------------------------------------------------------------- #
# Furniture and openings
# --------------------------------------------------------------------------- #
def instances(mask: np.ndarray, min_px: int) -> list[np.ndarray]:
    """Split a class mask into instances: opening, then 8-connected components of >= min_px pixels."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def _robust_xy(xy: np.ndarray, cell: float = 0.05) -> np.ndarray:
    """Keep the main body of an instance's plan points.

    1. Range filter: points whose distance from the camera lies outside [q25 - IQR, q75 + IQR] are
       mask-boundary pixels whose depth is smeared towards the background.
    2. Largest blob: the points are rasterised on a 5 cm grid, closed with a 3x3 kernel, and only
       the largest 8-connected component is kept.
    """
    if len(xy) < 20:
        return xy
    r = np.linalg.norm(xy, axis=1)
    q1, q3 = np.percentile(r, [25, 75])
    iqr = max(q3 - q1, 0.05)
    xy = xy[(r >= q1 - iqr) & (r <= q3 + iqr)]
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(int)
    occ = np.zeros(ij.max(0)[::-1] + 1, np.uint8)
    occ[ij[:, 1], ij[:, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    if n <= 2:
        return xy
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return xy[lab[ij[:, 1], ij[:, 0]] == big]


def footprint(xy: np.ndarray, prior_depth: float) -> dict:
    """Minimum-area rectangle of an instance's plan points, plus the prior-completed rectangle."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(xy.astype(np.float32))
    obs = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float64)
    c = np.array([cx, cy])
    t = np.radians(ang)
    ax_w, ax_h = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    short, ax = (w, ax_w) if w < h else (h, ax_h)
    comp, completed = obs, False
    if short < prior_depth:                            # extend the hidden back away from the camera
        sgn = 1.0 if ax @ c >= 0 else -1.0
        c2 = c + sgn * ax * (prior_depth - short) / 2
        size = (prior_depth, h) if w < h else (w, prior_depth)
        comp = cv2.boxPoints(((c2[0], c2[1]), size, ang)).astype(np.float64)
        completed = True
    return {"observed": obs, "rect": comp, "completed": completed, "centroid": comp.mean(0),
            "observed_centroid": obs.mean(0), "size": tuple(sorted(np.linalg.norm(comp[[1, 3]] - comp[0], axis=1)))}


def extract_furniture(pts: np.ndarray, valid: np.ndarray, class_map: np.ndarray, id2label: dict,
                      frame: dict) -> list[dict]:
    """Furniture instances with labels and plan footprints."""
    min_px = int(INSTANCE_MIN_SHARE * class_map.size)
    out = []
    for name, (label, prior) in FURNITURE.items():
        ids = class_ids(id2label, [name])
        if not ids:
            continue
        mask = np.isin(class_map, ids)
        if mask.sum() < min_px:
            continue
        for k, inst in enumerate(instances(mask, min_px)):
            m = cv2.erode(inst.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=INSTANCE_ERODE) > 0
            m &= valid
            if m.sum() < min_px // 2:
                continue
            xy, hgt = to_plan(pts[m], frame)
            xy = _robust_xy(xy[(hgt > 0.02) & (hgt < 2.6)])
            if len(xy) < 20:
                continue
            f = footprint(xy, prior)
            f.update({"label": label, "ade": name, "prior": prior, "xy": xy,
                      "parts": {f"{name}#{k}": int(inst.sum())}})
            out.append(f)
    return merge_instances(out)


def _overlap_of_smaller(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area of two plan polygons divided by the smaller area (rasterised, 2 cm)."""
    iou = polygon_iou(a, b, res=0.02)
    if iou <= 0:
        return 0.0
    aa, ab = cv2.contourArea(a.astype(np.float32)), cv2.contourArea(b.astype(np.float32))
    inter = iou * (aa + ab) / (1 + iou)
    return float(inter / max(min(aa, ab), 1e-9))


def merge_instances(furn: list[dict], min_overlap: float = 0.2) -> list[dict]:
    """Merge instances of the same label whose observed footprints overlap by at least `min_overlap`
    of the smaller one (one object split in the image by an occluder, e.g. a lamp on a chest of
    drawers); the merged footprint is refitted on the union of the points."""
    furn = list(furn)
    changed = True
    while changed:
        changed = False
        for i in range(len(furn)):
            for j in range(i + 1, len(furn)):
                a, b = furn[i], furn[j]
                if a["label"] != b["label"] or _overlap_of_smaller(a["observed"], b["observed"]) < min_overlap:
                    continue
                xy = np.vstack([a["xy"], b["xy"]])
                m = footprint(xy, max(a["prior"], b["prior"]))
                m.update({"label": a["label"], "ade": a["ade"], "prior": max(a["prior"], b["prior"]), "xy": xy,
                          "parts": {**a["parts"], **b["parts"]}})
                furn[i] = m
                del furn[j]
                changed = True
                break
            if changed:
                break
    return furn


def extract_openings(pts: np.ndarray, valid: np.ndarray, class_map: np.ndarray, id2label: dict, frame: dict,
                     walls: list[dict]) -> list[dict]:
    """Doors and windows: each instance is projected onto the wall line that most of its points are
    close to; its extent along that wall becomes the opening [t0, t1] (metres along the wall)."""
    if not walls:
        return []
    min_px = int(INSTANCE_MIN_SHARE * class_map.size)
    out = []
    for name, kind in OPENINGS.items():
        ids = class_ids(id2label, [name])
        mask = np.isin(class_map, ids) if ids else np.zeros(class_map.shape, bool)
        for inst in instances(mask, min_px):
            m = (cv2.erode(inst.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & valid
            if m.sum() < 50:
                continue
            xy, _ = to_plan(pts[m], frame)
            best, best_n = None, 0
            for wi, w in enumerate(walls):
                t = (xy - w["p0"]) @ w["dir"]
                near = (np.abs(xy @ w["n"] - w["offset"]) < OPENING_MAX_DIST) & (t > -0.3) & (t < w["length"] + 0.3)
                if near.sum() > best_n:
                    best, best_n = wi, int(near.sum())
            if best is None or best_n < 0.3 * len(xy):
                continue
            w = walls[best]
            t = (xy - w["p0"]) @ w["dir"]
            near = np.abs(xy @ w["n"] - w["offset"]) < OPENING_MAX_DIST
            t0, t1 = np.percentile(t[near], [3, 97])
            t0, t1 = max(0.0, t0), min(w["length"], t1)
            if t1 - t0 >= OPENING_MIN_WIDTH:
                out.append({"kind": kind, "wall": best, "t0": float(t0), "t1": float(t1), "width": float(t1 - t0)})
    return out


# --------------------------------------------------------------------------- #
# Whole plan
# --------------------------------------------------------------------------- #
def build_floorplan(depth: np.ndarray, k: tuple, class_map: np.ndarray, id2label: dict,
                    valid: np.ndarray | None = None, fov_deg: float | None = None) -> dict:
    """Floor plan of one view.

    depth: HxW metres; k = (fx, fy, cx, cy); class_map: HxW ADE20K ids at the depth resolution;
    valid: optional HxW bool mask of usable depth pixels.
    Returns walls, furniture, openings, room (closed polygon or None), floor occupancy, frame and stats.
    """
    valid = np.isfinite(depth) & (depth > 0.05) if valid is None else valid & np.isfinite(depth) & (depth > 0.05)
    pts = backproject(depth, k)
    sel = lambda names: np.isin(class_map, class_ids(id2label, names))  # noqa: E731
    floor_m, wall_m, ceil_m = sel(FLOOR_CLASSES), sel(WALL_CLASSES), sel(CEILING_CLASSES)
    frame = gravity_frame(pts, valid, floor_m, ceil_m)

    xy_all, h_all = to_plan(pts[valid], frame)
    grid = Grid(np.vstack([xy_all, [[0.0, 0.0]]]))

    wm = (cv2.erode(wall_m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & valid
    wxy, wh = to_plan(pts[wm], frame)
    em = sel(FLOOR_CLASSES | WALL_CLASSES | CEILING_CLASSES | set(FURNITURE)) & valid    # free-space evidence
    exy = to_plan(pts[em], frame)[0]
    if len(exy) > 30_000:
        exy = exy[np.random.default_rng(0).choice(len(exy), 30_000, replace=False)]
    walls, info, wall_occ = (extract_walls(wxy[wh > WALL_MIN_HEIGHT], grid, exy) if wm.sum() > 50
                             else ([], {"manhattan": False}, None))

    fm = floor_m & valid
    fxy, _ = to_plan(pts[fm], frame)
    floor_occ = clean_occupancy(grid.occupancy(fxy), close=15, open_=3) if len(fxy) else np.zeros((grid.h, grid.w), np.uint8)

    ceiling_h = None
    cm = ceil_m & valid
    if cm.sum() > 500:
        ceiling_h = float(np.median(to_plan(pts[cm], frame)[1]))

    furniture = extract_furniture(pts, valid, class_map, id2label, frame)
    openings = extract_openings(pts, valid, class_map, id2label, frame, walls)
    room = closed_room(walls, info)
    if fov_deg is None:
        fov_deg = float(np.degrees(2 * np.arctan(class_map.shape[1] / 2 / k[0])))
    return {"walls": walls, "furniture": furniture, "openings": openings, "room": room, "manhattan": info,
            "frame": frame, "grid": grid, "floor_occ": floor_occ, "wall_occ": wall_occ,
            "visible_floor_m2": float(floor_occ.sum() * grid.cell ** 2), "camera_height": frame["d"],
            "ceiling_height": ceiling_h, "fov_deg": fov_deg,
            "bounds": (xy_all.min(0), xy_all.max(0))}


def plan_summary(plan: dict) -> dict:
    """Flat numbers describing a plan (for tables)."""
    r = plan["room"]
    return {"walls": len(plan["walls"]), "wall_length_m": sum(w["length"] for w in plan["walls"]),
            "furniture": len(plan["furniture"]), "doors": sum(o["kind"] == "door" for o in plan["openings"]),
            "windows": sum(o["kind"] == "window" for o in plan["openings"]),
            "manhattan": bool(plan["manhattan"].get("manhattan")), "closed": r is not None,
            "area_m2": r["area"] if r else np.nan, "visible_floor_m2": plan["visible_floor_m2"],
            "camera_height_m": plan["camera_height"], "ceiling_height_m": plan["ceiling_height"] or np.nan,
            "gravity": plan["frame"]["source"]}


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
INK = "#1b1b1b"
FLOOR_TINT = (0.955, 0.945, 0.925)
FURN_FACE = "#ffffff"
FURN_EDGE = "#4a4a4a"
DIM_COLOR = "#2d5f8a"


def _readable(deg: float) -> float:
    """Text rotation in (-90, 90] so labels never read upside down."""
    deg = (deg + 180.0) % 360.0 - 180.0
    if deg > 90:
        deg -= 180
    elif deg <= -90:
        deg += 180
    return deg


def _dimension(ax, p0, p1, out_n, offset: float, text: str, fs: float):
    """Architect-style dimension line parallel to p0-p1, `offset` m along out_n, with slash ticks."""
    a, b = p0 + out_n * offset, p1 + out_n * offset
    for p, q in ((p0, a), (p1, b)):                         # extension lines
        s = p + out_n * 0.06
        ax.plot([s[0], q[0] + out_n[0] * 0.06], [s[1], q[1] + out_n[1] * 0.06], color=DIM_COLOR, lw=0.5)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=DIM_COLOR, lw=0.7)
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    tick = (d + out_n) / np.sqrt(2) * 0.07
    for q in (a, b):
        ax.plot([q[0] - tick[0], q[0] + tick[0]], [q[1] - tick[1], q[1] + tick[1]], color=DIM_COLOR, lw=1.1)
    mid = (a + b) / 2 + out_n * 0.09
    ang = _readable(np.degrees(np.arctan2(d[1], d[0])))
    ax.text(mid[0], mid[1], text, color=DIM_COLOR, fontsize=fs, ha="center", va="center", rotation=ang,
            rotation_mode="anchor", bbox=dict(fc="white", ec="none", pad=0.6))


def draw_floorplan(plan: dict, png_path=None, svg_path=None, title: str | None = None, dpi: int = 150,
                   note: str | None = "All dimensions approximate (monocular depth; scale error ~20%)",
                   ax=None) -> np.ndarray | None:
    """Draw the plan. Saves PNG / SVG if paths are given; returns the RGB image when it creates its own figure."""
    import matplotlib
    if ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Polygon, Rectangle

    walls, room = plan["walls"], plan["room"]
    pts = [np.zeros((1, 2))] + [np.array([w["p0"], w["p1"]]) for w in walls] + [f["rect"] for f in plan["furniture"]]
    if room is not None:
        pts.append(room["corners"])
    allp = np.vstack(pts)
    lo, hi = allp.min(0) - 0.9, allp.max(0) + 0.9
    span = hi - lo
    own = ax is None
    if own:
        scale = 7.0 / max(span.max(), 1e-6)
        fig = plt.figure(figsize=(max(4.0, span[0] * scale), max(4.0, span[1] * scale + 0.6)), dpi=dpi)
        ax = fig.add_axes([0.02, 0.07, 0.96, 0.86])
    fs = 7.0
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")

    # visible floor (light tint) and closed room outline
    g = plan["grid"]
    fo = plan["floor_occ"].astype(bool)
    if fo.any():
        rgba = np.zeros((*fo.shape, 4))
        rgba[fo] = (*FLOOR_TINT, 1.0)
        ax.imshow(rgba, origin="lower", extent=(g.x0, g.x0 + g.w * g.cell, g.y0, g.y0 + g.h * g.cell),
                  interpolation="nearest", zorder=0)
    if room is not None:
        ax.add_patch(Polygon(room["corners"], closed=True, fill=False, ec=INK, lw=0.8, ls=(0, (4, 3)), zorder=1))

    # furniture
    for f in plan["furniture"]:
        ax.add_patch(Polygon(f["rect"], closed=True, fc=FURN_FACE, ec=FURN_EDGE, lw=0.9,
                             ls=(0, (3, 2)) if f["completed"] else "-", zorder=3))
        if f["completed"]:
            ax.add_patch(Polygon(f["observed"], closed=True, fill=False, ec=FURN_EDGE, lw=0.9, zorder=3))
        c = f["centroid"]
        ax.text(c[0], c[1], f["label"], fontsize=fs - 1, ha="center", va="center", color="#333333", zorder=4)

    # walls (filled, thickness drawn outwards) with openings cut out
    for wi, w in enumerate(walls):
        poly = np.array([w["p0"], w["p1"], w["p1"] + w["n"] * WALL_THICKNESS, w["p0"] + w["n"] * WALL_THICKNESS])
        ax.add_patch(Polygon(poly, closed=True, fc=INK, ec=INK, lw=0.5, zorder=5))
    for o in plan["openings"]:
        w = walls[o["wall"]]
        a, b = w["p0"] + w["dir"] * o["t0"], w["p0"] + w["dir"] * o["t1"]
        n = w["n"]
        gap = np.array([a - n * 0.01, b - n * 0.01, b + n * (WALL_THICKNESS + 0.01), a + n * (WALL_THICKNESS + 0.01)])
        ax.add_patch(Polygon(gap, closed=True, fc="white", ec="none", zorder=6))
        if o["kind"] == "window":
            for frac in (0.0, 0.5, 1.0):
                s = n * WALL_THICKNESS * frac
                ax.plot([a[0] + s[0], b[0] + s[0]], [a[1] + s[1], b[1] + s[1]], color=INK, lw=0.7, zorder=7)
            for q in (a, b):
                ax.plot([q[0], q[0] + n[0] * WALL_THICKNESS], [q[1], q[1] + n[1] * WALL_THICKNESS], color=INK, lw=0.9, zorder=7)
        else:                                                      # door: leaf + quarter-circle swing into the room
            width = o["width"]
            leaf = a - n * width
            ax.plot([a[0], leaf[0]], [a[1], leaf[1]], color=INK, lw=1.0, zorder=7)
            t_dir = np.degrees(np.arctan2(w["dir"][1], w["dir"][0]))
            t_in = np.degrees(np.arctan2(-n[1], -n[0]))
            th1, th2 = sorted([t_dir, t_in])
            if th2 - th1 > 180:
                th1, th2 = th2, th1 + 360
            ax.add_patch(Arc(a, 2 * width, 2 * width, theta1=th1, theta2=th2, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=7))

    # dimension lines: closed room -> its sides; otherwise every wall of at least 0.5 m
    if room is not None:
        cs = room["corners"]
        centre = cs.mean(0)
        for i in range(4):
            p0, p1 = cs[i], cs[(i + 1) % 4]
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            n = np.array([d[1], -d[0]])
            if n @ ((p0 + p1) / 2 - centre) < 0:
                n = -n
            _dimension(ax, p0, p1, n, WALL_THICKNESS + 0.28, f"≈{np.linalg.norm(p1 - p0):.2f} m", fs)
    else:
        for w in walls:
            if w["length"] >= 0.5:
                _dimension(ax, w["p0"], w["p1"], w["n"], WALL_THICKNESS + 0.28, f"≈{w['length']:.2f} m", fs)

    # camera marker with its field of view
    half = np.radians(plan["fov_deg"] / 2)
    r = 0.9
    for s in (-1, 1):
        ax.plot([0, r * np.sin(s * half)], [0, r * np.cos(half)], color="#c0392b", lw=0.6, ls=(0, (3, 2)), zorder=8)
    tri = np.array([[0, 0.16], [-0.1, -0.08], [0.1, -0.08]])
    ax.add_patch(Polygon(tri, closed=True, fc="#c0392b", ec="none", zorder=9))
    ax.text(0, -0.22, "camera", fontsize=fs - 1, ha="center", va="top", color="#c0392b", zorder=9)

    # scale bar (1 m) in the lower left corner
    x0, y0 = lo[0] + 0.25, lo[1] + 0.25
    ax.add_patch(Rectangle((x0, y0), 0.5, 0.06, fc=INK, ec=INK, lw=0.6, zorder=9))
    ax.add_patch(Rectangle((x0 + 0.5, y0), 0.5, 0.06, fc="white", ec=INK, lw=0.6, zorder=9))
    ax.text(x0, y0 + 0.12, "0", fontsize=fs - 1, ha="center", va="bottom")
    ax.text(x0 + 1.0, y0 + 0.12, "1 m", fontsize=fs - 1, ha="center", va="bottom")

    # header
    if room is not None:
        area = f"Floor area ≈ {room['area']:.1f} m²  ({room['width']:.2f} × {room['depth']:.2f} m)"
    else:
        area = f"Visible floor ≈ {plan['visible_floor_m2']:.1f} m²  (room not closed in this view)"
    head = (title + "\n" if title else "") + area
    if own:
        fig.text(0.5, 0.985, head, ha="center", va="top", fontsize=fs + 2)
        if note:
            fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=fs - 0.5, color="#666666", style="italic")
        if png_path is not None:
            fig.savefig(png_path, dpi=dpi, facecolor="white")
        if svg_path is not None:
            fig.savefig(svg_path, facecolor="white")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
    ax.set_title(head, fontsize=fs + 1)
    return None


# --------------------------------------------------------------------------- #
# Comparison of two plans of the same view (validation against ground truth)
# --------------------------------------------------------------------------- #
def _azimuth_interval(w: dict) -> tuple[float, float]:
    """Angular extent (deg) of a wall as seen from the camera; angle 0 = straight ahead."""
    a = [np.degrees(np.arctan2(p[0], p[1])) for p in (w["p0"], w["p1"])]
    return min(a), max(a)


def match_walls(gt: list[dict], pred: list[dict], min_iou: float = 0.3, max_angle: float = 20.0) -> list[tuple]:
    """Greedy one-to-one matching of wall segments by the overlap of their azimuth intervals.

    The azimuth extent comes from which image columns the wall occupies, so it does not depend on the
    depth scale, and a wall in the predicted plan is matched to the same wall in the ground-truth plan
    even when the predicted room is too large or too small. Pairs must also agree in direction.
    """
    cand = []
    for i, g in enumerate(gt):
        ga = _azimuth_interval(g)
        for j, p in enumerate(pred):
            pa = _azimuth_interval(p)
            inter = min(ga[1], pa[1]) - max(ga[0], pa[0])
            union = max(ga[1], pa[1]) - min(ga[0], pa[0])
            iou = inter / union if union > 0 else 0.0
            dang = abs((_angle(g) - _angle(p) + 90.0) % 180.0 - 90.0)
            if iou >= min_iou and dang <= max_angle:
                cand.append((iou, i, j))
    used_g, used_p, out = set(), set(), []
    for iou, i, j in sorted(cand, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j, iou))
    return out


def polygon_iou(a: np.ndarray, b: np.ndarray, res: float = 0.01) -> float:
    """IoU of two polygons (plan metres) by rasterising both on a 1 cm grid."""
    lo = np.minimum(a.min(0), b.min(0)) - res
    hi = np.maximum(a.max(0), b.max(0)) + res
    shape = tuple((np.ceil((hi - lo) / res)).astype(int)[::-1] + 1)
    if shape[0] * shape[1] > 4e7:
        return 0.0
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round((a - lo) / res).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round((b - lo) / res).astype(np.int32)], 1)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def match_furniture(gt: list[dict], pred: list[dict]) -> list[tuple[int, int]]:
    """One-to-one matching of furniture by shared image instances (both plans are built from the same
    segmentation, so an object is the same set of pixel instances in both; merging can differ)."""
    cand = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            shared = sum(min(g["parts"][k], p["parts"][k]) for k in set(g["parts"]) & set(p["parts"]))
            if shared:
                cand.append((shared, i, j))
    ug, up, out = set(), set(), []
    for _, i, j in sorted(cand, reverse=True):
        if i not in ug and j not in up:
            ug.add(i)
            up.add(j)
            out.append((i, j))
    return out


def compare_plans(gt: dict, pred: dict) -> dict:
    """Wall-length error, furniture-footprint IoU and centroid error of `pred` against `gt`."""
    out = {"gt_walls": len(gt["walls"]), "pred_walls": len(pred["walls"])}
    m = match_walls(gt["walls"], pred["walls"])
    out["matched_walls"] = len(m)
    if m:
        lg = np.array([gt["walls"][i]["length"] for i, _, _ in m])
        lp = np.array([pred["walls"][j]["length"] for _, j, _ in m])
        og = np.array([gt["walls"][i]["offset"] for i, _, _ in m])
        op = np.array([pred["walls"][j]["offset"] for _, j, _ in m])
        out.update({"wall_len_abs_err_m": float(np.mean(np.abs(lp - lg))),
                    "wall_len_rel_err": float(np.mean(np.abs(lp - lg) / lg)),
                    "wall_len_ratio": float(np.median(lp / lg)),
                    "wall_dist_rel_err": float(np.mean(np.abs(op - og) / np.maximum(og, 0.1)))})
    pairs = match_furniture(gt["furniture"], pred["furniture"])
    out.update({"gt_furniture": len(gt["furniture"]), "pred_furniture": len(pred["furniture"]),
                "matched_furniture": len(pairs)})
    if pairs:
        g = [gt["furniture"][i] for i, _ in pairs]
        p = [pred["furniture"][j] for _, j in pairs]
        err = [np.linalg.norm(a["centroid"] - b["centroid"]) for a, b in zip(g, p)]
        out["furn_iou"] = float(np.mean([polygon_iou(a["rect"], b["rect"]) for a, b in zip(g, p)]))
        out["furn_iou_observed"] = float(np.mean([polygon_iou(a["observed"], b["observed"]) for a, b in zip(g, p)]))
        out["furn_centroid_err_m"] = float(np.mean(err))
        out["furn_centroid_rel_err"] = float(np.mean([e / max(np.linalg.norm(a["centroid"]), 0.3) for e, a in zip(err, g)]))
    ug, up = gt["frame"]["up"], pred["frame"]["up"]
    out["gravity_angle_deg"] = float(np.degrees(np.arccos(np.clip(ug @ up, -1, 1))))
    out["gt_closed"], out["pred_closed"] = gt["room"] is not None, pred["room"] is not None
    return out
