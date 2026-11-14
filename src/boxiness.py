"""How much a panorama looks like ONE enclosed room (PLAN_V3 milestone 5).

From the DAP distance and a class map, the room box is fitted (src/room_box.fit_box) and three
measures are combined (each 0..1, higher = more box-like):
    wall_fit      share of the wall surface (area-weighted wall pixels) within 10 cm of a box wall
    enclosed      1 - share of the view (solid angle) whose observed distance lies clearly beyond the box
                  (> 10% + 10 cm): openings into corridors, other rooms, outside
    sides         detected sides / 4
    score         mean of the three
Used to pick the best of several generated candidates.
"""

from __future__ import annotations

import numpy as np

from . import equirect as eq
from . import room_box as rb
from . import room_mesh as rm


def box_exit(dirs_B: np.ndarray, box: dict) -> np.ndarray:
    """Distance from the camera (inside the box) to the box surface along each direction (frame B)."""
    lo = np.array([box["x"][0], box["ceil_y"], box["z"][0]])
    hi = np.array([box["x"][1], box["floor_y"], box["z"][1]])
    inv = 1.0 / np.where(np.abs(dirs_B) < 1e-9, 1e-9, dirs_B)
    return np.maximum(lo * inv, hi * inv).min(-1)


def score(dist: np.ndarray, class_map: np.ndarray, id2label: dict, box: dict | None = None) -> dict:
    from .floorplan import WALL_CLASSES, class_ids
    box = box or rb.fit_box(dist, class_map, id2label)
    H, W = dist.shape
    P = rm.aligned_points(dist, box["R"]) @ box["Q"]
    wall = np.isin(class_map, class_ids(id2label, WALL_CLASSES))
    (x0, x1), (z0, z1) = box["x"], box["z"]
    dw = np.minimum.reduce([np.abs(P[..., 0] - x0), np.abs(P[..., 0] - x1), np.abs(P[..., 2] - z0), np.abs(P[..., 2] - z1)])
    area = rb.pixel_area(dist)
    wall_fit = float(area[wall & (dw < 0.10)].sum() / max(area[wall].sum(), 1e-9)) if wall.any() else 0.0
    dirs = (eq.pixel_dirs(W, H) @ box["R"].T) @ box["Q"]
    r_box = box_exit(dirs, box)
    sa = np.cos(np.pi / 2 - np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))   # solid angle weight
    beyond = dist > r_box * 1.10 + 0.10
    enclosed = float(1 - sa[beyond].sum() / sa.sum())
    sides = box["n_detected"] / 4
    return {"wall_fit": wall_fit, "enclosed": enclosed, "sides": sides, "score": (wall_fit + enclosed + sides) / 3,
            "area_m2": box["area"], "beyond_share": 1 - enclosed}
