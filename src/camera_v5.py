"""PLAN_V5 milestone 2: where the panorama camera stands, how big the room is, and which shots are usable.

    nearest_furniture   horizontal distance from the camera to the nearest furniture surface (m): furniture pixels
                        (ADE20K furniture classes) between 5 cm above the floor and 30 cm above the camera, in the
                        wall-aligned frame B; robust minimum = 2nd percentile of their horizontal distances
    candidate_score     boxiness + room-area term + camera-clearance term; a candidate whose camera is inside or
                        touching furniture (d_near < NEAR_REJECT) is rejected unless every candidate is
    view_validity       a level design shot is VALID when the main piece covers MAIN_MIN..MAIN_MAX of the frame,
                        the median depth of the frame is > DEPTH_MIN and content nearer than NEAR_M covers
                        < NEAR_MAX of the frame
"""

from __future__ import annotations

import numpy as np

from . import room_mesh as rm

NEAR_REJECT = 0.5          # m: camera inside / touching furniture
AREA_CAP, NEAR_CAP = 25.0, 1.5
MAIN_MIN, MAIN_MAX = 0.15, 0.50
DEPTH_MIN = 1.5            # m, median depth of a valid shot
NEAR_M, NEAR_MAX = 0.8, 0.15
DEFAULT_SEL = {"w_area": 0.0, "w_near": 0.0}      # v4 = boxiness only; v5 weights chosen on dev (results/tables)


def furniture_mask(class_map: np.ndarray, id2label: dict) -> np.ndarray:
    from .floorplan import FURNITURE, class_ids
    return np.isin(class_map, class_ids(id2label, set(FURNITURE)))


def nearest_furniture(dist: np.ndarray, class_map: np.ndarray, id2label: dict, box: dict, pct: float = 2.0) -> float:
    """Robust horizontal distance (m) from the camera to the nearest furniture surface (inf if no furniture)."""
    if class_map.shape != dist.shape:
        import cv2
        class_map = cv2.resize(class_map.astype(np.int32), dist.shape[::-1], interpolation=cv2.INTER_NEAREST)
    P = rm.aligned_points(dist, box["R"]) @ box["Q"]
    height = box["camera_height"] - P[..., 1]                      # Y down
    m = furniture_mask(class_map, id2label) & (height > 0.05) & (height < box["camera_height"] + 0.3)
    if m.sum() < 50:
        return float("inf")
    r = np.hypot(P[..., 0][m], P[..., 2][m])
    return float(np.percentile(r, pct))


def candidate_score(sc: dict, weights: dict | None = None) -> float:
    """sc: boxiness score dict + 'area_m2' + 'd_near'. Returns the selection score (higher = better)."""
    w = {**DEFAULT_SEL, **(weights or {})}
    a = min(float(sc.get("area_m2", 0.0)), AREA_CAP) / AREA_CAP
    d = min(float(sc.get("d_near", NEAR_CAP)), NEAR_CAP) / NEAR_CAP
    return float(sc["score"] + w["w_area"] * a + w["w_near"] * d)


def pick_candidate(cands: list[dict], weights: dict | None = None) -> int:
    """Index of the best candidate; candidates with d_near < NEAR_REJECT only if all are below it."""
    ok = [i for i, c in enumerate(cands) if c.get("d_near", np.inf) >= NEAR_REJECT] or list(range(len(cands)))
    return max(ok, key=lambda i: candidate_score(cands[i], weights))


def view_validity(main_share: float, depth_view: np.ndarray) -> dict:
    """depth_view: the frame's distance map (m, any size). Returns the three measures and the verdict."""
    med = float(np.median(depth_view))
    near = float((depth_view < NEAR_M).mean())
    ok_main = MAIN_MIN <= main_share <= MAIN_MAX
    ok = bool(ok_main and med > DEPTH_MIN and near < NEAR_MAX)
    # badness = how far from valid (0 = valid); used to pick the least bad view when none is valid
    bad = (max(0.0, MAIN_MIN - main_share) + max(0.0, main_share - MAIN_MAX) + max(0.0, DEPTH_MIN - med) / DEPTH_MIN
           + max(0.0, near - NEAR_MAX))
    return {"main_share": float(main_share), "median_depth": med, "near_share": near, "valid": ok,
            "badness": float(bad), "fail": [k for k, f in (("main piece size", not ok_main),
                                                          ("too close (median depth)", med <= DEPTH_MIN),
                                                          ("near content", near >= NEAR_MAX)) if f]}
