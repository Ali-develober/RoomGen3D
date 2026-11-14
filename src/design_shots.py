"""Readable 2D design shots from the 360° panorama (PLAN_V4 milestone 1).

The raw equirectangular panorama is distorted and hard to read, so the main 2D output of the 360° mode
becomes a set of normal-looking interior photos cut out of it:

    room box (src/room_box.fit_box)  ->  gravity frame A (R) and wall-aligned frame B (Q)
    level pinhole camera at the panorama centre (= eye height, the DAP camera height, ~1.45 m),
    roll 0 and pitch 0 in frame B (vertical lines stay vertical, horizon level), 4:3, ~1024 x 768
        wall views A-D   facing the four walls of the box (90° horizontal FOV: together they cover the horizon)
        hero view(s)     yaw chosen automatically (score_view): many and varied furniture instances fully in
                         view, furniture and wall filling the frame, little ceiling / floor; a second hero is
                         kept when it is at least HERO2_MIN_GAP away and scores >= HERO2_MIN_RATIO of the first
    each shot = rays of the view -> frame B -> pano camera frame -> equirectangular (u, v) -> cv2.remap
                (wrap-aware), so the shot is an exact perspective re-projection (no generation, no parallax)

Directions:  view camera (X right, Y down, Z forward) --V--> frame B --Q^T--> frame A --R^T--> pano camera
             as row vectors: d_cam = d_view @ V^T @ Q^T @ R      (A = cam @ R^T, B = A @ Q)
Yaw in frame B: 0 = facing +z, 90 = facing +x (turning right).
"""

from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np

from . import equirect as eq

SHOT_W, SHOT_H = 1024, 768
WALL_FOV = 90.0            # horizontal FOV of the wall views (4 x 90 = the full horizon)
HERO_FOV = 80.0            # horizontal FOV of the hero view(s)
HERO_STEP = 5.0            # yaw candidates every HERO_STEP degrees
HERO2_MIN_GAP = 60.0       # degrees between the two hero views
HERO2_MIN_RATIO = 0.8      # the second hero is kept only if it scores >= this share of the first
SCORE_W, SCORE_H = 256, 192
MIN_COMPLETE = 0.5         # an instance is "in view" if >= half of it (solid angle) lies inside the frame
MIN_FRAME = 0.003          # ... and it covers >= 0.3% of the frame
# main piece of a room (what an interior photo is built around)
ANCHOR = {"bed", "sofa", "table", "dining table", "desk", "bathtub", "kitchen island", "island", "counter",
          "countertop", "pool table", "coffee table"}
DEPTH_CAP = 4.0            # m: the depth term saturates here
# score weights; DEFAULT_WEIGHTS were fixed before any tuning; the sweep (scripts/eval_design_shots.py) picks
# weights on the development bench rooms and stores them in results/tables/shots_weights_selected.json
DEFAULT_WEIGHTS = {"inst": 1.0, "fam": 0.5, "furn": 4.0, "wall": 1.0, "cf": 2.0, "cf_free": 0.45,
                   "anchor": 0.0, "anchor_share": 0.0, "depth": 0.0}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def view_to_B(yaw_deg: float) -> np.ndarray:
    """3x3 V with columns (right, down, forward) of a level camera facing yaw (frame B, Y down)."""
    t = np.radians(yaw_deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])


def view_to_cam(box: dict, yaw_deg: float) -> np.ndarray:
    """3x3 M: d_cam = d_view @ M (row vectors) for a level view at yaw_deg in the box's frame B."""
    return view_to_B(yaw_deg).T @ box["Q"].T @ box["R"]


def view_rays(w: int, h: int, hfov_deg: float) -> tuple[np.ndarray, float]:
    """h x w x 3 pinhole rays (z = 1) with square pixels; returns (rays, vertical FOV in degrees)."""
    f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    vfov = 2 * np.degrees(np.arctan(0.5 * h / f))
    return np.stack([u / f, v / f, np.ones_like(u)], -1), vfov


def render_view(pano: np.ndarray, M: np.ndarray, w: int = SHOT_W, h: int = SHOT_H, hfov_deg: float = WALL_FOV,
                interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Perspective view (h x w) of an equirectangular image (any dtype / channels) for the view rotation M.
    Use interp=cv2.INTER_NEAREST for label maps."""
    H, W = pano.shape[:2]
    r, _ = view_rays(w, h, hfov_deg)
    d = r @ M
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    u, v = eq.dirs_to_uv(d, W, H)
    pad = 4
    ext = cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return cv2.remap(ext, (u + pad).astype(np.float32), v.astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)


@lru_cache(maxsize=4)
def _pano_dirs(W: int, H: int) -> np.ndarray:
    return eq.pixel_dirs(W, H)


def in_frustum(H: int, W: int, M: np.ndarray, hfov_deg: float, aspect: float = SHOT_H / SHOT_W) -> np.ndarray:
    """H x W bool: which panorama pixels are inside the view frustum of rotation M."""
    dv = _pano_dirs(W, H) @ M.T                         # pano camera -> view camera
    th = np.tan(np.radians(hfov_deg) / 2)
    z = dv[..., 2]
    return (z > 1e-6) & (np.abs(dv[..., 0]) <= th * z) & (np.abs(dv[..., 1]) <= th * aspect * z)


def wall_yaws(box: dict) -> dict:
    """Yaw (frame B) of the four wall views, keyed by the letter shown on the plan: A faces the top of the
    plan drawing, then clockwise B (right), C (bottom), D (left)."""
    from .plan_v3 import _to_plan
    (x0, x1), (z0, z1) = box["x"], box["z"]
    rot = (z1 - z0) > (x1 - x0)
    out = {}
    for yaw in (0.0, 90.0, 180.0, 270.0):
        t = np.radians(yaw)
        d = _to_plan(np.array([np.sin(t), np.cos(t)]), rot)  # plan (x, z) -> drawing (x, y)
        ang = np.degrees(np.arctan2(d[0], d[1])) % 360       # clockwise from "up" on the drawing
        out["ABCD"[int(round(ang / 90)) % 4]] = yaw
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Hero view selection
# --------------------------------------------------------------------------- #
def class_sets(id2label: dict) -> dict:
    """ADE20K ids of wall / floor / ceiling / furniture (plus door/window counted as wall surface)."""
    from .floorplan import CEILING_CLASSES, FLOOR_CLASSES, FURNITURE, WALL_CLASSES, class_ids
    return {"wall": class_ids(id2label, set(WALL_CLASSES) | {"door", "window", "windowpane", "curtain", "painting"}),
            "floor": class_ids(id2label, FLOOR_CLASSES), "ceiling": class_ids(id2label, CEILING_CLASSES),
            "furn": class_ids(id2label, set(FURNITURE))}


def view_measures(class_map: np.ndarray, inst_map: np.ndarray, families: dict, M: np.ndarray, hfov: float,
                  sets: dict, w: int = SCORE_W, h: int = SCORE_H, dist: np.ndarray | None = None) -> dict:
    """What a view at rotation M shows, measured on label maps (pano grid):
        frame shares of furniture / wall / ceiling / floor (nearest-neighbour rendered labels),
        instances in view: >= MIN_COMPLETE of the instance's solid angle inside the frustum and
        >= MIN_FRAME of the frame, and the number of distinct families among them,
        anchor: 1 if a main piece (ANCHOR class) is among them, anchor_share: its frame share,
        depth: median distance (m) over the frame (when dist is given; else 0).
    inst_map: -1 = none, else instance index; families: index -> family name."""
    # label maps as float32 (cv2.remap has no int32 path; exact for small integers), nearest neighbour
    cmv = render_view(class_map.astype(np.float32), M, w, h, hfov, cv2.INTER_NEAREST).astype(np.int32)
    imv = render_view(inst_map.astype(np.float32), M, w, h, hfov, cv2.INTER_NEAREST).astype(np.int32)
    share = {k: float(np.isin(cmv, ids).mean()) for k, ids in sets.items()}
    H, W = inst_map.shape
    inside = in_frustum(H, W, M, hfov, h / w)
    sa = np.cos(np.pi / 2 - np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    ids = np.unique(inst_map[inst_map >= 0])
    tot = np.bincount(inst_map[inst_map >= 0], sa[inst_map >= 0], minlength=int(ids.max()) + 1 if len(ids) else 1)
    sel = (inst_map >= 0) & inside
    vis = np.bincount(inst_map[sel], sa[sel], minlength=len(tot))
    frame = np.bincount(imv[imv >= 0], minlength=len(tot)) / imv.size
    full = [int(i) for i in ids if tot[i] > 0 and vis[i] / tot[i] >= MIN_COMPLETE and frame[i] >= MIN_FRAME]
    fams = {families.get(i, str(i)) for i in full}
    anc = [i for i in full if str(families.get(i, "")).strip() in ANCHOR]
    depth = 0.0
    if dist is not None:
        depth = float(np.median(render_view(dist.astype(np.float32), M, w // 4, h // 4, hfov, cv2.INTER_NEAREST)))
    return {"inst": len(full), "fam": len(fams), "furn": share["furn"], "wall": share["wall"],
            "ceiling": share["ceiling"], "floor": share["floor"], "anchor": float(len(anc) > 0),
            "anchor_share": float(max((frame[i] for i in anc), default=0.0)), "depth": depth, "inst_ids": full}


def score_view(m: dict, weights: dict | None = None) -> float:
    """Hero score: instances + variety + furniture / wall share - ceiling and floor beyond a free share
    + main piece in view (and its frame share) + scene depth. Works on scalars or on arrays of candidates."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    cf = np.maximum(0.0, m["ceiling"] + m["floor"] - w["cf_free"])
    return (w["inst"] * m["inst"] + w["fam"] * m["fam"] + w["furn"] * m["furn"] + w["wall"] * m["wall"] - w["cf"] * cf
            + w["anchor"] * m.get("anchor", 0.0) + w["anchor_share"] * m.get("anchor_share", 0.0)
            + w["depth"] * np.minimum(m.get("depth", 0.0), DEPTH_CAP))


def scan_yaws(box: dict, class_map: np.ndarray, inst_map: np.ndarray, families: dict, sets: dict,
              hfov: float = HERO_FOV, step: float = HERO_STEP, dist: np.ndarray | None = None) -> list[dict]:
    """view_measures for every candidate yaw (0, step, ... < 360) in frame B."""
    return [{"yaw": float(y), **view_measures(class_map, inst_map, families, view_to_cam(box, y), hfov, sets,
                                              dist=dist)} for y in np.arange(0.0, 360.0, step)]


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def pick_heroes(scan: list[dict], weights: dict | None = None, n_max: int = 2) -> list[dict]:
    """Best yaw; optionally a second one >= HERO2_MIN_GAP away scoring >= HERO2_MIN_RATIO of the best."""
    sc = sorted(({**m, "score": score_view(m, weights)} for m in scan), key=lambda m: -m["score"])
    out = [sc[0]]
    if n_max > 1:
        for m in sc[1:]:
            if _gap(m["yaw"], out[0]["yaw"]) >= HERO2_MIN_GAP:
                if m["score"] >= HERO2_MIN_RATIO * out[0]["score"] and out[0]["score"] > 0:
                    out.append(m)
                break
    return out


# --------------------------------------------------------------------------- #
# Whole step
# --------------------------------------------------------------------------- #
def design_shots(texture: np.ndarray, box: dict, class_map: np.ndarray, inst_map: np.ndarray, instances: list[dict],
                 id2label: dict, weights: dict | None = None, size: tuple[int, int] = (SHOT_W, SHOT_H),
                 n_hero: int = 2, dist: np.ndarray | None = None) -> dict:
    """Hero view(s) + four wall views from the (highest-resolution) panorama texture.
    Returns {"shots": [{"letter", "kind", "yaw", "fov", "image", "measures"}...] (hero first), "scan", "seconds"}."""
    t0 = time.perf_counter()
    fam = {int(i["id"]): i.get("family", i.get("label", "")) for i in instances}
    sets = class_sets(id2label)
    scan = scan_yaws(box, class_map, inst_map, fam, sets, dist=dist)
    heroes = pick_heroes(scan, selected_weights() if weights is None else weights, n_hero)
    shots = []
    for k, hm in enumerate(heroes):
        M = view_to_cam(box, hm["yaw"])
        shots.append({"letter": "H" if k == 0 else f"H{k + 1}", "kind": "hero", "yaw": hm["yaw"], "fov": HERO_FOV,
                      "image": render_view(texture, M, *size, HERO_FOV), "measures": hm})
    for letter, yaw in wall_yaws(box).items():
        M = view_to_cam(box, yaw)
        m = view_measures(class_map, inst_map, fam, M, WALL_FOV, sets, dist=dist)
        shots.append({"letter": letter, "kind": "wall", "yaw": yaw, "fov": WALL_FOV,
                      "image": render_view(texture, M, *size, WALL_FOV), "measures": m})
    return {"shots": shots, "scan": scan, "seconds": time.perf_counter() - t0}


def selected_weights() -> dict:
    """Weights chosen on the development bench rooms (results/tables/shots_weights_selected.json), else defaults."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "results/tables/shots_weights_selected.json"
    try:
        return {**DEFAULT_WEIGHTS, **json.loads(p.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_WEIGHTS)


def caption(shot: dict) -> str:
    if shot["kind"] == "hero":
        return f"{shot['letter']}: hero view ({shot['measures']['inst']} furniture pieces in view)"
    return f"{shot['letter']}: wall view"


def plan_views(shots: list[dict]) -> list[dict]:
    """The view cones to draw on the floor plan (camera at the panorama centre = frame B origin)."""
    return [{"letter": s["letter"], "yaw": s["yaw"], "fov": s["fov"], "hero": s["kind"] == "hero"} for s in shots]
