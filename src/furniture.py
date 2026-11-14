"""Solid furniture (PLAN_V3 milestone 3): every furniture instance becomes a closed, textured box.

Per instance of the v3 panoptic segmentation (src/instances.py), in the box frame B of the room
(src/room_box.py: walls along X and Z, Y down, camera at the origin):
    1. points:   the instance mask is eroded (ERODE px) and flying pixels (depth jump > 5% to a
                 neighbour) are dropped before back-projection; only points clearly inside the room
                 (>= 5 cm in front of every wall, above the floor) are kept
    2. trimming: points outside the [TRIM, 100-TRIM] percentiles of their distance from the camera, then
                 of each plan coordinate, are dropped -> footprints are no longer stretched along the
                 viewing ray by mixed boundary depths
    3. footprint: minimum-area rectangle; its orientation is snapped to the wall axes when within
                 SNAP_DEG (real furniture is usually parallel to the walls) and the extent is re-measured
    4. priors:   a table of plausible (long side, short side, height) ranges per class. A dimension within
                 x1.5 of its range is clamped into it (the hidden back grows AWAY from the camera, the
                 visible front stays); an instance with a dimension further outside is rejected
    5. walls:    a back side that ends within WALL_SNAP m of a parallel wall is moved onto the wall
    6. overlaps: duplicates of one family overlapping >= 50% of the smaller are merged; smaller overlaps
                 are removed by shrinking the smaller box along the axis of least penetration;
                 boxes are clipped to the room
    7. solid:    the footprint is extruded from the floor to the instance's top height (hanging classes:
                 from their lowest point up to the ceiling); 5 faces (no bottom), each subdivided into
                 ~10 cm cells. Texture coordinates point into the panorama: a vertex that the camera sees
                 on this object uses its own direction; a hidden vertex borrows the texture of the
                 corresponding point (same face coordinates) of the object's most visible face
"""

from __future__ import annotations

import cv2
import numpy as np

from . import equirect as eq
from . import room_mesh as rm

ERODE = 1                  # } defaults = the dev-set choice of scripts/eval_furniture.py
TRIM = 2.0                 # }
SNAP_DEG = 20.0
WALL_SNAP = 0.30
CELL = 0.10
MIN_POINTS = 60
HANGING = {"lamp": False, "chandelier": True, "light": True}   # True = hangs from the ceiling

# (long side, short side, height) ranges in m
PRIORS = {
    "bed": ((1.9, 2.3), (0.9, 2.0), (0.4, 1.2)), "cradle": ((0.8, 1.2), (0.5, 0.8), (0.6, 1.1)),
    "sofa": ((1.4, 3.0), (0.75, 1.1), (0.6, 1.0)), "armchair": ((0.6, 1.1), (0.6, 1.0), (0.6, 1.1)),
    "chair": ((0.4, 0.65), (0.4, 0.65), (0.75, 1.1)), "swivel chair": ((0.5, 0.75), (0.5, 0.75), (0.8, 1.3)),
    "seat": ((0.4, 0.7), (0.4, 0.7), (0.4, 1.1)), "stool": ((0.3, 0.5), (0.3, 0.5), (0.4, 0.8)),
    "bench": ((0.8, 2.0), (0.3, 0.5), (0.4, 0.5)), "ottoman": ((0.4, 1.2), (0.4, 0.9), (0.35, 0.5)),
    "table": ((0.8, 2.4), (0.6, 1.2), (0.65, 0.8)), "coffee table": ((0.6, 1.4), (0.4, 0.9), (0.3, 0.55)),
    "desk": ((0.9, 1.8), (0.5, 0.8), (0.7, 0.8)), "pool table": ((2.0, 2.8), (1.1, 1.6), (0.75, 0.85)),
    "cabinet": ((0.4, 2.4), (0.3, 0.7), (0.4, 2.2)), "wardrobe": ((0.8, 2.6), (0.5, 0.7), (1.8, 2.4)),
    "chest of drawers": ((0.4, 1.4), (0.35, 0.6), (0.45, 1.3)), "shelf": ((0.6, 2.0), (0.25, 0.45), (0.8, 2.2)),
    "bookcase": ((0.6, 2.0), (0.25, 0.45), (0.8, 2.2)), "buffet": ((1.2, 2.2), (0.4, 0.6), (0.7, 1.0)),
    "counter": ((0.6, 4.0), (0.55, 0.7), (0.85, 0.95)), "countertop": ((0.6, 4.0), (0.55, 0.7), (0.85, 0.95)),
    "kitchen island": ((1.0, 2.5), (0.6, 1.2), (0.85, 0.95)), "sink": ((0.5, 1.2), (0.4, 0.7), (0.8, 0.95)),
    "stove": ((0.5, 1.0), (0.5, 0.7), (0.8, 0.95)), "oven": ((0.5, 0.9), (0.5, 0.7), (0.5, 0.95)),
    "dishwasher": ((0.45, 0.65), (0.55, 0.65), (0.8, 0.9)), "refrigerator": ((0.55, 0.95), (0.55, 0.8), (1.5, 2.0)),
    "washer": ((0.55, 0.65), (0.5, 0.65), (0.8, 0.9)), "bathtub": ((1.4, 1.9), (0.65, 0.9), (0.45, 0.65)),
    "shower": ((0.8, 1.6), (0.8, 1.2), (1.9, 2.3)), "toilet": ((0.55, 0.75), (0.35, 0.5), (0.7, 0.85)),
    "television receiver": ((0.6, 1.8), (0.05, 0.5), (0.4, 1.1)), "lamp": ((0.2, 0.6), (0.2, 0.6), (0.3, 1.8)),
    "plant": ((0.2, 0.8), (0.2, 0.8), (0.3, 2.0)), "pot": ((0.2, 0.8), (0.2, 0.8), (0.3, 2.0)),
    "flower": ((0.15, 0.6), (0.15, 0.6), (0.2, 1.2)), "fireplace": ((0.8, 2.0), (0.3, 0.8), (0.8, 1.6)),
}


def _flying(dist: np.ndarray, rel: float = 0.05) -> np.ndarray:
    """Pixels whose distance differs by more than `rel` from a 4-neighbour (wrap-aware in x)."""
    d = dist
    out = np.zeros(d.shape, bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        n = np.roll(np.roll(d, dy, 0), dx, 1)
        out |= np.abs(n - d) > rel * d
    return out


def _rect_in(xy: np.ndarray, ang: float, trim: float):
    """Robust oriented rectangle of plan points at angle `ang` (rad): centre, (extent along u, along v), u, v."""
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-np.sin(ang), np.cos(ang)])
    a, b = xy @ u, xy @ v
    a0, a1 = np.percentile(a, [trim, 100 - trim])
    b0, b1 = np.percentile(b, [trim, 100 - trim])
    c = (a0 + a1) / 2 * u + (b0 + b1) / 2 * v
    return {"c": c, "e": np.array([a1 - a0, b1 - b0]), "u": u, "v": v}


def _corners(r: dict) -> np.ndarray:
    hu, hv = r["e"][0] / 2 * r["u"], r["e"][1] / 2 * r["v"]
    return np.array([r["c"] - hu - hv, r["c"] + hu - hv, r["c"] + hu + hv, r["c"] - hu + hv])


def _grow_away(r: dict, axis: int, new: float) -> None:
    """Change the extent along `axis` (0 = u, 1 = v) to `new`, keeping the side nearer the camera fixed."""
    ax = r["u"] if axis == 0 else r["v"]
    sgn = 1.0 if ax @ r["c"] >= 0 else -1.0            # the far side is in direction sgn * ax
    r["c"] = r["c"] + sgn * ax * (new - r["e"][axis]) / 2
    r["e"][axis] = new


def apply_prior(r: dict, h: tuple[float, float], name: str) -> str:
    """Clamp (or reject) footprint + height against the class priors. Returns 'ok' / 'clamped' / 'rejected'."""
    if name not in PRIORS:
        return "ok"
    (l0, l1), (s0, s1), (h0, h1) = PRIORS[name]
    long_ax = 0 if r["e"][0] >= r["e"][1] else 1
    rng = {long_ax: (l0, l1), 1 - long_ax: (s0, s1)}
    status = "ok"
    for ax in (0, 1):
        lo, hi = rng[ax]
        x = r["e"][ax]
        if x < lo / 1.5 or x > hi * 1.5:
            return "rejected"
        if not lo <= x <= hi:
            _grow_away(r, ax, float(np.clip(x, lo, hi)))
            status = "clamped"
    hb, ht = h
    hh = ht - hb
    if hh < h0 / 1.5 or hh > h1 * 1.5:
        return "rejected"
    if not h0 <= hh <= h1:
        r["top"] = hb + float(np.clip(hh, h0, h1))
        status = "clamped"
    return status


def instance_solid(P: np.ndarray, hgt: np.ndarray, box: dict, name: str, params: dict) -> dict | None:
    """Oriented box of one instance from its points (N, 3) in frame B and their heights above the floor."""
    (x0, x1), (z0, z1) = box["x"], box["z"]
    m = 0.05
    keep = (P[:, 0] > x0 + m) & (P[:, 0] < x1 - m) & (P[:, 2] > z0 + m) & (P[:, 2] < z1 - m) & (hgt > 0.02)
    P, hgt = P[keep], hgt[keep]
    if len(P) < MIN_POINTS:
        return None
    rr = np.linalg.norm(P[:, [0, 2]], axis=1)                       # trim along the viewing ray first
    lo, hi = np.percentile(rr, [params["trim"], 100 - params["trim"]])
    s = (rr >= lo) & (rr <= hi)
    P, hgt = P[s], hgt[s]
    xy = P[:, [0, 2]]
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang) % (np.pi / 2)
    if a > np.pi / 4:
        a -= np.pi / 2
    snapped = abs(np.degrees(a)) <= params["snap_deg"]
    if snapped:
        a = 0.0
    r = _rect_in(xy, a, params["trim"])
    hang = HANGING.get(name, False)
    hb = float(np.percentile(hgt, params["trim"])) if hang or name == "television receiver" else 0.0
    r["top"] = float(np.percentile(hgt, 100 - params["trim"]))
    if hang:
        r["top"] = box["ceiling_height"]
    status = apply_prior(r, (hb, r["top"]), name) if params.get("priors", True) else "ok"
    if status == "rejected":
        return {"status": "rejected", "label": name}
    if params["wall_snap"] > 0 and snapped:                        # back side onto a nearby wall
        for ax, (w0, w1), k in ((0, (x0, x1), 0), (1, (z0, z1), 2)):
            # rect axis u = +x, v = +z when snapped (angle 0)
            c, e = r["c"][ax], r["e"][ax]
            sgn = 1.0 if c >= 0 else -1.0                          # far side direction
            far = c + sgn * e / 2
            wall = w1 if sgn > 0 else w0
            gap = sgn * (wall - far)
            if 0 < gap < params["wall_snap"]:
                pr = PRIORS.get(name)
                new_e = e + gap - 0.01
                if pr is None or new_e <= max(pr[0][1], pr[1][1]) * 1.2:
                    r["c"][ax] += sgn * (gap - 0.01) / 2
                    r["e"][ax] = new_e
    r.update({"status": status, "label": name, "bottom": hb, "snapped": snapped, "angle": float(a),
              "n_points": int(len(P))})
    return r


def _poly_area(p):
    return float(cv2.contourArea(p.astype(np.float32)))


def _overlap(a: dict, b: dict) -> tuple[float, float]:
    """(IoU, intersection / smaller area) of two footprints."""
    pa, pb = _corners(a).astype(np.float32), _corners(b).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    if inter <= 0:
        return 0.0, 0.0
    A, B = _poly_area(pa), _poly_area(pb)
    return inter / (A + B - inter), inter / max(min(A, B), 1e-9)


def resolve(solids: list[dict], box: dict, family: dict) -> tuple[list[dict], dict]:
    """Merge duplicates, shrink minor overlaps, clip to the room. Returns (solids, counts)."""
    cnt = {"merged": 0, "shrunk": 0, "dropped_outside": 0}
    S = list(solids)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                iou, ovs = _overlap(S[i], S[j])
                if ovs >= 0.5 and family.get(S[i]["label"]) == family.get(S[j]["label"]):
                    bi, si = (i, j) if np.prod(S[i]["e"]) >= np.prod(S[j]["e"]) else (j, i)
                    big, small = S[bi], S[si]
                    pts = np.vstack([_corners(big), _corners(small)])
                    nr = _rect_in(pts, big["angle"], 0.0)
                    big.update({"c": nr["c"], "e": nr["e"], "top": max(big["top"], small["top"])})
                    del S[si]
                    cnt["merged"] += 1
                    changed = True
                    break
            if changed:
                break
    for i in range(len(S)):                                            # minor overlaps: shrink the smaller one
        for j in range(len(S)):
            if i == j or np.prod(S[i]["e"]) > np.prod(S[j]["e"]):
                continue
            a, b = S[i], S[j]
            if _overlap(a, b)[0] <= 0 or not (a["snapped"] and b["snapped"]):
                continue
            pen = []
            for ax in (0, 1):
                lo = max(a["c"][ax] - a["e"][ax] / 2, b["c"][ax] - b["e"][ax] / 2)
                hi = min(a["c"][ax] + a["e"][ax] / 2, b["c"][ax] + b["e"][ax] / 2)
                pen.append(hi - lo)
            ax = int(np.argmin(pen))
            if pen[ax] <= 0 or pen[ax] >= a["e"][ax] * 0.6:
                continue
            sgn = 1.0 if a["c"][ax] < b["c"][ax] else -1.0                 # move a's side facing b away
            a["e"][ax] -= pen[ax] + 0.01
            a["c"][ax] -= sgn * (pen[ax] + 0.01) / 2
            cnt["shrunk"] += 1
    out = []
    (x0, x1), (z0, z1) = box["x"], box["z"]
    for s in S:
        c = _corners(s)
        if not (x0 < s["c"][0] < x1 and z0 < s["c"][1] < z1):
            cnt["dropped_outside"] += 1
            continue
        if s["snapped"]:                                                # clip to the room
            lo = np.maximum(s["c"] - s["e"] / 2, [x0 + 0.01, z0 + 0.01])
            hi = np.minimum(s["c"] + s["e"] / 2, [x1 - 0.01, z1 - 0.01])
            s["c"], s["e"] = (lo + hi) / 2, np.maximum(hi - lo, 0.05)
        elif (c[:, 0] < x0).any() or (c[:, 0] > x1).any() or (c[:, 1] < z0).any() or (c[:, 1] > z1).any():
            s["e"] = s["e"] * 0.9
        out.append(s)
    return out, cnt


DEFAULT_PARAMS = {"erode": ERODE, "trim": TRIM, "snap_deg": SNAP_DEG, "wall_snap": WALL_SNAP, "priors": True,
                  "leftovers": True}


def add_leftovers(inst_map: np.ndarray, instances: list[dict], class_map: np.ndarray, id2label: dict):
    """Furniture pixels of the class map that no panoptic instance claims (the panoptic model missed
    them, SegFormer did not) become extra instances: per class, 5x5 opening + connected components of at
    least 0.12% of the panorama (the v2 rule). Returns (instance map, instance list) extended."""
    from .floorplan import FURNITURE, class_ids, instances as components
    from .instances import MIN_SHARE, family_of
    fam = family_of(id2label)
    inst_map = inst_map.copy()
    out = list(instances)
    free = inst_map < 0
    min_px = int(MIN_SHARE * class_map.size)
    for name in FURNITURE:
        ids = class_ids(id2label, [name])
        if not ids:
            continue
        m = np.isin(class_map, ids) & free
        if m.sum() < min_px:
            continue
        for comp in components(m, min_px):
            idx = max([i["id"] for i in out], default=-1) + 1
            inst_map[comp & free] = idx
            out.append({"id": idx, "label_id": ids[0], "label": name, "family": fam.get(ids[0], name),
                        "pixels": int(comp.sum()), "leftover": True})
    return inst_map, out


def build_solids(dist: np.ndarray, inst_map: np.ndarray, instances: list[dict], box: dict, id2label: dict,
                 params: dict | None = None, class_map: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """All furniture solids of a room (frame B). Returns (solids, counts). With params["leftovers"] and a
    class map, unclaimed furniture pixels are added as instances first (add_leftovers)."""
    from .instances import family_of
    params = {**DEFAULT_PARAMS, **(params or {})}
    if params.get("leftovers") and class_map is not None:
        inst_map, instances = add_leftovers(inst_map, instances, class_map, id2label)
    fam = family_of(id2label)
    family = {str(id2label[k]).split(";")[0].strip(): v for k, v in fam.items()}
    P = rm.aligned_points(dist, box["R"]) @ box["Q"]
    hgt = box["camera_height"] - P[..., 1]
    fly = _flying(dist)
    solids, cnt = [], {"instances": len(instances), "too_few_points": 0, "clamped": 0, "rejected": 0}
    for it in instances:
        m = (inst_map == it["id"]).astype(np.uint8)
        if params["erode"] > 0:
            m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(params["erode"]))
        m = (m > 0) & ~fly
        s = instance_solid(P[m], hgt[m], box, it["label"], params) if m.sum() >= MIN_POINTS else None
        if s is None:
            cnt["too_few_points"] += 1
            continue
        if s["status"] == "rejected":
            cnt["rejected"] += 1
            continue
        cnt["clamped"] += s["status"] == "clamped"
        s.update({"instance": it["id"], "family": it["family"]})
        solids.append(s)
    solids, c2 = resolve(solids, box, family)
    cnt.update(c2)
    cnt["solids"] = len(solids)
    return solids, cnt


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #
def _faces_of(s: dict, box: dict) -> list[dict]:
    """5 faces (4 sides + top) of a solid in frame B: origin o, edges a, b (subdivided later)."""
    c = _corners(s)
    yb, yt = box["floor_y"] - s["bottom"], box["floor_y"] - s["top"]      # Y down
    P3 = lambda p, y: np.array([p[0], y, p[1]], float)  # noqa: E731
    faces = []
    for i in range(4):
        p0, p1 = c[i], c[(i + 1) % 4]
        faces.append({"o": P3(p0, yt), "a": P3(p1, yt) - P3(p0, yt), "b": np.array([0.0, yb - yt, 0.0]), "kind": "side"})
    faces.append({"o": P3(c[0], yt), "a": P3(c[1], yt) - P3(c[0], yt), "b": P3(c[3], yt) - P3(c[0], yt), "kind": "top"})
    return faces


def solids_part(solids: list[dict], box: dict, dist: np.ndarray, inst_map: np.ndarray, texture: np.ndarray) -> dict:
    """One textured mesh part of all solids (vertices in frame A, texture = the panorama extended by a
    quarter width on the right so that triangles crossing the seam can be mapped without a jump)."""
    H, W = dist.shape
    th, tw = texture.shape[:2]
    ext_w = tw + tw // 4
    tex = np.concatenate([texture, texture[:, :tw // 4]], 1)
    verts, uvs, faces = [], [], []
    base = 0
    ctr_y = box["floor_y"]
    for s in solids:
        F = _faces_of(s, box)
        centre = np.array([s["c"][0], box["floor_y"] - (s["top"] + s["bottom"]) / 2, s["c"][1]])
        grids = []
        for f in F:
            na = max(int(np.ceil(np.linalg.norm(f["a"]) / CELL)), 1)
            nb = max(int(np.ceil(np.linalg.norm(f["b"]) / CELL)), 1)
            sa, sb = np.linspace(0, 1, na + 1), np.linspace(0, 1, nb + 1)
            PB = f["o"] + sa[None, :, None] * f["a"] + sb[:, None, None] * f["b"]
            Pc = (PB @ box["Q"].T) @ box["R"]                               # camera frame
            r = np.linalg.norm(Pc, axis=-1)
            d = Pc / r[..., None]
            u, v = eq.dirs_to_uv(d, W, H)
            ui = np.clip(np.round(u).astype(int) % W, 0, W - 1)
            vi = np.clip(np.round(v).astype(int), 0, H - 1)
            n = np.cross(f["a"], f["b"])
            n = n / max(np.linalg.norm(n), 1e-9)
            if n @ (centre - (f["o"] + 0.5 * f["a"] + 0.5 * f["b"])) > 0:
                n = -n                                                      # outward normal
            facing = (n @ box["Q"].T @ box["R"] @ (-Pc.reshape(-1, 3)).T).reshape(r.shape) > 0
            obs = dist[vi, ui]
            vis = facing & (np.abs(obs - r) < np.maximum(0.12, 0.08 * r)) & (inst_map[vi, ui] == s["instance"])
            grids.append({"PB": PB, "u": u, "v": v, "vis": vis, "na": na, "nb": nb, "n": n})
        donor = max(range(4), key=lambda i: grids[i]["vis"].mean())        # most visible side
        for gi, g in enumerate(grids):
            u, v, vis = g["u"].copy(), g["v"].copy(), g["vis"]
            if (~vis).any():
                own = gi == donor or (gi == 4 and vis.any())
                src = g if own else grids[donor]
                if not src["vis"].any():
                    pass                                                    # nothing visible: keep own projection
                elif src is g:                                              # nearest visible vertex of this face
                    yy, xx = np.nonzero(vis)
                    _, idx = cv2.distanceTransformWithLabels((~vis).astype(np.uint8), cv2.DIST_L2, 3,
                                                             labelType=cv2.DIST_LABEL_PIXEL)
                    lut = np.zeros(idx.max() + 1, int)
                    lut[idx[vis]] = np.arange(len(yy))
                    k = lut[idx]
                    u = np.where(vis, u, g["u"][yy[k], xx[k]])
                    v = np.where(vis, v, g["v"][yy[k], xx[k]])
                else:                                                       # same face coordinates on the donor
                    sb_, sa_ = np.meshgrid(np.linspace(0, 1, g["nb"] + 1), np.linspace(0, 1, g["na"] + 1), indexing="ij")
                    ia = np.round(sa_ * src["na"]).astype(int)
                    ib = np.round(sb_ * src["nb"]).astype(int)
                    u = np.where(vis, u, src["u"][ib, ia])
                    v = np.where(vis, v, src["v"][ib, ia])
            PA = (g["PB"] @ box["Q"].T).reshape(-1, 3)
            uu, vv = u.reshape(-1), v.reshape(-1)
            na, nb = g["na"], g["nb"]
            idx = np.arange((na + 1) * (nb + 1)).reshape(nb + 1, na + 1)
            q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
            tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            # outward-facing winding
            p0, p1, p2 = PA[tri[0, 0]], PA[tri[0, 1]], PA[tri[0, 2]]
            if np.cross(p1 - p0, p2 - p0) @ (g["n"] @ box["Q"].T) < 0:
                tri = tri[:, ::-1]
            # unshared vertices per triangle so seam-crossing triangles can be shifted by one width
            tv = PA[tri].reshape(-1, 3)
            tu = (uu[tri] + 0.5) / W
            span = tu.max(1) - tu.min(1)
            wrap = span > 0.5
            tu[wrap] = np.where(tu[wrap] < 0.5, tu[wrap] + 1.0, tu[wrap])
            tvv = (vv[tri] + 0.5) / H
            verts.append(tv)
            uvs.append(np.stack([tu.reshape(-1) * tw / ext_w, tvv.reshape(-1)], 1))
            faces.append(np.arange(len(tv)).reshape(-1, 3) + base)
            base += len(tv)
    if not verts:
        return {"vertices": np.zeros((0, 3)), "uvs": np.zeros((0, 2)), "faces": np.zeros((0, 3), int),
                "texture": tex, "kind": "furniture"}
    del ctr_y
    return {"vertices": np.concatenate(verts), "uvs": np.concatenate(uvs), "faces": np.concatenate(faces),
            "texture": tex, "kind": "furniture"}


def footprints_cam(solids: list[dict], box: dict) -> list[np.ndarray]:
    """Footprint corners (x, z) of each solid in the camera frame (for comparison with a layout)."""
    out = []
    for s in solids:
        c = _corners(s)
        PB = np.c_[c[:, 0], np.zeros(4), c[:, 1]]
        out.append(((PB @ box["Q"].T) @ box["R"])[:, [0, 2]])
    return out
