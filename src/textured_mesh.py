"""Textured grid mesh from a depth map, with disocclusion filling and room-plane completion.

The Poisson mesh (reconstruct_3d.poisson_mesh) blurs colours (one colour per vertex), rounds off
edges and leaves holes. This module builds the mesh directly on the pixel grid instead:

  Layer 0  observed surface   one vertex per pixel, two triangles per pixel quad, the generated
                              image as UV texture. Triangles are cut at depth discontinuities.
  Layer 1  disocclusion band  behind every cut the background surface is continued under the
                              foreground object: depth grown in from the background side,
                              colour from inpainting (cv2.inpaint).
  Layer 2  room planes        RANSAC planes fitted to the floor / wall / ceiling masks; every
                              pixel ray is intersected with them, which gives the room surface
                              hidden behind furniture, out to the edges of the view frustum.

All three layers share one texture atlas [image | inpainted for layer 1 | plane texture for layer 2]
and are written as one .glb (unlit material, so the colours are exactly the image colours).
The vertex grid is coarser than the image (one vertex every GRID_STRIDE pixels) while the texture
keeps the full resolution, which keeps the file small without losing image detail.

Image-processing operations used: morphological erosion / dilation (toggle mapping to sharpen
depth edges, geodesic dilation to build the band masks, opening to clean masks), Canny edges
(to confirm weak depth steps), inpainting (Telea fast marching), RANSAC plane fitting.
"""

import json
import struct
from pathlib import Path

import cv2
import numpy as np

from .reconstruct_3d import DEFAULT_FOV_DEG, backproject, intrinsics

# All constants are fixed design choices (no ground truth exists for generated rooms to tune on).
CUT_STRONG = 0.08       # neighbouring pixels differing by > 8% in depth: always a discontinuity
CUT_WEAK = 0.03         # > 3%: a discontinuity only if a Canny edge of the image is within CUT_EDGE_DIST px
CUT_EDGE_DIST = 2
SHARPEN_KSIZE = 5       # window of the toggle mapping that turns blurred depth edges into steps
SHARPEN_ITERS = 3
BAND = 24               # px: how far the background is continued behind a foreground edge
INPAINT_RADIUS = 5
PLANE_THRESH = 0.015    # RANSAC inlier distance, as a share of the median scene depth
FLOOR_MIN_INLIERS = 0.60   # floor / ceiling: share of the mask points on the plane
WALL_MIN_SHARE = 0.12      # a wall plane must hold at least 12% of all wall points
PLANE_MIN_POINTS = 2000
MAX_WALLS = 3
PLANE_MAX_BEHIND = 0.15    # reject a plane if > 15% of the image shows geometry clearly behind it
GRID_STRIDE = 2         # one vertex every 2 pixels (the texture stays at full resolution)
FG_GROW = 2             # px: foreground depth is grown by this much at discontinuities, so the mixed
#                         boundary pixels of an object stay on the object and not on the wall behind
INPAINT_GROW = 11       # px (ellipse): inpainting masks are dilated by this much, so that no
#                         foreground-coloured boundary pixel is used as a source for the background
PLANE_VISIBLE_TOL = 0.04   # a pixel shows a plane if its depth is within 4% of the plane's depth
PLANE_BLEND_NEAR, PLANE_BLEND_WIDTH = 6, 24   # px: where inpainted plane colour fades to the plane's median colour
TEXTURE_JPEG_QUALITY = 95


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# --------------------------------------------------------------------------- #
# Depth edges
# --------------------------------------------------------------------------- #
def sharpen_depth_edges(depth: np.ndarray, ksize: int = SHARPEN_KSIZE, iters: int = SHARPEN_ITERS) -> np.ndarray:
    """Morphological toggle mapping: turn blurred depth transitions into clean steps.

    A depth network spreads an object boundary over several pixels, so the boundary pixels get
    depths between foreground and background (flying pixels). In every window whose depth range
    is larger than CUT_STRONG, a pixel is replaced by the local minimum (erosion) or the local
    maximum (dilation), whichever is closer to its value. Smooth surfaces are not touched.
    """
    d = depth.astype(np.float32).copy()
    k = np.ones((ksize, ksize), np.uint8)
    for _ in range(iters):
        lo, hi = cv2.erode(d, k), cv2.dilate(d, k)
        trans = (hi / np.maximum(lo, 1e-6) - 1) > CUT_STRONG
        d = np.where(trans, np.where(d - lo < hi - d, lo, hi), d)
    return d


def grow_foreground(depth: np.ndarray, px: int = FG_GROW) -> np.ndarray:
    """Grow the foreground by `px` pixels at depth discontinuities (grey erosion of the depth map).

    The depth edge and the colour edge of an object never coincide exactly, and the colour edge
    itself is a few mixed pixels wide. Wherever a (2px+1) window contains a discontinuity, the
    pixel takes the window minimum = the foreground depth. Object-coloured boundary pixels then
    belong to the object's geometry instead of being painted on the background behind it.
    """
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    lo, hi = cv2.erode(depth, k), cv2.dilate(depth, k)
    return np.where(hi / np.maximum(lo, 1e-6) - 1 > CUT_STRONG, lo, depth).astype(np.float32)


def depth_cuts(depth: np.ndarray, edges: np.ndarray | None = None, weak: float = CUT_WEAK) -> tuple[np.ndarray, np.ndarray]:
    """Where neighbouring pixels must NOT be joined by a triangle.

    Returns (cut_h, cut_v): cut_h[v, u] is the link between pixel (v, u) and (v, u+1), cut_v[v, u]
    between (v, u) and (v+1, u). A link is cut if the relative depth step is above CUT_STRONG, or
    above CUT_WEAK with a Canny edge next to it (hysteresis: the image edge confirms a weak step).
    """
    d = depth.astype(np.float64)
    near_edge = np.zeros(d.shape, bool)
    if edges is not None:
        near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((2 * CUT_EDGE_DIST + 1,) * 2, np.uint8)) > 0

    def step(a, b):
        return np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-6) - 1

    rh, rv = step(d[:, :-1], d[:, 1:]), step(d[:-1], d[1:])
    cut_h = (rh > CUT_STRONG) | ((rh > weak) & (near_edge[:, :-1] | near_edge[:, 1:]))
    cut_v = (rv > CUT_STRONG) | ((rv > weak) & (near_edge[:-1] | near_edge[1:]))
    return cut_h, cut_v


def cut_sides(depth: np.ndarray, cut_h: np.ndarray, cut_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every cut link: mark its nearer pixel (`near`) and its farther pixel (`far`)."""
    near, far = np.zeros(depth.shape, bool), np.zeros(depth.shape, bool)
    a, b = depth[:, :-1], depth[:, 1:]
    near[:, :-1] |= cut_h & (a < b)
    far[:, 1:] |= cut_h & (a < b)
    near[:, 1:] |= cut_h & (b < a)
    far[:, :-1] |= cut_h & (b < a)
    a, b = depth[:-1], depth[1:]
    near[:-1] |= cut_v & (a < b)
    far[1:] |= cut_v & (a < b)
    near[1:] |= cut_v & (b < a)
    far[:-1] |= cut_v & (b < a)
    return near, far


def grid_faces(valid: np.ndarray, cut_h: np.ndarray, cut_v: np.ndarray) -> np.ndarray:
    """Two triangles for every pixel quad whose 4 corners are valid and whose 4 sides are not cut.

    Returns Mx3 vertex indices; the vertex of pixel (v, u) has index v * W + u.
    """
    w = valid.shape[1]
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    q &= ~cut_h[:-1] & ~cut_h[1:] & ~cut_v[:, :-1] & ~cut_v[:, 1:]
    vv, uu = np.nonzero(q)
    a = (vv * w + uu).astype(np.int64)
    b, c, d = a + 1, a + w, a + w + 1
    return np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])


# --------------------------------------------------------------------------- #
# Layer 1: disocclusion band behind foreground edges
# --------------------------------------------------------------------------- #
def disocclusion_layer(depth: np.ndarray, cut_h: np.ndarray, cut_v: np.ndarray, band: int = BAND) -> dict:
    """Continue the background behind foreground objects along every cut.

    1. `near` / `far` = the foreground / background pixel of each cut link.
    2. Geodesic dilation: grow the `near` pixels into the foreground, one pixel per step (3x3
       cross), never entering background-boundary pixels -> the strip of foreground that hides
       background, with the step number = distance from the edge.
    3. Depth "filled from the background side": the background depth at the cut is pushed into
       that strip by repeated grey dilation (each step copies the farthest known neighbour).
    Returns {"depth", "valid", "hidden", "inpaint_mask"}: hidden = strip up to `band` px where a
    background surface is placed; inpaint_mask = strip up to 2 x band px (colours are inpainted
    over the wider strip so that the part next to the background takes background colours).
    """
    near, far = cut_sides(depth, cut_h, cut_v)
    far_only = far & ~near
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    reach, dist = near.copy(), np.where(near, 1, 0).astype(np.int32)
    for i in range(2, 2 * band + 1):
        new = (cv2.dilate(reach.astype(np.uint8), cross) > 0) & ~reach & ~far_only
        if not new.any():
            break
        reach |= new
        dist[new] = i
    bg = np.where(far_only, depth, 0).astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(2 * band + 1):
        new = reach & (bg == 0) & (cv2.dilate(bg, k3) > 0)
        if not new.any():
            break
        bg[new] = cv2.dilate(bg, k3)[new]
    wide = reach & (bg > 0) & (depth < bg * (1 - CUT_WEAK))
    hidden = wide & (dist <= band)
    ring = (cv2.dilate(hidden.astype(np.uint8), k3) > 0) & ~hidden & far_only
    return {"depth": np.where(hidden, bg, depth).astype(np.float32), "valid": hidden | ring, "hidden": hidden,
            "inpaint_mask": wide}


# --------------------------------------------------------------------------- #
# Layer 2: room planes
# --------------------------------------------------------------------------- #
def _ransac_plane(points: np.ndarray, thresh: float):
    """Open3D RANSAC plane. Returns (unit normal n, offset d with n.x + d = 0 and d > 0, inlier indices)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.utility.random.seed(0)
    (a, b, c, d), inl = pcd.segment_plane(distance_threshold=thresh, ransac_n=3, num_iterations=400)
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    if d < 0:                                   # make the camera (origin) lie on the positive side
        n, d = -n, -d
    return n, float(d), np.asarray(inl)


def fit_room_planes(depth: np.ndarray, masks: dict, k: tuple) -> list[dict]:
    """RANSAC planes for the floor, the ceiling and up to MAX_WALLS walls, from the segmentation masks.

    Each returned dict has kind, n, d, inlier_share, rms (m), accepted and, if rejected, reason.
    A plane is skipped when the fit is poor:
      - floor / ceiling: fewer than FLOOR_MIN_INLIERS of the mask points lie on it, or it is not
        roughly horizontal (more than 35 degrees from the image's vertical axis);
      - wall: it holds less than WALL_MIN_SHARE of the wall points, or it is not roughly vertical;
      - any plane: more than PLANE_MAX_BEHIND of the image shows geometry clearly BEHIND it (then
        it is not a boundary of the room, e.g. a column or a wall segment in the foreground).
    """
    pts = backproject(depth, *k)
    rays = backproject(np.ones_like(depth), *k)
    scale = float(np.median(depth))
    thresh = PLANE_THRESH * scale
    rng = np.random.default_rng(0)
    planes = []

    def check(kind, n, d, share, rms, ok, reason):
        den = rays @ n
        zp = np.where(den < -1e-6, -d / np.minimum(den, -1e-6), np.inf)
        behind = float(np.mean(np.isfinite(zp) & (depth > zp * 1.10)))
        if ok and behind > PLANE_MAX_BEHIND:
            ok, reason = False, f"{100 * behind:.0f}% of the image lies behind it"
        planes.append({"kind": kind, "n": n, "d": d, "inlier_share": share, "rms": rms, "behind_share": behind,
                       "accepted": bool(ok), "reason": "" if ok else reason})
        return ok

    def sample(mask):
        p = pts[cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0]     # stay off the mask borders
        return p if len(p) <= 60_000 else p[rng.choice(len(p), 60_000, replace=False)]

    up = np.array([0.0, 1.0, 0.0])
    for kind in ("floor", "ceiling"):
        p = sample(masks[kind])
        if len(p) < PLANE_MIN_POINTS:
            continue
        n, d, inl = _ransac_plane(p, thresh)
        share = len(inl) / len(p)
        rms = float(np.sqrt(np.mean((p[inl] @ n + d) ** 2)))
        level = abs(n[1]) >= np.cos(np.radians(35))
        ok = check(kind, n, d, share, rms, share >= FLOOR_MIN_INLIERS and level,
                   "not horizontal" if not level else f"only {100 * share:.0f}% of the mask on the plane")
        if ok and kind == "floor":
            up = n
    p = sample(masks["wall"])
    total = len(p)
    for i in range(MAX_WALLS):
        if len(p) < max(PLANE_MIN_POINTS, WALL_MIN_SHARE * total):
            break
        n, d, inl = _ransac_plane(p, thresh)
        share = len(inl) / total
        rms = float(np.sqrt(np.mean((p[inl] @ n + d) ** 2)))
        upright = abs(float(n @ up)) <= 0.35
        ok = share >= WALL_MIN_SHARE and upright
        check(f"wall{i + 1}", n, d, share, rms, ok,
              "not vertical" if not upright else f"only {100 * share:.0f}% of the wall points")
        if share < WALL_MIN_SHARE:
            break
        p = p[np.abs(p @ n + d) > 2 * thresh]
    return planes


def room_depth(planes: list[dict], rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth of the room surface along every pixel ray, and which plane it belongs to.

    rays = HxWx3 ray directions with z = 1 (one per vertex of the grid).
    Returns (z, index): z is inf and index is -1 where no accepted plane is hit.
    The room is treated as the convex space in front of all accepted planes with the camera
    inside. A ray leaves that space at the FIRST plane it crosses, so the room surface is
        z(u, v) = min over planes of  -d / (n . ray(u, v))     for planes the ray runs towards.
    This automatically extends every plane to the edges of the view frustum and clips the planes
    against each other (floor against walls, wall against wall).
    """
    shape = rays.shape[:2]
    z, index = np.full(shape, np.inf), np.full(shape, -1, np.int32)
    for i, p in enumerate(planes):
        if p["accepted"]:
            den = rays @ p["n"]
            zp = np.where(den < -1e-6, -p["d"] / np.minimum(den, -1e-6), np.inf)
            index[zp < z] = i
            z = np.minimum(z, zp)
    return z, index


def plane_texture(rgb: np.ndarray, hidden: np.ndarray, visible: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Texture of the hidden room surface: each plane is painted from the VISIBLE part of the same plane.

    For every plane, cv2.inpaint (Telea) fills its hidden pixels using only pixels that show
    that plane as sources: everything else (furniture, other planes, shadows on other surfaces)
    is masked out. So the floor behind a sofa gets floor colours, not sofa or shadow colours.
    Inpainting only carries colour a short way: far from the visible part it drags the last
    colour it saw (often the dark contact shadow next to the furniture) across the whole gap.
    So beyond PLANE_BLEND_NEAR px from the nearest visible pixel the colour fades, over
    PLANE_BLEND_WIDTH px, to the MEDIAN colour of the visible part of the plane.
    All arrays are full resolution; index = plane id per pixel. A plane with almost nothing
    visible falls back to its neighbourhood (plain inpainting of the hidden area).
    """
    out = cv2.inpaint(rgb, cv2.dilate(hidden.astype(np.uint8), _ellipse(INPAINT_GROW)), INPAINT_RADIUS,
                      cv2.INPAINT_TELEA)                                  # fallback for every hidden pixel
    h, w = hidden.shape
    for i in np.unique(index[hidden]):
        hid = hidden & (index == i)
        src = cv2.erode((visible & (index == i)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        vv, uu = np.nonzero(hid)
        pad = 48
        v0, v1, u0, u1 = max(vv.min() - pad, 0), min(vv.max() + pad + 1, h), max(uu.min() - pad, 0), min(uu.max() + pad + 1, w)
        if src[v0:v1, u0:u1].sum() < 200:
            v0, v1, u0, u1 = 0, h, 0, w
        if src[v0:v1, u0:u1].sum() < 200:
            continue
        roi = cv2.inpaint(np.ascontiguousarray(rgb[v0:v1, u0:u1]), (~src[v0:v1, u0:u1]).astype(np.uint8),
                          INPAINT_RADIUS, cv2.INPAINT_TELEA)
        median = np.median(rgb[src], axis=0)
        dist = cv2.distanceTransform((~src[v0:v1, u0:u1]).astype(np.uint8), cv2.DIST_L2, 3)
        a = np.clip((dist - PLANE_BLEND_NEAR) / PLANE_BLEND_WIDTH, 0, 1)[..., None]
        roi = ((1 - a) * roi + a * median).astype(np.uint8)
        fill = (cv2.dilate(hid.astype(np.uint8), _ellipse(INPAINT_GROW)) > 0) & (index == i) & ~src
        out[v0:v1, u0:u1][fill[v0:v1, u0:u1]] = roi[fill[v0:v1, u0:u1]]
    return out


def room_layer(depth: np.ndarray, room_z: np.ndarray) -> dict:
    """Room surface where it is hidden behind something (furniture, objects): layer 2."""
    k3 = np.ones((3, 3), np.uint8)
    hidden = np.isfinite(room_z) & (room_z > depth * (1 + 2 * CUT_WEAK))
    hidden = cv2.morphologyEx(hidden.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0          # drop specks
    ring = (cv2.dilate(hidden.astype(np.uint8), k3) > 0) & ~hidden
    layer = np.where(hidden, np.where(np.isfinite(room_z), room_z, depth), depth).astype(np.float32)
    return {"depth": layer, "valid": hidden | ring, "hidden": hidden, "inpaint_mask": hidden}


# --------------------------------------------------------------------------- #
# Mesh assembly
# --------------------------------------------------------------------------- #
def build_textured_mesh(depth: np.ndarray, rgb: np.ndarray, edges: np.ndarray | None = None,
                        masks: dict | None = None, fov_deg: float = DEFAULT_FOV_DEG, fill: bool = True,
                        planes: bool = True, stride: int = GRID_STRIDE) -> dict:
    """Depth (m) + RGB uint8 (+ Canny edges, + room masks) -> textured mesh.

    Returns {"vertices" Nx3 (camera coordinates), "uvs" Nx2 (atlas, v down), "faces" Mx3,
    "texture" RGB atlas, "layer" per-face layer id, "planes", "stats", "target_z"}.
    fill=False builds only layer 0; planes=False (or masks=None) skips layer 2.
    The geometry lives on a grid with one vertex every `stride` pixels (plus the last pixel row
    and column, so the mesh reaches the image border); textures are full size.
    """
    hh, ww = depth.shape
    fx, fy, cx, cy = intrinsics(ww, hh, fov_deg)
    s = int(stride)
    ys = np.unique(np.r_[np.arange(0, hh, s), hh - 1])     # pixel rows / columns that carry a vertex
    xs = np.unique(np.r_[np.arange(0, ww, s), ww - 1])
    pick = np.ix_(ys, xs)
    gx, gy = np.meshgrid(xs, ys)
    rays = np.stack([(gx - cx) / fx, (gy - cy) / fy, np.ones(gx.shape)], axis=-1)   # vertex = ray * depth
    d_full = grow_foreground(sharpen_depth_edges(depth))
    d0 = np.ascontiguousarray(d_full[pick])
    h, w = d0.shape
    e0 = None if edges is None else np.ascontiguousarray(
        (cv2.dilate(edges.astype(np.uint8), np.ones((s, s), np.uint8)) > 0)[pick])
    weak = CUT_WEAK * (1 + s) / 2                        # a smooth slope spans `s` pixels per link
    cut_h, cut_v = depth_cuts(d0, e0, weak)

    def up(m):                                           # coarse mask -> full-resolution mask
        return cv2.resize(m.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST) > 0

    layers = [{"depth": d0, "valid": np.ones((h, w), bool), "tex": rgb}]
    plane_list = []
    stats = {"cut_links": int(cut_h.sum() + cut_v.sum())}
    if fill:
        covered = np.zeros((h, w), bool)
        room_z = np.full((h, w), np.inf)
        l2 = None
        if planes and masks is not None:
            plane_list = fit_room_planes(d_full, masks, (fx, fy, cx, cy))
            room_z, index = room_depth(plane_list, rays)
            l2 = room_layer(d0, room_z)
            covered = l2["hidden"]
        l1 = disocclusion_layer(d0, cut_h, cut_v, band=max(BAND // s, 4))
        # where the room surface already fills the gap, the band is only kept for objects that
        # stand in front of the room surface (object behind object)
        l1["hidden"] &= ~covered | (l1["depth"] < room_z * (1 - 0.05))
        ring = (cv2.dilate(l1["hidden"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~l1["hidden"]
        l1["valid"] = l1["hidden"] | (l1["valid"] & ring)
        l1["depth"] = np.where(l1["hidden"], l1["depth"], d0)
        m1 = cv2.dilate(up(l1["inpaint_mask"]).astype(np.uint8), _ellipse(INPAINT_GROW))
        l1["tex"] = cv2.inpaint(rgb, m1, INPAINT_RADIUS, cv2.INPAINT_TELEA) if m1.any() else rgb
        layers.append(l1)
        if covered.any():
            visible = np.isfinite(room_z) & (np.abs(d0 - room_z) < PLANE_VISIBLE_TOL * np.where(np.isfinite(room_z), room_z, 1))
            idx_full = cv2.resize(index, (ww, hh), interpolation=cv2.INTER_NEAREST)
            l2["tex"] = plane_texture(rgb, up(covered), up(visible), idx_full)
            layers.append(l2)
        stats["hidden_band_px"], stats["hidden_room_px"] = int(l1["hidden"].sum()) * s * s, int(covered.sum()) * s * s

    n_tiles = len(layers)
    verts, uvs, faces, face_layer = [], [], [], []
    for i, layer in enumerate(layers):
        ch, cv = (cut_h, cut_v) if i == 0 else depth_cuts(layer["depth"], None)
        f = grid_faces(layer["valid"], ch, cv)
        faces.append(f + i * h * w)
        face_layer.append(np.full(len(f), i, np.uint8))
        verts.append((rays * layer["depth"][..., None].astype(np.float64)).reshape(-1, 3))
        uvs.append(np.stack([((gx + 0.5) / ww + i) / n_tiles, (gy + 0.5) / hh], -1).reshape(-1, 2))
    verts, uvs, faces = np.concatenate(verts), np.concatenate(uvs), np.concatenate(faces)
    used = np.unique(faces)                         # drop vertices no triangle uses
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    face_layer = np.concatenate(face_layer)
    stats.update({"vertices": int(len(used)), "triangles": int(len(faces)),
                  **{f"triangles_layer{i}": int((face_layer == i).sum()) for i in range(n_tiles)},
                  "planes_accepted": sum(p["accepted"] for p in plane_list), "planes_fitted": len(plane_list)})
    return {"vertices": verts[used].astype(np.float32), "uvs": uvs[used].astype(np.float32),
            "faces": remap[faces].astype(np.uint32), "texture": np.hstack([la["tex"] for la in layers]),
            "layer": face_layer, "planes": plane_list, "stats": stats, "target_z": float(np.median(depth))}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_glb(mesh: dict, path, double_sided: bool = True) -> None:
    """Write the textured mesh as a binary glTF 2.0 file (.glb), Y up, camera at the origin looking down -Z.

    One mesh, one unlit material with the atlas as base-colour texture (JPEG embedded).
    Unlit = the viewer shows the texture colours as they are, without adding its own lighting.
    double_sided=False: back faces are culled by the viewer (v3 meshes: the shell faces inwards, so the
    walls between an outside viewer and the room disappear by themselves - the dollhouse effect).
    """
    pos = (mesh["vertices"] * np.array([1, -1, -1], np.float32)).astype("<f4")
    uv = mesh["uvs"].astype("<f4")                                  # glTF: v = 0 at the top of the image
    idx = mesh["faces"].astype("<u4")
    ok, png = cv2.imencode(".jpg", cv2.cvtColor(mesh["texture"], cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, TEXTURE_JPEG_QUALITY])
    assert ok
    chunks, views, offset = [], [], 0
    for data, target in ((pos.tobytes(), 34962), (uv.tobytes(), 34962), (idx.tobytes(), 34963), (png.tobytes(), None)):
        pad = (-len(data)) % 4
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        chunks.append(data + b"\x00" * pad)
        offset += len(data) + pad
    gltf = {
        "asset": {"version": "2.0", "generator": "RoomGen3D"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "room"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                                "roughnessFactor": 1.0},
                       "doubleSided": bool(double_sided), "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": int(idx.size), "type": "SCALAR"},
        ],
        "bufferViews": views, "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    body = b"".join(chunks)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(body)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(body), b"BIN\x00") + body)


def export_obj(mesh: dict, path) -> None:
    """Write the textured mesh as .obj + .mtl + .png (same base name), Y up."""
    path = Path(path)
    cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(mesh["texture"], cv2.COLOR_RGB2BGR))
    path.with_suffix(".mtl").write_text(f"newmtl room\nKa 1 1 1\nKd 1 1 1\nillum 0\nmap_Kd {path.with_suffix('.png').name}\n")
    v = mesh["vertices"] * np.array([1, -1, -1], np.float32)
    f = mesh["faces"].astype(np.int64) + 1
    with open(path, "w") as out:
        out.write(f"mtllib {path.with_suffix('.mtl').name}\nusemtl room\n")
        np.savetxt(out, v, fmt="v %.5f %.5f %.5f")
        np.savetxt(out, np.stack([mesh["uvs"][:, 0], 1 - mesh["uvs"][:, 1]], 1), fmt="vt %.6f %.6f")
        np.savetxt(out, np.stack([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]], 1), fmt="f %d/%d %d/%d %d/%d")


# --------------------------------------------------------------------------- #
# Software renderer for figures (perspective, from the original camera or rotated around the scene)
# --------------------------------------------------------------------------- #
def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0):
    """Random points on a triangle mesh. Returns (points, face index, barycentric weights).

    Triangles are drawn with probability proportional to area / depth^2, i.e. to the area they
    cover in the original image, so the samples are evenly dense on screen.
    """
    rng = np.random.default_rng(seed)
    a, b, c = (vertices[faces[:, i]].astype(np.float64) for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    z = np.maximum((a[:, 2] + b[:, 2] + c[:, 2]) / 3, 0.1)
    wgt = area / z ** 2 + 1e-12
    fi = rng.choice(len(faces), n, p=wgt / wgt.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = a[fi] * bary[:, :1] + b[fi] * bary[:, 1:2] + c[fi] * bary[:, 2:]
    return pts, fi, bary


def textured_samples(mesh: dict, n: int = 2_500_000):
    """Coloured surface samples of a textured mesh (colour looked up in the atlas)."""
    pts, fi, bary = sample_surface(mesh["vertices"], mesh["faces"], n)
    uv = (mesh["uvs"][mesh["faces"][fi]] * bary[:, :, None]).sum(1)
    tex = mesh["texture"]
    x = np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
    y = np.clip((uv[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
    return pts, tex[y, x].astype(np.float64) / 255.0


def vertex_color_samples(o3d_mesh, n: int = 2_500_000):
    """Coloured surface samples of an Open3D mesh with vertex colours (the Poisson mesh)."""
    v, f = np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles)
    pts, fi, bary = sample_surface(v, f, n)
    cols = (np.asarray(o3d_mesh.vertex_colors)[f[fi]] * bary[:, :, None]).sum(1)
    return pts, cols


def render_view(points: np.ndarray, colors: np.ndarray, yaw_deg: float = 0.0, size: int = 512,
                fov_deg: float = DEFAULT_FOV_DEG, pivot_z: float | None = None, splat: int = 2,
                bg=(255, 255, 255)) -> np.ndarray:
    """Perspective render of coloured points with the pipeline's pinhole camera.

    yaw_deg = 0 is the original camera. Otherwise the camera orbits around the point on the
    optical axis at depth pivot_z (default: the median depth) by that angle, still looking at it,
    like turning the model in a viewer. Points are painted far-to-near; holes stay `bg`.
    """
    p = np.asarray(points, np.float64)
    pivot = np.array([0.0, 0.0, float(np.median(p[:, 2])) if pivot_z is None else pivot_z])
    a = np.radians(yaw_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    q = (p - pivot) @ rot.T + pivot
    fx, fy, cx, cy = intrinsics(size, size, fov_deg)
    front = q[:, 2] > 0.05
    q, c = q[front], np.asarray(colors)[front]
    order = np.argsort(-q[:, 2])
    q, c = q[order], (c[order] * 255).astype(np.uint8)
    u0 = np.round(q[:, 0] / q[:, 2] * fx + cx).astype(int)
    v0 = np.round(q[:, 1] / q[:, 2] * fy + cy).astype(int)
    off = np.arange(splat) - splat // 2
    du, dv = [x.ravel() for x in np.meshgrid(off, off)]
    u, v = (u0[:, None] + du[None]).ravel(), (v0[:, None] + dv[None]).ravel()
    c = np.repeat(c, len(du), axis=0)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    img = np.empty((size, size, 3), np.uint8)
    img[:] = bg
    img[v[ok], u[ok]] = c[ok]
    return img
