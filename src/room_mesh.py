"""Full-room 3D from a 360 degree panorama: equirectangular distance map -> textured room mesh.

    panorama RGB (texture, e.g. 2048 x 1024 after super-resolution)
    + equirectangular distance map (m) + ADE20K class map (same grid as the distance map)
      -> gravity alignment: RANSAC floor plane (floorplan.gravity_frame); the mesh is rotated so
         that "up" is exactly -Y (camera axes, Y down) and the panorama centre is the origin
      -> room box (box_planes): floor plane, ceiling height, and four walls from the 360 degree
         floor plan (floorplan.build_floorplan on the spherical point cloud: Manhattan walls,
         closed rectangle); fallback: 2nd / 98th percentiles of the scene along the dominant axes
      -> layer 0, observed surface: one vertex every `stride` pixels on a grid that WRAPS around
         (the last column is joined to the first one, with a duplicate column that carries
         texture u = 1), plus a pole cap of triangles at the top and bottom rows. Depth edges are
         sharpened and the foreground grown (textured_mesh), all filters run on wrap-padded maps;
         triangles are cut at depth discontinuities (8%, or 4.5% next to a Canny edge)
      -> layer 1, disocclusion band: the background continued 24 px behind every cut, colour by
         inpainting (textured_mesh.disocclusion_layer)
      -> layer 2, room box: where the box surface is hidden behind furniture, the box is added
         and painted from the visible part of the same plane (textured_mesh.plane_texture)
      -> "dollhouse" variant: every triangle within 25 cm of the ceiling (and the top cap) removed,
         so the room can be looked into from above
      -> .glb export (Y up, panorama centre at the origin)

Rendering for the metrics: exact ray casting against the triangles (Open3D RaycastingScene), with
the texture looked up from the hit triangle's barycentric coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import equirect as eq
from . import floorplan as fp
from . import textured_mesh as tm
from .analysis import canny_edges

STRIDE = 2
WRAP_PAD = 16                 # grid columns copied from the opposite border before 2D filtering
CEILING_CUT = 0.25            # dollhouse: triangles higher than ceiling - 25 cm are removed
DEFAULT_CEILING = 2.7         # m above the floor if no ceiling is visible


# --------------------------------------------------------------------------- #
# Gravity and room box
# --------------------------------------------------------------------------- #
def gravity_rotation(dist: np.ndarray, class_map: np.ndarray, id2label: dict) -> tuple[np.ndarray, dict]:
    """Rotation R (rows = new axes) that makes the fitted floor normal point to -Y, keeping the
    panorama's forward direction; returns (R, frame of floorplan.gravity_frame)."""
    pts = eq.pixel_dirs(dist.shape[1], dist.shape[0]) * dist[..., None]
    sel = lambda names: np.isin(class_map, fp.class_ids(id2label, names))  # noqa: E731
    frame = fp.gravity_frame(pts, np.isfinite(dist) & (dist > 0.05), sel(fp.FLOOR_CLASSES), sel(fp.CEILING_CLASSES))
    R = np.stack([frame["right"], -frame["up"], frame["fwd"]])
    return R, frame


def aligned_points(dist: np.ndarray, R: np.ndarray) -> np.ndarray:
    """HxWx3 points of the distance map in the gravity-aligned frame."""
    return (eq.pixel_dirs(dist.shape[1], dist.shape[0]) * dist[..., None]) @ R.T


def plan_360(dist: np.ndarray, class_map: np.ndarray, id2label: dict, R: np.ndarray) -> dict:
    """Floor plan of the whole panorama (spherical point cloud, 360 degree camera)."""
    return fp.build_floorplan(dist, None, class_map, id2label, points=aligned_points(dist, R), fov_deg=360.0)


def box_planes(plan: dict, pts: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Room box as planes {n, d, accepted, kind} with n.x + d = 0 and the camera on the positive side
    (the convention of textured_mesh.room_depth), in the frame the plan was built in.

    Walls: the closed rectangle of the plan if there is one; otherwise the 2nd / 98th percentiles of
    the points along the plan's dominant directions (an axis-aligned bounding box in plan space).
    """
    fr = plan["frame"]
    up, right, fwd, cam_h = fr["up"], fr["right"], fr["fwd"], fr["d"]
    ceil_h = plan["ceiling_height"] if plan["ceiling_height"] and plan["ceiling_height"] > cam_h + 0.2 else max(DEFAULT_CEILING, cam_h + 0.5)
    planes = [{"kind": "floor", "n": up, "d": cam_h, "accepted": True},
              {"kind": "ceiling", "n": -up, "d": ceil_h - cam_h, "accepted": True}]
    if plan["room"] is not None:
        corners = plan["room"]["corners"]
        src = "closed plan rectangle"
    else:
        t = np.radians(plan["manhattan"].get("dominant_deg", 0.0))
        u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        xy = np.stack([pts.reshape(-1, 3) @ right, pts.reshape(-1, 3) @ fwd], -1)
        a, b = np.percentile(xy @ u, [2, 98]), np.percentile(xy @ v, [2, 98])
        corners = np.array([a[0] * u + b[0] * v, a[1] * u + b[0] * v, a[1] * u + b[1] * v, a[0] * u + b[1] * v])
        src = "percentile box (plan not closed)"
    c = corners.mean(0)
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n2 = np.array([d[1], -d[0]])
        if n2 @ ((p0 + p1) / 2 - c) < 0:
            n2 = -n2                                     # pointing out of the room
        off = float(n2 @ p0)
        if off <= 0.05:                                  # camera outside / on this wall: skip it
            continue
        N = n2[0] * right + n2[1] * fwd
        planes.append({"kind": f"wall{i + 1}", "n": -N, "d": off, "accepted": True})
    info = {"source": src, "ceiling_height": ceil_h, "camera_height": cam_h, "corners": corners,
            "area": float(cv2.contourArea(corners.astype(np.float32)))}
    return planes, info


# --------------------------------------------------------------------------- #
# Wrap-aware grid helpers
# --------------------------------------------------------------------------- #
def _pad(a: np.ndarray, p: int = WRAP_PAD) -> np.ndarray:
    return np.concatenate([a[:, -p:], a, a[:, :p]], 1)


def _crop(a: np.ndarray, p: int = WRAP_PAD) -> np.ndarray:
    return a[:, p:-p]


def _ext(a: np.ndarray) -> np.ndarray:
    """Append the first column at the end (the duplicate column that closes the wrap)."""
    return np.concatenate([a, a[:, :1]], 1)


def _wrap_filter(fn, img: np.ndarray, p: int) -> np.ndarray:
    return fn(cv2.copyMakeBorder(img, 0, 0, p, p, cv2.BORDER_WRAP))[:, p:-p]


def _cap_faces(top_idx: np.ndarray, pole_idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fan triangles (pole_j, ring_j, ring_j+1) between a pole and the first / last row of the grid."""
    j = np.nonzero(valid[:-1] & valid[1:])[0]
    return np.stack([pole_idx[j], top_idx[j], top_idx[j + 1]], 1)


# --------------------------------------------------------------------------- #
# Mesh
# --------------------------------------------------------------------------- #
def build_room_mesh(dist: np.ndarray, rgb: np.ndarray, class_map: np.ndarray, id2label: dict,
                    stride: int = STRIDE, fill: bool = True, box: bool = True) -> dict:
    """Equirectangular distance (H x W) + panorama texture (any size, same aspect) -> textured room mesh.

    Returns the mesh dict of textured_mesh (vertices in the gravity-aligned frame, uvs, faces,
    texture atlas, layer per face) plus "R", "plan", "box", "heights" (per vertex, m above the
    floor), "stats".
    """
    H, W = dist.shape
    th, tw = rgb.shape[:2]
    R, frame = gravity_rotation(dist, class_map, id2label)
    plan = plan_360(dist, class_map, id2label, R)
    s = int(stride)
    ys = np.unique(np.r_[np.arange(0, H, s), H - 1])
    xs = np.arange(0, W, s)
    pick = np.ix_(ys, xs)
    dirs = eq.pixel_dirs(W, H) @ R.T                               # aligned ray directions (per pixel)
    g_dirs = _ext(dirs[pick])                                       # (h, w + 1, 3)
    h, w = len(ys), len(xs)

    d_full = _wrap_filter(lambda e: tm.grow_foreground(tm.sharpen_depth_edges(e)), dist.astype(np.float32), 8)
    d0 = np.ascontiguousarray(d_full[pick])
    edges = _wrap_filter(lambda e: canny_edges(e), cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA), 8)
    e0 = (cv2.dilate(edges.astype(np.uint8), np.ones((s, s), np.uint8)) > 0)[pick]
    weak = tm.CUT_WEAK * (1 + s) / 2
    cut_h, cut_v = tm.depth_cuts(_ext(d0), _ext(e0), weak)

    def up(m):                                                      # grid mask -> texture-resolution mask
        return cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0

    layers = [{"depth": d0, "valid": np.ones((h, w), bool), "tex": rgb, "cuts": (cut_h, cut_v)}]
    planes, binfo = [], None
    stats = {"cut_links": int(cut_h.sum() + cut_v.sum()), "gravity": frame["source"]}
    if fill:
        covered = np.zeros((h, w), bool)
        room_z = np.full((h, w), np.inf)
        if box:
            planes, binfo = box_planes(plan, aligned_points(dist, R))
            room_z, index = tm.room_depth(planes, dirs[pick])
            l2 = tm.room_layer(d0, room_z)
            covered = l2["hidden"]
        dp_ = _pad(d0)
        ch, cv = tm.depth_cuts(dp_, _pad(e0), weak)
        l1 = tm.disocclusion_layer(dp_, ch, cv, band=max(tm.BAND // s, 4))
        l1 = {k: _crop(v) for k, v in l1.items()}
        l1["hidden"] &= ~covered | (l1["depth"] < room_z * (1 - 0.05))
        ring = (cv2.dilate(_pad(l1["hidden"]).astype(np.uint8), np.ones((3, 3), np.uint8))[:, WRAP_PAD:-WRAP_PAD] > 0) & ~l1["hidden"]
        l1["valid"] = l1["hidden"] | (l1["valid"] & ring)
        l1["depth"] = np.where(l1["hidden"], l1["depth"], d0)
        m1 = cv2.dilate(up(l1["inpaint_mask"]).astype(np.uint8), tm._ellipse(tm.INPAINT_GROW))
        l1["tex"] = _wrap_filter(lambda e: cv2.inpaint(e, cv2.copyMakeBorder(m1, 0, 0, 32, 32, cv2.BORDER_WRAP),
                                                       tm.INPAINT_RADIUS, cv2.INPAINT_TELEA), rgb, 32) if m1.any() else rgb
        layers.append(l1)
        if covered.any():
            visible = np.isfinite(room_z) & (np.abs(d0 - room_z) < tm.PLANE_VISIBLE_TOL * np.where(np.isfinite(room_z), room_z, 1))
            idx_full = cv2.resize(index, (tw, th), interpolation=cv2.INTER_NEAREST)
            l2["tex"] = tm.plane_texture(rgb, up(covered), up(visible), idx_full)
            layers.append(l2)
        stats["hidden_band_cells"], stats["hidden_box_cells"] = int(l1["hidden"].sum()), int(covered.sum())

    n_tiles = len(layers)
    gx = _ext(np.broadcast_to(xs[None, :], (h, w)).astype(np.float64))
    gx[:, -1] = W                                                   # duplicate column: u = 1 (+ half a pixel)
    gy = np.broadcast_to(ys[:, None], (h, w + 1)).astype(np.float64)
    verts, uvs, faces, face_layer = [], [], [], []
    base = 0
    for i, layer in enumerate(layers):
        de = _ext(layer["depth"]).astype(np.float64)
        ve = _ext(layer["valid"])
        ch, cv = layer["cuts"] if i == 0 else tm.depth_cuts(de, None)
        f = tm.grid_faces(ve, ch, cv) + base
        v = (g_dirs * de[..., None]).reshape(-1, 3)
        uv = np.stack([(gx + 0.5) / W, ((gy + 0.5) / H + i) / n_tiles], -1).reshape(-1, 2)
        verts.append(v)
        uvs.append(uv)
        faces.append(f)
        face_layer.append(np.full(len(f), i, np.uint8))
        n_grid = h * (w + 1)
        if i == 0:                                                  # pole caps on the observed surface
            ring_top, ring_bot = np.arange(w + 1) + base, (h - 1) * (w + 1) + np.arange(w + 1) + base
            caps_v, caps_uv, caps_f = [], [], []
            for ring, row, vv in ((ring_top, 0, 0.0), (ring_bot, h - 1, 1.0)):
                pole = v[ring - base].mean(0)
                idx = base + n_grid + len(caps_v) * 0 + sum(len(c) for c in caps_v) + np.arange(w + 1)
                caps_v.append(np.repeat(pole[None], w + 1, 0))
                caps_uv.append(np.stack([(gx[0] + 0.5) / W, np.full(w + 1, (vv + i) / n_tiles)], -1))
                caps_f.append(_cap_faces(ring, idx, ve[row]))
            verts += caps_v
            uvs += caps_uv
            faces += caps_f
            face_layer += [np.full(len(c), 0, np.uint8) for c in caps_f]
            n_grid += 2 * (w + 1)
        base += n_grid
    verts, uvs, faces = np.concatenate(verts), np.concatenate(uvs), np.concatenate(faces)
    face_layer = np.concatenate(face_layer)
    used = np.unique(faces)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    cam_h = frame["d"]
    verts = verts[used]
    heights = cam_h - verts[:, 1]                                   # Y is down in the aligned frame
    tex = np.vstack([cv2.resize(la["tex"], (tw, th)) for la in layers])
    stats.update({"vertices": int(len(used)), "triangles": int(len(faces)),
                  **{f"triangles_layer{i}": int((face_layer == i).sum()) for i in range(n_tiles)}})
    return {"vertices": verts.astype(np.float32), "uvs": uvs[used].astype(np.float32),
            "faces": remap[faces].astype(np.uint32), "texture": tex, "layer": face_layer, "planes": planes,
            "box": binfo, "plan": plan, "R": R, "heights": heights.astype(np.float32), "stats": stats,
            "target_z": float(np.median(dist)), "camera_height": cam_h}


def dollhouse(mesh: dict, cut: float = CEILING_CUT) -> dict:
    """Copy of the room mesh without the ceiling: triangles whose three vertices are all within `cut`
    of the ceiling height are dropped (the ceiling, its lamps, the top pole cap)."""
    ceil_h = mesh["box"]["ceiling_height"] if mesh["box"] else float(np.percentile(mesh["heights"], 98))
    high = mesh["heights"] > ceil_h - cut
    keep = ~high[mesh["faces"]].all(1)
    out = dict(mesh)
    out["faces"], out["layer"] = mesh["faces"][keep], mesh["layer"][keep]
    out["stats"] = {**mesh["stats"], "dollhouse_triangles": int(keep.sum())}
    return out


# --------------------------------------------------------------------------- #
# Ray-cast rendering (metrics and figures)
# --------------------------------------------------------------------------- #
class Raycaster:
    """Exact ray casting against a triangle mesh (Open3D), with texture lookup."""

    def __init__(self, mesh: dict):
        import open3d as o3d
        self.o3d = o3d
        self.mesh = mesh
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(mesh["vertices"].astype(np.float32)),
                                 o3d.core.Tensor(mesh["faces"].astype(np.uint32)))

    def cast(self, origins: np.ndarray, dirs: np.ndarray, color: bool = False):
        """origins / dirs (N, 3). Returns hit mask, distance and (optionally) RGB uint8 colours."""
        rays = np.concatenate([np.broadcast_to(origins, dirs.shape), dirs], -1).astype(np.float32)
        ans = self.scene.cast_rays(self.o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        hit = np.isfinite(t)
        if not color:
            return hit, t, None
        prim = ans["primitive_ids"].numpy()
        b = ans["primitive_uvs"].numpy()
        rgb = np.full((len(t), 3), 255, np.uint8)
        f = self.mesh["faces"][prim[hit]]
        w = np.stack([1 - b[hit, 0] - b[hit, 1], b[hit, 0], b[hit, 1]], 1)
        uv = (self.mesh["uvs"][f] * w[:, :, None]).sum(1)
        tex = self.mesh["texture"]
        x = np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
        y = np.clip((uv[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
        rgb[hit] = tex[y, x]
        return hit, t, rgb


def view_dirs(yaw: float, pitch: float, fov: float, size: int) -> np.ndarray:
    """(size*size, 3) unit ray directions of a square pinhole view in the aligned frame (Y down)."""
    r = eq.perspective_rays(size, fov) @ eq.rotation(yaw, pitch).T
    return (r / np.linalg.norm(r, axis=-1, keepdims=True)).reshape(-1, 3)


def look_at_dirs(eye: np.ndarray, target: np.ndarray, fov: float, size: int) -> np.ndarray:
    """Ray directions of a view at `eye` looking at `target` (aligned frame, Y down, up = -Y)."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(np.array([0.0, -1.0, 0.0]), f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)                                              # image "down"
    rays = eq.perspective_rays(size, fov).reshape(-1, 3)
    out = rays[:, :1] * r + rays[:, 1:2] * d + rays[:, 2:] * f
    return out / np.linalg.norm(out, axis=1, keepdims=True)
