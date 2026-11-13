"""3D reconstruction: pinhole back-projection to a point cloud and Poisson meshing with Open3D.

    depth map (m) + RGB
      -> back-projection with a pinhole camera           (backproject)
      -> flying-pixel removal on the depth grid          (flying_pixel_mask)
      -> statistical outlier removal on the point cloud  (remove_outliers)
      -> normal estimation                               (estimate_normals)
      -> Poisson surface reconstruction + density trim   (poisson_mesh)
      -> export .ply (points) and .glb / .obj (mesh)

Camera model. A generated image has no real camera, so one is ASSUMED: a pinhole camera with a
60 degree horizontal field of view and the principal point at the image centre:
    f = (W / 2) / tan(FOV / 2)          (443 px for a 512-px-wide image)
    X = (u - cx) * Z / f,   Y = (v - cy) * Z / f,   Z = depth
Camera coordinates: X right, Y down, Z forward (into the scene), in metres.
A wrong FOV stretches or squeezes the room along Z but keeps it recognisable.
"""

import cv2
import numpy as np

DEFAULT_FOV_DEG = 60.0        # assumed horizontal field of view of the virtual camera
FLYING_REL_GAP = 0.05         # a point is "flying" if both neighbours are > 5% of its depth away
OUTLIER_NEIGHBORS = 20        # statistical outlier removal: mean distance to the 20 nearest points ...
OUTLIER_STD_RATIO = 2.0       # ... more than 2 std above the average -> removed
POISSON_DEPTH = 9             # octree depth of the Poisson solver (grid 2^9 = 512 cells per side)
DENSITY_QUANTILE = 0.06       # mesh vertices in the 6% least-supported regions are trimmed
MESH_MAX_TRIANGLES = 200_000  # the exported mesh is simplified to at most this many triangles
# Pseudo-metric range used ONLY to visualise a relative (affine-invariant) depth model in 3D:
REL_NEAR, REL_FAR = 1.5, 6.0  # metres assumed for the nearest / farthest part of a room


# --------------------------------------------------------------------------- #
# Camera model and back-projection
# --------------------------------------------------------------------------- #
def intrinsics(width: int, height: int, fov_deg: float = DEFAULT_FOV_DEG) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a pinhole camera with the given horizontal FOV and square pixels."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return f, f, (width - 1) / 2, (height - 1) / 2


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Depth map (m, HxW) -> HxWx3 points in camera coordinates (pinhole model)."""
    h, w = depth.shape
    z = depth.astype(np.float64)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)


def disparity_to_depth(disp: np.ndarray, near: float = REL_NEAR, far: float = REL_FAR) -> np.ndarray:
    """Turn RELATIVE inverse depth into a pseudo-metric depth map (for 3D visualisation only).

    A relative model returns d with  true inverse depth = s * d + t  for an unknown scale s and
    shift t. Without ground truth both must be guessed: the 2nd / 98th percentile of d are mapped
    to 1/far and 1/near. A wrong shift t is not a uniform scaling of the scene: it bends planes
    and changes angles, which is the geometric distortion this model has in 3D.
    """
    lo, hi = np.percentile(disp, [2, 98])
    d = np.clip((disp - lo) / max(hi - lo, 1e-8), 0, None)
    return (1.0 / (1.0 / far + d * (1.0 / near - 1.0 / far))).astype(np.float32)


def flying_pixel_mask(depth: np.ndarray, valid: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      rel_gap: float = FLYING_REL_GAP) -> np.ndarray:
    """Flying pixels: 3D points hanging in empty space between a foreground and a background surface.

    A depth map that blurs an object boundary puts intermediate depths there; back-projected,
    these become a "curtain" of points joining the object to the wall behind it. A point counts
    as flying if, along its image row or its column, the 3D points on BOTH sides are farther
    away than rel_gap x its depth (15 cm at 3 m) and one is nearer, the other farther.
    A perfectly sharp step has none (each side has a close neighbour on its own surface).
    """
    pts, z = backproject(depth, fx, fy, cx, cy), depth.astype(np.float64)
    fly = np.zeros(depth.shape, bool)
    for ax in (0, 1):
        p, zz, m = np.moveaxis(pts, ax, 0), np.moveaxis(z, ax, 0), np.moveaxis(valid, ax, 0)
        gap = np.linalg.norm(p[1:] - p[:-1], axis=-1)           # 3D distance between neighbours
        lim = rel_gap * zz[1:-1]
        between = (zz[2:] - zz[1:-1]) * (zz[1:-1] - zz[:-2]) > 0
        np.moveaxis(fly, ax, 0)[1:-1] |= (gap[:-1] > lim) & (gap[1:] > lim) & between & m[2:] & m[:-2]
    return fly & valid


def depth_to_points(depth: np.ndarray, rgb: np.ndarray, fov_deg: float = DEFAULT_FOV_DEG,
                    remove_flying: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth (m) + RGB uint8 -> (points Nx3, colours Nx3 in 0..1, keep mask HxW).

    With remove_flying the flying pixels (dilated by one pixel, to also take the half-mixed
    pixels next to them) are dropped before the points are returned.
    """
    h, w = depth.shape
    k = intrinsics(w, h, fov_deg)
    keep = np.isfinite(depth) & (depth > 0)
    if remove_flying:
        fly = flying_pixel_mask(depth, keep, *k)
        keep &= ~(cv2.dilate(fly.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    pts = backproject(depth, *k)
    return pts[keep], rgb[keep].astype(np.float64) / 255.0, keep


# --------------------------------------------------------------------------- #
# Point cloud cleaning, normals, meshing (Open3D)
# --------------------------------------------------------------------------- #
def to_o3d(points: np.ndarray, colors: np.ndarray):
    """Nx3 points + Nx3 colours (0..1) -> open3d.geometry.PointCloud."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    return pcd


def remove_outliers(pcd, nb_neighbors: int = OUTLIER_NEIGHBORS, std_ratio: float = OUTLIER_STD_RATIO):
    """Statistical outlier removal. Returns (clean cloud, number of removed points).

    For every point the mean distance to its nb_neighbors nearest neighbours is computed; points
    whose value is more than std_ratio standard deviations above the average over the whole
    cloud are isolated specks and are removed.
    """
    clean, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean, len(pcd.points) - len(idx)


def estimate_normals(pcd, k: int = 30):
    """Surface normals by a plane fit (PCA) to the k nearest neighbours, turned to face the camera.

    PCA gives the normal direction but not its sign. Every point was seen from the camera at the
    origin, so the correct side is the one facing the origin. Poisson needs this consistency.
    """
    import open3d as o3d
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


def poisson_mesh(pcd, depth: int = POISSON_DEPTH, density_quantile: float = DENSITY_QUANTILE,
                 max_triangles: int | None = MESH_MAX_TRIANGLES):
    """Screened Poisson surface reconstruction of a point cloud that has oriented normals.

    Poisson fits a smooth indicator function whose gradient matches the normals and extracts its
    iso-surface, so it closes the surface even where there are no points ("bubbles" behind the
    visible room). Each mesh vertex comes with a density = how many input points support it;
    vertices below the density_quantile are trimmed, and the mesh is cropped to the bounding box
    of the points. Finally it is simplified (quadric decimation) to at most max_triangles.
    """
    import open3d as o3d
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    if max_triangles and len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def reconstruct(depth: np.ndarray, rgb: np.ndarray, fov_deg: float = DEFAULT_FOV_DEG, clean: bool = True,
                mesh: bool = True) -> dict:
    """Depth (m) + RGB -> {"pcd", "mesh", "stats"}; clean=False skips flying-pixel and outlier removal."""
    pts, cols, keep = depth_to_points(depth, rgb, fov_deg, remove_flying=clean)
    pcd = to_o3d(pts, cols)
    stats = {"points_raw": int(depth.size), "flying_removed": int(depth.size - keep.sum()), "outliers_removed": 0}
    if clean:
        pcd, stats["outliers_removed"] = remove_outliers(pcd)
    stats["points"] = len(pcd.points)
    out = {"pcd": pcd, "mesh": None, "stats": stats}
    if mesh:
        estimate_normals(pcd)
        out["mesh"] = poisson_mesh(pcd)
        stats["vertices"], stats["triangles"] = len(out["mesh"].vertices), len(out["mesh"].triangles)
    return out


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])    # camera (Y down, Z forward) -> glTF / viewer (Y up, Z towards viewer)


def export_pointcloud(pcd, path) -> None:
    """Write the coloured point cloud as .ply (Y up, so it opens upright in MeshLab / Blender)."""
    import copy

    import open3d as o3d
    o3d.io.write_point_cloud(str(path), copy.deepcopy(pcd).transform(_TO_GL))


def export_mesh(mesh, path) -> None:
    """Write the vertex-coloured mesh as .glb or .obj (by extension), Y up."""
    import copy

    import open3d as o3d
    o3d.io.write_triangle_mesh(str(path), copy.deepcopy(mesh).transform(_TO_GL))


# --------------------------------------------------------------------------- #
# Simple software renderer (figures only; no OpenGL needed)
# --------------------------------------------------------------------------- #
VIEWS = {"front": (0.0, 0.0), "side": (-40.0, 18.0), "top": (0.0, 90.0)}   # name -> (yaw, pitch) in degrees


def render_points(points: np.ndarray, colors: np.ndarray, view: str = "side", size: int = 512,
                  normals: np.ndarray | None = None, splat: int = 2, drop_ceiling: float = 0.12,
                  bg: int = 255) -> np.ndarray:
    """Orthographic render of coloured 3D points (camera coordinates) from a named view. uint8 image.

    The scene is rotated about its centre by the view's yaw (around the vertical axis) and pitch
    (tilt down), projected orthographically and painted far-to-near, so near points cover far
    ones. Orthographic = no perspective, so straight walls and right angles are easy to judge.
    "top" looks straight down; the highest `drop_ceiling` share of the points is hidden so the
    ceiling does not cover the room. With `normals` the colours get simple Lambert shading.
    """
    yaw, pitch = np.radians(VIEWS[view])
    p = np.asarray(points, np.float64)
    c = np.asarray(colors, np.float64)
    n = None if normals is None else np.asarray(normals, np.float64)
    if view == "top" and drop_ceiling > 0:
        sel = p[:, 1] > np.quantile(p[:, 1], drop_ceiling)        # Y is down: small Y = high
        p, c, n = p[sel], c[sel], (None if n is None else n[sel])
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rot = rx @ ry
    q = (p - (lo + hi) / 2) @ rot.T
    if n is not None:                                             # light from the viewer, slightly above
        light = np.array([0.25, -0.45, -0.85])
        shade = 0.45 + 0.55 * np.abs((n @ rot.T) @ (light / np.linalg.norm(light)))
        c = np.clip(c * shade[:, None], 0, 1)
    qlo, qhi = np.percentile(q[:, :2], 0.5, axis=0), np.percentile(q[:, :2], 99.5, axis=0)
    scale = (size * 0.92) / max((qhi - qlo).max(), 1e-9)
    uv = (q[:, :2] - (qlo + qhi) / 2) * scale + size / 2
    order = np.argsort(-q[:, 2])                                  # far first; later writes win
    u0, v0 = np.round(uv[order, 0]).astype(int), np.round(uv[order, 1]).astype(int)
    col = (c[order] * 255).astype(np.uint8)
    off = np.arange(splat) - splat // 2
    du, dv = [a.ravel() for a in np.meshgrid(off, off)]
    u, v = (u0[:, None] + du[None]).ravel(), (v0[:, None] + dv[None]).ravel()   # a point's splat stays contiguous
    col = np.repeat(col, len(du), axis=0)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    img = np.full((size, size, 3), bg, np.uint8)
    img[v[ok], u[ok]] = col[ok]
    return img


def render_pcd(pcd, view: str = "side", size: int = 512) -> np.ndarray:
    """Render an Open3D point cloud."""
    return render_points(np.asarray(pcd.points), np.asarray(pcd.colors), view, size)


def render_mesh(mesh, view: str = "side", size: int = 512, n_samples: int = 900_000) -> np.ndarray:
    """Render a mesh by sampling its surface densely and shading the samples with the surface normals."""
    s = mesh.sample_points_uniformly(number_of_points=n_samples, use_triangle_normal=True)
    return render_points(np.asarray(s.points), np.asarray(s.colors), view, size, normals=np.asarray(s.normals))


# --------------------------------------------------------------------------- #
# Geometry checks without ground truth (used to compare depth models on generated rooms)
# --------------------------------------------------------------------------- #
def fit_plane(points: np.ndarray, threshold: float = 0.03, iters: int = 300) -> dict | None:
    """RANSAC plane fit (Open3D). Returns unit normal, inlier share and RMS distance of all points (m)."""
    import open3d as o3d
    if len(points) < 500:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.utility.random.seed(0)
    (a, b, c, d), inl = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=iters)
    norm = float(np.linalg.norm([a, b, c]))
    dist = np.abs(points @ np.array([a, b, c]) + d) / norm
    return {"normal": np.array([a, b, c]) / norm, "inlier_share": len(inl) / len(points),
            "rms": float(np.sqrt(np.mean(dist ** 2)))}


def room_plane_stats(depth: np.ndarray, masks: dict, fov_deg: float = DEFAULT_FOV_DEG) -> dict:
    """How room-like the back-projected geometry is, from the segmentation masks.

    floor_rms_pct / wall_rms_pct: RMS distance of the floor / wall points from their best plane,
        in % of the median scene depth (a real floor is flat -> small).
    floor_wall_angle: angle between the floor plane and the dominant wall plane (a real room: 90).
    floor_tilt: angle between the floor normal and the image's vertical axis (a level camera: ~0-25).
    """
    h, w = depth.shape
    pts = backproject(depth, *intrinsics(w, h, fov_deg))
    scale = float(np.median(depth))
    out = {"floor_rms_pct": np.nan, "wall_rms_pct": np.nan, "floor_wall_angle": np.nan, "floor_tilt": np.nan}
    planes = {}
    for g in ("floor", "wall"):
        m = cv2.erode(masks[g].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # stay off the mask borders
        planes[g] = fit_plane(pts[m], threshold=0.01 * scale)
        if planes[g]:
            out[f"{g}_rms_pct"] = 100 * planes[g]["rms"] / scale
    if planes["floor"]:
        out["floor_tilt"] = float(np.degrees(np.arccos(abs(planes["floor"]["normal"][1]))))
        if planes["wall"]:
            cosang = abs(float(planes["floor"]["normal"] @ planes["wall"]["normal"]))
            out["floor_wall_angle"] = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
    return out
