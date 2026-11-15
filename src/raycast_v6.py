"""CPU ray casting of a triangle scene (PLAN_V6 milestone 1): depth, normals, instance ids and colours
for any camera, equirectangular or perspective, without a GPU or a display (Open3D RaycastingScene).

World frame (as src/layout.py): x right, y up, z forward (plan coordinates x, y = world x, z),
floor at y = 0. Camera frame of images (as src/equirect.py): X right, Y down, Z forward.

    Tracer(parts)                parts = [(trimesh.Trimesh, instance_id), ...]; face colours kept for shading
    Tracer.cast(origins, dirs)   -> dict t (distance, inf = miss), normal, inst, face, color
    equirect_rays(C, w, h, yaw)  rays of a level 360° panorama centred at C
    persp_rays(C, yaw, pitch, hfov, w, h)   pinhole rays of a level / pitched perspective camera
    ortho_down_rays(...)         orthographic top-down rays
"""

from __future__ import annotations

import numpy as np

from . import equirect as eq


def _o3d():
    import open3d as o3d
    return o3d


class Tracer:
    def __init__(self, parts: list):
        o3d = _o3d()
        self.scene = o3d.t.geometry.RaycastingScene()
        self.inst_of_geom, self.face_colors, self.vert_colors, self.faces = [], [], [], []
        for mesh, inst in parts:
            v = np.asarray(mesh.vertices, np.float32)
            f = np.asarray(mesh.faces, np.uint32)
            if len(f) == 0:
                continue
            gid = self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
            while len(self.inst_of_geom) <= gid:
                self.inst_of_geom.append(-99)
                self.face_colors.append(None)
                self.vert_colors.append(None)
                self.faces.append(None)
            vc = getattr(mesh, "_vcolors", None)                  # optional per-vertex colours (smooth shading)
            if vc is not None:
                self.vert_colors[gid] = np.asarray(vc, np.float32)
                self.faces[gid] = f.astype(np.int64)
            self.inst_of_geom[gid] = inst
            fc = getattr(mesh.visual, "face_colors", None) if hasattr(mesh, "visual") else None
            self.face_colors[gid] = (np.asarray(fc)[:, :3].astype(np.uint8) if fc is not None and len(fc) == len(f)
                                     else np.full((len(f), 3), 200, np.uint8))
        self.inst_of_geom = np.array(self.inst_of_geom, np.int32)

    def cast(self, origins: np.ndarray, dirs: np.ndarray, colors: bool = True) -> dict:
        o3d = _o3d()
        shape = dirs.shape[:-1]
        o = np.broadcast_to(origins, dirs.shape).reshape(-1, 3)
        d = dirs.reshape(-1, 3)
        rays = np.concatenate([o, d], 1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        gid = ans["geometry_ids"].numpy().astype(np.int64)
        hit = np.isfinite(t)
        gid_c = np.where(hit, gid, 0)
        inst = np.where(hit, self.inst_of_geom[np.clip(gid_c, 0, len(self.inst_of_geom) - 1)], -100)
        face = np.where(hit, ans["primitive_ids"].numpy().astype(np.int64), -1)
        nrm = ans["primitive_normals"].numpy()
        nrm = np.where((np.sum(nrm * d, 1) > 0)[:, None], -nrm, nrm)          # facing the camera
        out = {"t": t.reshape(shape), "normal": nrm.reshape(*shape, 3), "inst": inst.reshape(shape),
               "geom": np.where(hit, gid, -1).reshape(shape), "face": face.reshape(shape)}
        if colors:
            col = np.zeros((len(t), 3), np.uint8)
            uv = ans["primitive_uvs"].numpy()
            for g in np.unique(gid[hit]):
                m = hit & (gid == g)
                if self.vert_colors[g] is not None:                 # barycentric interpolation
                    tri = self.faces[g][face[m]]
                    a, b = uv[m, 0:1], uv[m, 1:2]
                    vc = self.vert_colors[g]
                    col[m] = np.clip(vc[tri[:, 0]] * (1 - a - b) + vc[tri[:, 1]] * a + vc[tri[:, 2]] * b, 0, 255)
                else:
                    col[m] = self.face_colors[g][face[m]]
            out["color"] = col.reshape(*shape, 3)
        return out


def equirect_rays(w: int, h: int, yaw_deg: float = 0.0) -> np.ndarray:
    """World-frame unit directions (h, w, 3) of a level equirectangular image; column w/2 looks along
    world +z rotated by yaw (positive = to the right, towards +x)."""
    dc = eq.pixel_dirs(w, h)
    d = np.stack([dc[..., 0], -dc[..., 1], dc[..., 2]], -1)
    return d @ _yaw(yaw_deg).T


def _yaw(yaw_deg: float) -> np.ndarray:
    a = np.deg2rad(yaw_deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def cam_rotation(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation (world y up): camera looks along world +z rotated by yaw (towards +x
    for positive yaw), pitched up by pitch; camera X = right, Y = down, Z = forward."""
    p = np.deg2rad(pitch_deg)
    # camera frame (X right, Y down, Z fwd) -> world-like (x right, y up, z fwd)
    flip = np.diag([1.0, -1.0, 1.0])
    Rp = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])  # pitch up about x (y up)
    return _yaw(yaw_deg) @ Rp @ flip


def persp_rays(yaw_deg: float, pitch_deg: float, hfov_deg: float, w: int, h: int, shift: float = 0.0) -> np.ndarray:
    """World-frame unit directions (h, w, 3) of a pinhole camera. shift = vertical lens shift as a fraction
    of the image height (positive = window moved down): the optical axis stays level, so verticals stay
    vertical and the horizon level (architectural shift lens), but more of the room below eye height shows."""
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    xs = (np.arange(w) + 0.5 - w / 2) / f
    ys = (np.arange(h) + 0.5 - h / 2 + shift * h) / f
    X, Y = np.meshgrid(xs, ys)
    dc = np.stack([X, Y, np.ones_like(X)], -1)
    dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
    return dc @ cam_rotation(yaw_deg, pitch_deg).T


def focal(hfov_deg: float, w: int) -> float:
    return (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)


def ortho_down(x0: float, x1: float, z0: float, z1: float, px_per_m: float, top: float):
    """Origins (h, w, 3) and directions of an orthographic view straight down; image right = +x,
    image up = +z (plan orientation)."""
    w, h = int(round((x1 - x0) * px_per_m)), int(round((z1 - z0) * px_per_m))
    X, Z = np.meshgrid(x0 + (np.arange(w) + 0.5) / px_per_m, z1 - (np.arange(h) + 0.5) / px_per_m)
    O = np.stack([X, np.full_like(X, top), Z], -1)
    D = np.broadcast_to(np.array([0.0, -1.0, 0.0]), O.shape)
    return O, D


def shade(res: dict, light=(0.35, 0.8, 0.45), ambient: float = 0.45) -> np.ndarray:
    """Lambert-shaded colour image of a cast result (for thumbnails / previews)."""
    L = np.array(light) / np.linalg.norm(light)
    k = ambient + (1 - ambient) * np.clip(np.abs(res["normal"] @ L), 0, 1)
    img = res["color"].astype(np.float32) * k[..., None]
    img[~np.isfinite(res["t"])] = 255
    return np.clip(img, 0, 255).astype(np.uint8)
