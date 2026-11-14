"""Viewer helpers for the v3 room (PLAN_V3 milestone 6).

    single-sided export   textured_mesh.export_glb(double_sided=False): the shell's quads face inwards and the
                          solids outwards, so in the dollhouse viewer the walls between the viewer and the room
                          are culled automatically (back faces), the far walls stay; the ceiling is not exported
    background colour     the inside viewer's background is the room's mean colour, so any remaining crack
                          between surfaces shows a matching colour instead of white
    top-down view         orthographic ray casting straight down onto the dollhouse mesh, in the orientation
                          of the floor plan (longer side horizontal), to show that plan and model agree
"""

from __future__ import annotations

import numpy as np

from . import room_mesh as rm


def mean_color(mesh: dict) -> str:
    """Hex colour of the mean texture colour of the shell (walls, floor, ceiling) part of a v3 mesh."""
    tex = mesh["texture"].reshape(-1, 3)
    c = np.median(tex[:: max(1, len(tex) // 200_000)], axis=0)
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in c])


def topdown(mesh: dict, box: dict, px_per_m: float = 60.0, margin: float = 0.25) -> np.ndarray:
    """Orthographic top view of a mesh (frame A vertices) over the room box, oriented like plan_v3:
    image right = plan x, image up = plan y, the longer side horizontal."""
    (x0, x1), (z0, z1) = box["x"], box["z"]
    rot = (z1 - z0) > (x1 - x0)
    # plan extent (drawing coords): rot -> (X, Y) = (z, -x)
    if rot:
        X0, X1, Y0, Y1 = z0, z1, -x1, -x0
    else:
        X0, X1, Y0, Y1 = x0, x1, z0, z1
    X0, X1, Y0, Y1 = X0 - margin, X1 + margin, Y0 - margin, Y1 + margin
    w, h = int((X1 - X0) * px_per_m), int((Y1 - Y0) * px_per_m)
    X, Y = np.meshgrid(X0 + (np.arange(w) + 0.5) / px_per_m, Y1 - (np.arange(h) + 0.5) / px_per_m)
    if rot:
        xb, zb = -Y, X
    else:
        xb, zb = X, Y
    top = box["ceil_y"] - 1.0                                          # above the ceiling (Y down)
    PB = np.stack([xb, np.full_like(xb, top), zb], -1).reshape(-1, 3)
    PA = PB @ box["Q"].T
    dA = np.broadcast_to(np.array([0.0, 1.0, 0.0]) @ box["Q"].T, PA.shape)
    _, _, col = _cast(mesh, PA, dA)
    return col.reshape(h, w, 3)


def _cast(mesh: dict, origins: np.ndarray, dirs: np.ndarray):
    """Ray casting with one origin per ray (room_mesh.Raycaster.cast takes one origin)."""
    rc = rm.Raycaster(mesh)
    o3d = rc.o3d
    rays = np.concatenate([origins, dirs], -1).astype(np.float32)
    ans = rc.scene.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy()
    hit = np.isfinite(t)
    prim = ans["primitive_ids"].numpy()
    b = ans["primitive_uvs"].numpy()
    rgb = np.full((len(t), 3), 255, np.uint8)
    f = mesh["faces"][prim[hit]]
    wgt = np.stack([1 - b[hit, 0] - b[hit, 1], b[hit, 0], b[hit, 1]], 1)
    uv = (mesh["uvs"][f] * wgt[:, :, None]).sum(1)
    tex = mesh["texture"]
    x = np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
    y = np.clip((uv[:, 1] * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
    rgb[hit] = tex[y, x]
    return hit, t, rgb
