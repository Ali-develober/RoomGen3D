"""Perspective views of a v6 panorama that are pixel-aligned with the proxy renders (PLAN_V6 milestones 2-3).

The v6 panorama is generated on the proxy equirectangular render of src/scene_v6.Renderer.equirect (centre C,
yaw 0), so a world direction d maps to the panorama pixel of the camera-frame direction (d_x, -d_y, d_z).
sample_equirect(pano, dirs) samples it bilinearly (wrap in x) for any set of world rays, e.g. the rays of a
perspective camera at C: the result is the generated view, and Renderer.perspective(C, ...) with the same
rays is its exact proxy (depth, normals, instance ids).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import equirect as eq
from . import raycast_v6 as R


def sample_equirect(pano: np.ndarray, dirs: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Sample an equirectangular image (H x W x C) along world directions (..., 3)."""
    shp = dirs.shape[:-1]
    if len(shp) != 2 or max(shp) >= 30000:                    # cv2.remap needs 2-D maps < 32767 per side
        flat = dirs.reshape(-1, 3)
        n = len(flat)
        cols = 4096
        rows = -(-n // cols)
        pad = np.zeros((rows * cols - n, 3))
        pad[:, 2] = 1
        out = sample_equirect(pano, np.concatenate([flat, pad]).reshape(rows, cols, 3), interp)
        return out.reshape(rows * cols, *pano.shape[2:])[:n].reshape(*shp, *pano.shape[2:])
    h, w = pano.shape[:2]
    dc = np.stack([dirs[..., 0], -dirs[..., 1], dirs[..., 2]], -1)
    dc = dc / np.linalg.norm(dc, axis=-1, keepdims=True)
    u, v = eq.dirs_to_uv(dc, w, h)
    pad = 2
    src = cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return cv2.remap(src, (u + pad).astype(np.float32), v.astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)


def level_views(pano: np.ndarray, yaws=(0, 45, 90, 135, 180, 225, 270, 315), fov: float = 90.0, w: int = 768,
                h: int = 576) -> list[np.ndarray]:
    """Level perspective views of the panorama (camera at its centre)."""
    return [sample_equirect(pano, R.persp_rays(y, 0.0, fov, w, h)) for y in yaws]
