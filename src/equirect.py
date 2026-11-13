"""Equirectangular (360 x 180 degree) geometry shared by the panorama stages.

Conventions (same camera axes as reconstruct_3d: X right, Y down, Z forward):
    pixel (u, v) of a W x H panorama, centre of the pixel at (u + 0.5, v + 0.5)
    longitude  lon = 2*pi * (u + 0.5) / W - pi       (-pi .. pi, 0 = forward = image centre column)
    latitude   lat = pi/2 - pi * (v + 0.5) / H       (+pi/2 = straight up, -pi/2 = straight down)
    direction  d = (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))
The left and right image borders are the same meridian (lon = -pi = pi), so every filter that
works across the image width must wrap around (cv2.BORDER_WRAP / circular padding).
"""

from __future__ import annotations

import cv2
import numpy as np


def pixel_dirs(w: int, h: int) -> np.ndarray:
    """H x W x 3 unit ray directions of an equirectangular image."""
    lon = 2 * np.pi * (np.arange(w) + 0.5) / w - np.pi
    lat = np.pi / 2 - np.pi * (np.arange(h) + 0.5) / h
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def dirs_to_uv(d: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions (..., 3) -> continuous equirectangular pixel coordinates (u, v) (pixel centres at +0.5)."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
    u = (lon + np.pi) / (2 * np.pi) * w - 0.5
    v = (np.pi / 2 - lat) / np.pi * h - 0.5
    return u, v


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera-to-world rotation: pitch about X (positive = look up), then yaw about Y (positive = turn right)."""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return ry @ rx


def perspective_rays(size: int, fov_deg: float) -> np.ndarray:
    """size x size x 3 pinhole rays (z = 1) of a square view with the given field of view."""
    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2)
    c = (size - 1) / 2
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    return np.stack([(u - c) / f, (v - c) / f, np.ones_like(u, float)], -1)


def e2p(pano: np.ndarray, fov_deg: float, yaw_deg: float, pitch_deg: float = 0.0, size: int = 512,
        interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Perspective view (size x size) cut out of an equirectangular image (any dtype / channel count)."""
    h, w = pano.shape[:2]
    r = perspective_rays(size, fov_deg)
    d = r @ rotation(yaw_deg, pitch_deg).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    u, v = dirs_to_uv(d, w, h)
    # wrap horizontally by padding one copy of the borders (cv2.remap's BORDER_WRAP also wraps v)
    pad = 4
    ext = cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return cv2.remap(ext, (u + pad).astype(np.float32), v.astype(np.float32), interp,
                     borderMode=cv2.BORDER_REPLICATE)


def horizontal_views(pano: np.ndarray, n: int = 4, fov_deg: float = 90.0, pitch_deg: float = 0.0,
                     size: int = 512) -> list[np.ndarray]:
    """n perspective views evenly spaced in yaw (0, 360/n, ...) at the given pitch."""
    return [e2p(pano, fov_deg, 360.0 * i / n, pitch_deg, size) for i in range(n)]


def seam_ratio(pano: np.ndarray) -> float:
    """Seam discontinuity: mean |difference| across the wrap (last column vs first column) divided by
    the mean |difference| between neighbouring columns inside the image. 1 = the seam looks like any
    other column boundary; much larger = a visible seam."""
    g = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY).astype(np.float64) if pano.ndim == 3 else pano.astype(np.float64)
    seam = np.mean(np.abs(g[:, 0] - g[:, -1]))
    inner = np.mean(np.abs(np.diff(g, axis=1)))
    return float(seam / max(inner, 1e-9))


def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness: variance of the Laplacian of the grey image."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
