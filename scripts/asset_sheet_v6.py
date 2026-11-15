"""Contact sheet of every catalogue model seen from 4 sides (front = +z after normalisation, i.e. the
camera on the +z side looking towards -z sees the model's front). Used to set each model's `front` angle
in src/assets_v6.CATALOG by eye.  ->  results/figures/v6_assets_sheet.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import assets_v6 as A          # noqa: E402
from src import raycast_v6 as R         # noqa: E402

S = 160


def view(mesh, side: str) -> np.ndarray:
    """Orthographic view from `side` (+z = front camera, -z, +x, -x)."""
    ext = np.abs(mesh.vertices).max(0)
    r = max(ext[0], ext[2], mesh.extents[1] / 2) * 1.15
    cy = mesh.extents[1] / 2
    u = (np.arange(S) + 0.5) / S * 2 * r - r
    U, V = np.meshgrid(u, cy + r - (np.arange(S) + 0.5) / S * 2 * r)
    far = 5.0
    if side == "+z":
        O, d = np.stack([-U, V, np.full_like(U, far)], -1), (0, 0, -1)     # looking -z: image right = -x
    elif side == "-z":
        O, d = np.stack([U, V, np.full_like(U, -far)], -1), (0, 0, 1)
    elif side == "+x":
        O, d = np.stack([np.full_like(U, far), V, U], -1), (-1, 0, 0)
    else:
        O, d = np.stack([np.full_like(U, -far), V, -U], -1), (1, 0, 0)
    D = np.broadcast_to(np.array(d, float), O.shape)
    res = R.Tracer([(mesh, 0)]).cast(O, D)
    return R.shade(res)


def main():
    rows = []
    for cls, opts in A.CATALOG.items():
        for src, mid, front, _ in opts:
            m = A.load_asset(src, mid, front)
            tiles = [view(m, s) for s in ("+z", "-z", "+x", "-x")]
            row = Image.fromarray(np.concatenate(tiles, 1))
            ImageDraw.Draw(row).text((4, 4), f"{cls}: {mid} f={front}", fill=(255, 0, 0))
            rows.append(np.asarray(row))
            print(cls, mid, np.round(m.extents, 2), flush=True)
    sheet = np.concatenate(rows, 0)
    out = Path(__file__).resolve().parents[1] / "results" / "figures" / "v6_assets_sheet.png"
    half = len(rows) // 2 + len(rows) % 2
    a, b = np.concatenate(rows[:half], 0), np.concatenate(rows[half:], 0)
    b = np.pad(b, ((0, a.shape[0] - b.shape[0]), (0, 0), (0, 0)), constant_values=255)
    Image.fromarray(np.concatenate([a, np.full((a.shape[0], 10, 3), 0, np.uint8), b], 1)).save(out)
    print(out, sheet.shape)


if __name__ == "__main__":
    main()
