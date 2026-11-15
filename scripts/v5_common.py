"""Shared definitions of the PLAN_V5 evaluation scripts: room sets, perspective views, image metrics.

Rooms (fixed in PLAN_V5.md before any run):
    dev   a {scandinavian, farmhouse} style {bedroom, living room, kitchen, dining room, bathroom}, seed 701
    test  a {industrial, bohemian} style {...}, seed 701      (held-out: reported only)
    demo  the six demo prompts and seeds (src/pipeline.DEMO_PROMPTS)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import design_shots as ds  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import pipeline as pl  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
ROOM_TYPES = ["bedroom", "living room", "kitchen", "dining room", "bathroom"]
STYLES = {"scandinavian": "dev", "farmhouse": "dev", "industrial": "test", "bohemian": "test"}
SEED = 701
VIEW_YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
VIEW_W, VIEW_H, VIEW_FOV = 768, 576, 90.0


def rooms(sets=("dev", "test", "demo")) -> pd.DataFrame:
    rows = []
    for style, split in STYLES.items():
        for r in ROOM_TYPES:
            p = f"a {style} style {r}"
            rows.append({"id": f"{split}_{pl.slug(p, SEED)}", "prompt": p, "seed": SEED, "set": split, "room": r})
    for p, s in pl.DEMO_PROMPTS:
        rows.append({"id": f"demo_{pl.slug(p, s)}", "prompt": p, "seed": s, "set": "demo",
                     "room": ds.room_type_of(p) or ""})
    df = pd.DataFrame(rows)
    return df[df["set"].isin(sets)].reset_index(drop=True)


def rgb(p) -> np.ndarray:
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save_rgb(p, img: np.ndarray, q: int | None = None) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    args = [cv2.IMWRITE_JPEG_QUALITY, q] if q else []
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), args)


def level_views(pano: np.ndarray, w: int = VIEW_W, h: int = VIEW_H, fov: float = VIEW_FOV, yaws=VIEW_YAWS):
    """Level perspective views of the panorama camera frame (no room box needed), yaw in degrees."""
    out = []
    for y in yaws:
        M = ds.view_to_B(y).T                               # camera frame = frame B with identity rotations
        out.append(ds.render_view(pano, M, w, h, fov))
    return out


def brisque(imgs: list[np.ndarray]) -> float:
    import piq
    import torch
    vals = []
    for im in imgs:
        x = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255)
        vals.append(float(piq.brisque(x, data_range=1.0)))
    return float(np.mean(vals))


def sharpness(imgs: list[np.ndarray]) -> float:
    return float(np.mean([eq.laplacian_variance(im) for im in imgs]))


def write_json(p, obj) -> None:
    Path(p).write_text(json.dumps(obj, indent=1, default=float))


def ade_id2label() -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    return {int(k): v for k, v in cfg.id2label.items()}
