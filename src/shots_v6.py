"""v6 design shots from cameras placed in the scene (PLAN_V6 milestone 3).

    reproject   for every pixel of the shot camera, the proxy surface point X it sees (ray casting) is looked
                up in the panorama if the panorama centre C0 sees the same point (z-buffer test: the ray
                C0 -> X hits the proxy at |X - C0| within 2 % + 3 cm); otherwise the pixel is a hole
                (surface hidden from C0, e.g. behind furniture). The result is the panorama's content moved
                to the shot camera through the known geometry (no estimated depth).
    fill        holes filled with LaMa (TorchScript big-lama) on the GPU
    re-render   SD 1.5 + our MMIS LoRA (fused) + depth ControlNet (control_v11f1p_sd15_depth) img2img on the
                filled reprojection, conditioned on the EXACT proxy planar depth of the shot; strength s
                (swept in scripts/eval_v6_shots.py); Real-ESRGAN x2 -> 1536 x 1152 last.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from . import scene_v6 as S
from . import shot_render as SR
from . import views_v6 as V

W, H = 768, 576


def reproject(rd: S.Renderer, pano: np.ndarray, c0, cam: dict, w: int = W, h: int = H) -> tuple:
    """(image, visible mask, proxy render) of shot camera `cam` ({pos, yaw, hfov, shift})."""
    r = rd.perspective(cam["pos"], cam["yaw"], 0.0, cam["hfov"], w, h, cam.get("shift", 0.0))
    d = r["dirs"]
    t = r["t"]
    hit = np.isfinite(t)
    X = np.asarray(cam["pos"])[None, None, :] + d * np.where(hit, t, 0)[..., None]
    v = X - np.asarray(c0)[None, None, :]
    dist = np.linalg.norm(v, axis=-1)
    u = v / np.maximum(dist[..., None], 1e-6)
    bk = rd.tracer.cast(np.asarray(c0, float), u, colors=False)
    back = bk["t"]
    vis = hit & np.isfinite(back) & (np.abs(back - dist) < 0.02 * dist + 0.03)
    # outside a window (the backdrop 3 m behind the wall) is far away: use it as an environment map (the shot
    # ray's DIRECTION looked up in the panorama), where the panorama also looks out of a window that way
    win = hit & (r["inst"] == S.WINDOW) & (t > 1.0)
    if win.any():
        through = rd.tracer.cast(np.asarray(c0, float), d, colors=False)["inst"] == S.WINDOW
        sel = win & through
        u = np.where(sel[..., None], d, u)
        vis = vis | sel
    img = V.sample_equirect(pano, u)
    img[~vis] = 0
    return img, vis, r


def fill_holes(img: np.ndarray, vis: np.ndarray, device="cuda") -> np.ndarray:
    from . import dollhouse_v5 as dh
    hole = (~vis).astype(np.uint8)
    if hole.sum() == 0:
        return img
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    return dh.lama_fill(img, hole > 0, device)


def shot_prompt(lay: dict, cam_name: str) -> str:
    """SD 1.5 shot prompt (within 77 tokens incl. the suffix of shot_render)."""
    from . import pano_v6 as PV
    mat, pal = PV.style_words(lay["style"])
    p = f"{lay['style']} style {lay['room']}, {mat}, {pal}, {PV.floor_words(lay)}"
    while PV.n_tokens(p + SR.SUFFIX) > PV.MAX_TOKENS and "," in p:
        p = p.rsplit(",", 1)[0]
    return PV.assert_tokens(p + SR.SUFFIX)[: -len(SR.SUFFIX)]


@torch.inference_mode()
def rerender(pipe, init: np.ndarray, z: np.ndarray, prompt: str, strength: float, seed: int,
             cn_scale: float = SR.CN_SCALE) -> np.ndarray:
    from . import pano_v6 as PV
    PV.assert_tokens(prompt + SR.SUFFIX)
    PV.assert_tokens(SR.NEGATIVE)
    zz = np.where(np.isfinite(z), z, np.nanmax(z[np.isfinite(z)]))
    return SR.rerender(pipe, init, zz, prompt, strength, seed, W, H, cn_scale)


def make_shots(rd: S.Renderer, pano: np.ndarray, lay: dict, cams: dict, pipe, strength: float, seed: int,
               esrgan=None, device="cuda", lama_first: bool = True) -> dict:
    """All shots of a room: {name: {"image", "init", "raw", "vis", "z", "inst", "seconds"}}."""
    out = {}
    c0 = lay["camera"]
    prompt = shot_prompt(lay, "H")
    for k, (name, cam) in enumerate(c for c in cams.items() if c[1] is not None):
        t0 = time.perf_counter()
        raw, vis, r = reproject(rd, pano, c0, cam)
        init = fill_holes(raw, vis, device) if lama_first else raw
        img = rerender(pipe, init, r["z"], prompt, strength, seed + k) if pipe is not None else init
        big = SR.upscale(esrgan, img) if esrgan is not None else img
        out[name] = {"image": big, "base": img, "init": init, "raw": raw, "vis": vis, "z": r["z"], "inst": r["inst"],
                     "seconds": time.perf_counter() - t0, "hole_share": float(1 - vis.mean())}
    return out
