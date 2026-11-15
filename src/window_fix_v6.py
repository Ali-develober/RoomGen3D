"""Repair windows the panorama model painted over (PLAN_V6 review round 5).

The depth condition always contains the layout's windows, but SDXL sometimes paints a wall, a wardrobe or a mirror
there (coastal bedroom seed 22, art deco bathroom seed 55: window presence 0.0). For every window of the layout:
    view      perspective camera at the panorama centre aimed at the window centre (pitch allowed), FOV so that the
              window + 60 % margin fits, 768 x 576
    mask      proxy instance ids of that view: window (frame, mullions, outside backdrop) + curtains, dilated 6 px
    inpaint   SD 1.5 + depth ControlNet inpainting (StableDiffusionControlNetInpaintPipeline built from the components
              of the shot pipeline already in memory) on the exact proxy depth, prompt "window with frame and curtains,
              daylight, trees outside" + the style words
    paste     every panorama pixel whose ray falls inside the view and the mask takes the inpainted colour (feathered
              6 px), so the panorama, the design shots (reprojected from it) and the dollhouse all show the window.
Applied only when the panorama's window presence (src/pipeline_v6.window_presence) is below 0.5.

remove_extra_windows: the reverse error, a window painted on a wall the layout leaves blank (kitchen seed 33):
connected window / sky regions of the panorama's segmentation that do not touch any proxy window (dilated) and
cover >= 0.2 % of the panorama are filled with LaMa (the wall continued) in a perspective view aimed at them.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from . import raycast_v6 as R
from . import scene_v6 as S
from . import views_v6 as V

W, H = 768, 576


def window_views(lay: dict) -> list[dict]:
    c = np.asarray(lay["camera"], float)
    out = []
    for o in lay["openings"]:
        if o["kind"] != "window":
            continue
        org, a, n, L = S.wall_frame(lay, o["wall"])
        ctr = org + a * o["c"] + np.array([0, (o["y0"] + o["y1"]) / 2, 0])
        v = ctr - c
        dist = np.linalg.norm(v)
        yaw = float(np.degrees(np.arctan2(v[0], v[2])))
        pitch = float(np.degrees(np.arctan2(v[1], np.hypot(v[0], v[2]))))
        half = max(o["w"] + 0.7, (o["y1"] - o["y0"]) * W / H) * 0.8
        hfov = float(np.clip(np.degrees(2 * np.arctan(half / max(dist, 0.3))), 45, 100))
        out.append({"yaw": yaw, "pitch": pitch, "hfov": hfov, "opening": o})
    return out


def _inpaint_pipe(pipe):
    from diffusers import StableDiffusionControlNetInpaintPipeline
    p = StableDiffusionControlNetInpaintPipeline(**pipe.components)
    p.set_progress_bar_config(disable=True)
    return p


@torch.inference_mode()
def fix_windows(pano: np.ndarray, rd: S.Renderer, lay: dict, pipe, style_words: str, seed: int) -> tuple[np.ndarray, list]:
    """Returns the repaired panorama (same size) and per-window info."""
    from PIL import Image
    from . import shot_render as SR
    ip = _inpaint_pipe(pipe)
    c0 = np.asarray(lay["camera"], float)
    out = pano.astype(np.float32).copy()
    h, w = pano.shape[:2]
    dirs = R.equirect_rays(w, h)                               # world directions of every panorama pixel
    info = []
    for k, v in enumerate(window_views(lay)):
        r = rd.perspective(c0, v["yaw"], v["pitch"], v["hfov"], W, H)
        mask = np.isin(r["inst"], [S.WINDOW, S.CURTAIN_ID]).astype(np.uint8)
        if mask.sum() < 200:
            continue
        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8))
        crop = V.sample_equirect(pano, r["dirs"])
        z = np.where(np.isfinite(r["z"]), r["z"], 10.0)
        prompt = f"a window with a white frame and sheer curtains, daylight, green trees outside, {style_words}"
        img = ip(prompt=prompt + SR.SUFFIX, negative_prompt=SR.NEGATIVE + ", wall, wardrobe, mirror, painting",
                 image=Image.fromarray(crop), mask_image=Image.fromarray(mask * 255),
                 control_image=Image.fromarray(SR.control_image(z)), controlnet_conditioning_scale=1.0,
                 num_inference_steps=30, guidance_scale=7.0, strength=1.0, height=H, width=W,
                 generator=torch.Generator("cpu").manual_seed(seed + 100 + k)).images[0]
        img = np.asarray(img.convert("RGB")).astype(np.float32)
        # paste back: panorama pixels whose ray projects into the view
        Rm = R.cam_rotation(v["yaw"], v["pitch"])
        dc = dirs @ Rm                                          # world -> camera (rows: R^T d)
        zc = dc[..., 2]
        f = R.focal(v["hfov"], W)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = dc[..., 0] / zc * f + W / 2 - 0.5
            py = dc[..., 1] / zc * f + H / 2 - 0.5
        ok = (zc > 0.05) & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)
        mx = np.where(ok, px, 0).astype(np.float32)
        my = np.where(ok, py, 0).astype(np.float32)
        col = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        alpha = cv2.remap(soft, mx, my, cv2.INTER_LINEAR) * ok
        out = out * (1 - alpha[..., None]) + col * alpha[..., None]
        info.append({"wall": v["opening"]["wall"], "yaw": v["yaw"], "hfov": v["hfov"], "mask_px": int(mask.sum())})
    del ip
    return np.clip(out, 0, 255).astype(np.uint8), info


@torch.inference_mode()
def remove_extra_windows(pano: np.ndarray, seg: np.ndarray, rd: S.Renderer, lay: dict, pipe=None, style_words: str = "",
                         seed: int = 0, min_share: float = 0.002, device="cuda") -> tuple[np.ndarray, list]:
    h, w = pano.shape[:2]
    sw, sh = 1024, 512
    sg = cv2.resize(seg.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    win = np.isin(sg, (8, 2)).astype(np.uint8)
    prox = (rd.equirect(lay["camera"], sw, sh)["inst"] == S.WINDOW).astype(np.uint8)
    prox = cv2.dilate(prox, np.ones((21, 21), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(win, connectivity=8)
    c0 = np.asarray(lay["camera"], float)
    dirs_small = R.equirect_rays(sw, sh)
    out = pano.astype(np.float32).copy()
    dirs = R.equirect_rays(w, h)
    ip = None
    info = []
    for k in range(1, n):
        comp = lab == k
        if comp.mean() < min_share or (comp & (prox > 0)).any():
            continue
        d = dirs_small[comp].mean(0)
        d /= np.linalg.norm(d)
        yaw = float(np.degrees(np.arctan2(d[0], d[2])))
        pitch = float(np.degrees(np.arcsin(np.clip(d[1], -1, 1))))
        ang = np.degrees(np.arccos(np.clip(dirs_small[comp] @ d, -1, 1))).max()
        hfov = float(np.clip(2.6 * ang, 40, 100))
        r = rd.perspective(c0, yaw, pitch, hfov, W, H)
        cv_ = r["dirs"]
        u = np.floor((np.arctan2(cv_[..., 0], cv_[..., 2]) + np.pi) / (2 * np.pi) * sw).astype(int) % sw
        v = np.clip(np.floor((np.pi / 2 - np.arcsin(np.clip(cv_[..., 1], -1, 1))) / np.pi * sh).astype(int), 0, sh - 1)
        mask = comp[v, u] & ~np.isin(r["inst"], [S.WINDOW, S.CURTAIN_ID])
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((35, 35), np.uint8))   # incl. the window frame
        if mask.sum() < 200:
            continue
        crop = V.sample_equirect(pano, r["dirs"])
        # LaMa continues the wall around the hole (SD inpainting with a "plain wall" prompt invented a plank board in
        # the kitchen and a pale-blue patch from the palette words in the coastal bedroom, review round 8)
        from . import dollhouse_v5 as dh5
        img = dh5.lama_fill(crop, mask > 0, device)
        img = np.asarray(img).astype(np.float32)
        Rm = R.cam_rotation(yaw, pitch)
        dc = dirs @ Rm
        zc = dc[..., 2]
        f = R.focal(hfov, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = dc[..., 0] / zc * f + W / 2 - 0.5
            py = dc[..., 1] / zc * f + H / 2 - 0.5
        ok = (zc > 0.05) & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)
        mx, my = np.where(ok, px, 0).astype(np.float32), np.where(ok, py, 0).astype(np.float32)
        col = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        alpha = cv2.remap(soft, mx, my, cv2.INTER_LINEAR) * ok
        out = out * (1 - alpha[..., None]) + col * alpha[..., None]
        info.append({"yaw": yaw, "pitch": pitch, "hfov": hfov, "share": float(comp.mean())})
    _ = ip
    return np.clip(out, 0, 255).astype(np.uint8), info
