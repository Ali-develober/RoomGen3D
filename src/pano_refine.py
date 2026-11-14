"""A sharper panorama by refining perspective views (PLAN_V4 milestone 2).

    texture (Real-ESRGAN x2 of the generated panorama, 2048 x 1024, ~5.7 px/deg)
      -> VIEWS: 26 overlapping pinhole views (75 deg FOV; rings at pitch 0 and +-42 deg with 8 views each,
         one view at each pole), each 512 x 512 (~5.8 px/deg at the centre = no loss from the texture)
      -> optional IMG2IMG on each 512 view (SD 1.5 at its native resolution, optionally with our MMIS LoRA
         fused, strength 0.2-0.35, same prompt, fixed seed per view) to add texture detail. Low-frequency lock:
         only the detail band of the SD output is kept (out = input + (sd - G*sd) - (input - G*input),
         G = Gaussian, sigma LOCK_SIGMA px), so colours and layout cannot drift and neighbouring views keep
         agreeing in their overlap. (img2img on the 1024 views was tried first: 16-21 s per view and it
         SMOOTHED the views - SD 1.5 is out of its training resolution there; see README.)
      -> UPSCALE each view x2 (Real-ESRGAN on an undistorted perspective image) -> 1024 x 1024
         (~11.6 px/deg at the centre; the design shots need ~9-11 px/deg)       [baseline: bicubic x2]
      -> BLEND back into a 4096 x 2048 equirectangular panorama: every output pixel is looked up in each view
         (exact inverse projection), feather weight cos(pi/2 x) cos(pi/2 y) in normalised view coordinates
         (0 at the view border); multi-band (Laplacian pyramid, LEVELS levels) blending with Gaussian-blurred
         weights; the panorama is padded by wrap-around columns before the pyramids, so the left / right seam
         stays continuous; outside a view's frustum its image is replaced by the bicubic base panorama so no
         pyramid sees an artificial edge.
Checks (scripts/eval_pano_refine.py): overlap disagreement of neighbouring views, seam ratio and view-boundary
gradient ratio of the result, edge alignment and DAP depth disagreement against the original.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from . import equirect as eq

VIEW_FOV = 75.0
IN_SIZE = 512
OUT_SIZE = 1024
OUT_W, OUT_H = 4096, 2048
LEVELS = 5
LOCK_SIGMA = 2.0
I2I_STEPS = 30
I2I_GUIDANCE = 5.0
I2I_SUFFIX = ", interior photo, sharp focus, high detail, realistic materials and textures"
I2I_NEGATIVE = "blurry, lowres, soft, distorted, text, watermark, people, frame, border"


def view_grid() -> list[tuple[float, float]]:
    """(yaw, pitch) of the 26 views; coverage of the sphere is checked by coverage()."""
    g = [(45.0 * i, 0.0) for i in range(8)]
    g += [(45.0 * i + 22.5, 42.0) for i in range(8)] + [(45.0 * i + 22.5, -42.0) for i in range(8)]
    return g + [(0.0, 90.0), (0.0, -90.0)]


def view_maps(dirs: np.ndarray, yaw: float, pitch: float, fov: float = VIEW_FOV, size: int = OUT_SIZE):
    """For panorama ray directions (H x W x 3, pano camera frame): pixel coordinates in the view (map_x, map_y,
    float32) and the feather weight (0 outside the frustum)."""
    R = eq.rotation(yaw, pitch)
    dv = dirs @ R                                          # world -> view (rows: d_view = d_world @ R)
    z = dv[..., 2]
    t = np.tan(np.radians(fov) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = dv[..., 0] / (z * t)
        ny = dv[..., 1] / (z * t)
    ok = (z > 1e-6) & (np.abs(nx) < 1) & (np.abs(ny) < 1)
    w = np.where(ok, np.cos(np.pi / 2 * np.clip(nx, -1, 1)) * np.cos(np.pi / 2 * np.clip(ny, -1, 1)), 0.0)
    c = (size - 1) / 2
    f = (size / 2) / t                                     # same intrinsics as equirect.perspective_rays
    mx = np.where(ok, nx * t * f + c, -1).astype(np.float32)
    my = np.where(ok, ny * t * f + c, -1).astype(np.float32)
    return mx, my, w.astype(np.float32)


def coverage(w: int = 512, h: int = 256) -> dict:
    """Minimum over the sphere of the summed / maximum feather weight of the grid (> 0 = fully covered)."""
    d = eq.pixel_dirs(w, h)
    ws = np.stack([view_maps(d, y, p, size=64)[2] for y, p in view_grid()])
    return {"min_sum": float(ws.sum(0).min()), "min_max": float(ws.max(0).min()), "views_per_pixel_mean":
            float((ws > 0).sum(0).mean())}


def extract_views(texture: np.ndarray, size: int = IN_SIZE) -> list[np.ndarray]:
    return [eq.e2p(texture, VIEW_FOV, y, p, size) for y, p in view_grid()]


@torch.inference_mode()
def esrgan_view(net, img: np.ndarray, tile: int = 512, overlap: int = 16) -> np.ndarray:
    """x2 Real-ESRGAN of a perspective view (no wrap padding: a view has real borders)."""
    from .generate_pano import esrgan_upscale
    return esrgan_upscale(net, img, tile=tile, overlap=overlap, wrap_pad=0)


def bicubic_view(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def load_img2img(device, lora_dir=None):
    """SD 1.5 img2img (fp16, DDIM) with the MMIS LoRA fused (lora_dir) or without (None)."""
    from diffusers import StableDiffusionImg2ImgPipeline
    from .generate_2d import load_pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pipe(load_pipeline(lora_dir, device))
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    return pipe


def low_freq_lock(src: np.ndarray, ref: np.ndarray, sigma: float = LOCK_SIGMA) -> np.ndarray:
    """Keep the low frequencies of src and add only the detail band of ref."""
    s, r = src.astype(np.float32), ref.astype(np.float32)
    out = s + (r - cv2.GaussianBlur(r, (0, 0), sigma)) - (s - cv2.GaussianBlur(s, (0, 0), sigma))
    return np.clip(out, 0, 255).astype(np.uint8)


@torch.inference_mode()
def img2img_view(pipe, img: np.ndarray, prompt: str, strength: float, seed: int, lock: bool = True) -> np.ndarray:
    from PIL import Image
    gen = torch.Generator(pipe.device).manual_seed(int(seed))
    out = pipe(prompt=prompt + I2I_SUFFIX, negative_prompt=I2I_NEGATIVE, image=Image.fromarray(img),
               strength=float(strength), num_inference_steps=I2I_STEPS, guidance_scale=I2I_GUIDANCE,
               generator=gen).images[0]
    out = np.asarray(out.convert("RGB"))
    if out.shape[:2] != img.shape[:2]:
        out = cv2.resize(out, img.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    return low_freq_lock(img, out) if lock else out


def _wrap_pad(a: np.ndarray, p: int) -> np.ndarray:
    return cv2.copyMakeBorder(a, 0, 0, p, p, cv2.BORDER_WRAP)


@torch.inference_mode()
def blend(views: list[np.ndarray], base: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H,
          levels: int = LEVELS, multiband: bool = True, device=None) -> tuple[np.ndarray, dict]:
    """GPU version of blend_cpu (same maths; torch grid_sample bicubic and Gaussian pyramids)."""
    import torch.nn.functional as F
    t0 = time.perf_counter()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = torch.from_numpy(eq.pixel_dirs(out_w, out_h).astype(np.float32)).to(dev)
    base_up = torch.from_numpy(cv2.resize(base, (out_w, out_h), interpolation=cv2.INTER_CUBIC)).to(dev)
    base_up = base_up.permute(2, 0, 1)[None].float()
    P = 2 ** levels * 8
    best_w = torch.zeros(out_h, out_w, device=dev)
    best_i = torch.full((out_h, out_w), -1, dtype=torch.int16, device=dev)
    num = den = None
    acc = torch.zeros_like(base_up)
    wsum = torch.zeros(1, 1, out_h, out_w, device=dev)
    t = np.tan(np.radians(VIEW_FOV) / 2)
    for i, ((yaw, pitch), v) in enumerate(zip(view_grid(), views)):
        Rm = torch.from_numpy(eq.rotation(yaw, pitch).astype(np.float32)).to(dev)
        dv = dirs @ Rm
        z = dv[..., 2]
        nx, ny = dv[..., 0] / (z * t), dv[..., 1] / (z * t)
        ok = (z > 1e-6) & (nx.abs() < 1) & (ny.abs() < 1)
        w = torch.where(ok, torch.cos(np.pi / 2 * nx.clamp(-1, 1)) * torch.cos(np.pi / 2 * ny.clamp(-1, 1)),
                        torch.zeros_like(nx))
        S = v.shape[0]
        # pixel index coordinate (centres at integers) -> grid_sample normalised (align_corners=False)
        px, py = nx * t * (S / 2) / t + (S - 1) / 2, ny * t * (S / 2) / t + (S - 1) / 2
        grid = torch.stack([(px + 0.5) / S * 2 - 1, (py + 0.5) / S * 2 - 1], -1).clamp(-1.1, 1.1)
        vt = torch.from_numpy(v).to(dev).permute(2, 0, 1)[None].float()
        warped = F.grid_sample(vt, grid[None], mode="bicubic", padding_mode="border", align_corners=False)
        upd = w > best_w
        best_w = torch.where(upd, w, best_w)
        best_i = torch.where(upd, torch.full_like(best_i, i), best_i)
        w4 = w[None, None]
        if not multiband:
            acc += warped * w4
            wsum += w4
            continue
        img = torch.where(w4 > 0, warped, base_up)
        lap, gw = _lap_pyr_t(_wrap_pad_t(img, P), levels), _gauss_pyr_t(_wrap_pad_t(w4, P), levels)
        if num is None:
            num = [l_ * g for l_, g in zip(lap, gw)]
            den = [g.clone() for g in gw]
        else:
            for k in range(levels + 1):
                num[k] += lap[k] * gw[k]
                den[k] += gw[k]
    if multiband:
        eps = 1e-6
        bl = _lap_pyr_t(_wrap_pad_t(base_up, P), levels)
        lev = [torch.where(d > eps, n / d.clamp(min=eps), b) for n, d, b in zip(num, den, bl)]
        out = _collapse_t(lev)[..., P:-P]
    else:
        out = torch.where(wsum > 1e-6, acc / wsum.clamp(min=1e-6), base_up)
    out = out[0].permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()
    info = {"argmax_view": best_i.cpu().numpy(), "seconds": time.perf_counter() - t0}
    del num, den, dirs
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out, info


_G5 = None


def _blur_t(x):
    import torch.nn.functional as F
    global _G5
    if _G5 is None or _G5.device != x.device:
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
        _G5 = (k[:, None] * k[None, :]).to(x.device)
    c = x.shape[1]
    xp = F.pad(x, (2, 2, 2, 2), mode="replicate")
    return F.conv2d(xp, _G5[None, None].expand(c, 1, 5, 5), groups=c)


def _down_t(x):
    return _blur_t(x)[..., ::2, ::2]


def _up_t(x, size):
    import torch.nn.functional as F
    return _blur_t(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


def _gauss_pyr_t(x, levels):
    g = [x]
    for _ in range(levels):
        g.append(_down_t(g[-1]))
    return g


def _lap_pyr_t(x, levels):
    g = _gauss_pyr_t(x, levels)
    return [g[k] - _up_t(g[k + 1], g[k].shape[-2:]) for k in range(levels)] + [g[-1]]


def _collapse_t(lap):
    a = lap[-1]
    for k in range(len(lap) - 2, -1, -1):
        a = _up_t(a, lap[k].shape[-2:]) + lap[k]
    return a


def _wrap_pad_t(x, p):
    return torch.cat([x[..., -p:], x, x[..., :p]], -1)


def blend_cpu(views: list[np.ndarray], base: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H,
              levels: int = LEVELS, multiband: bool = True) -> tuple[np.ndarray, dict]:
    """Reference CPU implementation (OpenCV pyramids); blend() is the GPU version used by refine()."""
    """Views (OUT_SIZE^2, in view_grid order) -> equirectangular out_w x out_h (see module docstring).
    base: the panorama to fall back on (resized to the output size). Returns (pano, info: per-pixel argmax view
    index for the boundary check)."""
    t0 = time.perf_counter()
    dirs = eq.pixel_dirs(out_w, out_h)
    base_up = cv2.resize(base, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    P = 2 ** levels * 8
    best_w = np.zeros((out_h, out_w), np.float32)
    best_i = np.full((out_h, out_w), -1, np.int16)
    if multiband:
        num, den = None, None
    else:
        acc = np.zeros((out_h, out_w, 3), np.float32)
        wsum = np.zeros((out_h, out_w), np.float32)
    for i, ((yaw, pitch), v) in enumerate(zip(view_grid(), views)):
        mx, my, w = view_maps(dirs, yaw, pitch, VIEW_FOV, v.shape[0])
        warped = cv2.remap(v, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        upd = w > best_w
        best_w[upd], best_i[upd] = w[upd], i
        if not multiband:
            acc += warped * w[..., None]
            wsum += w
            continue
        img = np.where(w[..., None] > 0, warped, base_up)
        img, wp = _wrap_pad(img, P), _wrap_pad(w, P)
        lap, gw = _lap_pyr(img, levels), _gauss_pyr(wp, levels)
        if num is None:
            num = [l_ * g[..., None] for l_, g in zip(lap, gw)]
            den = [g.copy() for g in gw]
        else:
            for k in range(levels + 1):
                num[k] += lap[k] * gw[k][..., None]
                den[k] += gw[k]
    if multiband:
        eps = 1e-6
        bl = _lap_pyr(_wrap_pad(base_up, P), levels)          # fallback where no view has weight
        lev = [np.where(d[..., None] > eps, n / np.maximum(d[..., None], eps), b) for n, d, b in zip(num, den, bl)]
        out = _collapse(lev)[:, P:-P]
    else:
        out = np.where(wsum[..., None] > 1e-6, acc / np.maximum(wsum[..., None], 1e-6), base_up)
    return np.clip(out, 0, 255).astype(np.uint8), {"argmax_view": best_i, "seconds": time.perf_counter() - t0}


def _gauss_pyr(a: np.ndarray, levels: int) -> list[np.ndarray]:
    g = [a]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    return g


def _lap_pyr(a: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gauss_pyr(a, levels)
    lap = [g[k] - cv2.pyrUp(g[k + 1], dstsize=g[k].shape[1::-1]) for k in range(levels)]
    return lap + [g[-1]]


def _collapse(lap: list[np.ndarray]) -> np.ndarray:
    a = lap[-1]
    for k in range(len(lap) - 2, -1, -1):
        a = cv2.pyrUp(a, dstsize=lap[k].shape[1::-1]) + lap[k]
    return a


def refine(texture: np.ndarray, upscaler, i2i=None, prompt: str = "", strength: float = 0.25, seed: int = 0,
           lock: bool = True) -> tuple[np.ndarray, dict]:
    """Whole step. upscaler: an ESRGAN net (load_esrgan) or "bicubic"; i2i: load_img2img(...) or None.
    Returns (4096 x 2048 panorama, info with timings and the refined views)."""
    t = {}
    t0 = time.perf_counter()
    views = extract_views(texture)
    t["extract"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if i2i is not None:
        views = [img2img_view(i2i, v, prompt, strength, seed + k, lock) for k, v in enumerate(views)]
    t["img2img"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    up = [bicubic_view(v) if isinstance(upscaler, str) else esrgan_view(upscaler, v) for v in views]
    t["upscale"] = time.perf_counter() - t0
    pano, info = blend(up, texture)
    t["blend"] = info["seconds"]
    return pano, {"seconds": t, "views": up, "argmax_view": info["argmax_view"]}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def overlap_disagreement(views: list[np.ndarray], w: int = 1024, h: int = 512, min_w: float = 0.2) -> float:
    """Mean absolute difference (grey levels) between two views that both see a panorama pixel with feather
    weight >= min_w, over all such pixel / view pairs (views warped to a w x h equirectangular grid)."""
    d = eq.pixel_dirs(w, h)
    warped, ws = [], []
    for (yaw, pitch), v in zip(view_grid(), views):
        mx, my, wt = view_maps(d, yaw, pitch, VIEW_FOV, v.shape[0])
        g = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY).astype(np.float32)
        warped.append(cv2.remap(g, mx, my, cv2.INTER_AREA if v.shape[0] > 2 * h else cv2.INTER_LINEAR))
        ws.append(wt >= min_w)
    tot, n = 0.0, 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            m = ws[i] & ws[j]
            if m.sum() > 50:
                tot += float(np.abs(warped[i][m] - warped[j][m]).sum())
                n += int(m.sum())
    return tot / max(n, 1)


def boundary_ratio(pano: np.ndarray, argmax_view: np.ndarray) -> float:
    """Mean gradient magnitude on the lines where the dominant view changes, divided by the mean gradient
    elsewhere (1 = blending seams invisible; > 1 = visible seams)."""
    g = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    a = argmax_view.astype(np.int32)
    b = (a != np.roll(a, 1, 1)) | (a != np.roll(a, 1, 0))
    b = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    b[:4], b[-4:] = False, False
    return float(mag[b].mean() / max(mag[~b].mean(), 1e-6))


def edge_f1(ref: np.ndarray, test: np.ndarray, tol: float = 2.0, size: tuple = (1024, 512)) -> float:
    """Structural edges (Canny after a 1.5 px blur at 1024 x 512, so new fine texture does not count) of both;
    F1 of edge pixels matched within tol pixels: does the geometry stay where it was?"""
    def edges(x):
        g = cv2.cvtColor(cv2.resize(x, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        return cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 40, 100) > 0
    e1, e2 = edges(ref), edges(test)
    d1 = cv2.distanceTransform((~e1).astype(np.uint8), cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform((~e2).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((d1[e2] <= tol).mean()) if e2.any() else 0.0
    r = float((d2[e1] <= tol).mean()) if e1.any() else 0.0
    return 2 * p * r / max(p + r, 1e-9)
