"""PLAN_V5 milestone 3: real detail in the 2D design shots by RE-RENDERING each shot, not only upscaling it.

v4 cut the shot out of the panorama and ran Real-ESRGAN on it: edges got sharper, but no new detail appeared,
because the panorama itself has little (painterly surfaces). Here each shot is re-generated at a resolution where
SD 1.5 is at home, constrained to keep the room as it is:

    crop (perspective shot, level)         -> 768 x 576 init image
    crop depth (panorama distance rendered -> z-depth -> inverse depth, 2-98 % normalised, near = white)
                                           -> ControlNet depth (lllyasviel/control_v11f1p_sd15_depth)
    SD 1.5 img2img + our MMIS LoRA (fused) + ControlNet, strength s, the room prompt + a photo suffix
                                           -> Real-ESRGAN x2 -> 1536 x 1152

The depth condition keeps the geometry (walls, furniture outlines) where it is even at higher strength, so the
strength can be raised until the diffusion model repaints the surfaces with real texture. The structure guard in the
evaluation (scripts/eval_v5_shots.py) checks that the edges and the depth of the shot do not move.

SDXL variant (milestone 1 adopted SDXL): SDXL base img2img at 1024 x 768 (no ControlNet download; the structure guard
decides the strength).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import design_shots as ds
from . import memory as mem

ROOT = Path(__file__).resolve().parents[1]
CONTROLNET_DEPTH = ROOT / "checkpoints" / "controlnet_depth"
LORA = ROOT / "checkpoints" / "lora_mmis" / "step_03000"
W, H = 768, 576
OUT_W, OUT_H = 1536, 1152
STEPS, GUIDANCE, CN_SCALE = 30, 6.0, 0.8
SUFFIX = ", interior design photograph, wide angle, sharp focus, highly detailed, realistic materials, natural light"
NEGATIVE = ("blurry, soft, painting, illustration, cartoon, lowres, jpeg artifacts, deformed, distorted, warped, "
            "text, watermark, people, close-up")


def load_cn_img2img(device, lora: bool = True):
    """SD 1.5 fp16 + depth ControlNet img2img; the MMIS LoRA fused into the UNet when lora=True."""
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline
    from .generate_2d import BASE_MODEL
    mem.wait_for_ram(3.0, label="load SD1.5 ControlNet img2img")
    cn = ControlNetModel.from_pretrained(str(CONTROLNET_DEPTH), dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(BASE_MODEL, controlnet=cn, dtype=torch.float16,
                                                                    variant="fp16", safety_checker=None,
                                                                    requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if lora:
        pipe.load_lora_weights(str(LORA))
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def load_xl_img2img(device=None):
    """SDXL base img2img (fp16, fp16-fix VAE), model CPU offload."""
    from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
    from .generate_pano_xl import SDXL_BASE, SDXL_VAE
    mem.wait_for_ram(7.5, label="load SDXL img2img")
    vae = AutoencoderKL.from_pretrained(str(SDXL_VAE), dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(SDXL_BASE), vae=vae, dtype=torch.float16,
                                                            variant="fp16", use_safetensors=True,
                                                            add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    return pipe


def shot_depth(dist: np.ndarray, box: dict, yaw: float, fov: float, w: int = W, h: int = H) -> np.ndarray:
    """z-depth (m) of a level design shot, from the panorama distance map (bilinear)."""
    M = ds.view_to_cam(box, yaw)
    r = ds.render_view(dist.astype(np.float32), M, w, h, fov, cv2.INTER_LINEAR)
    rays, _ = ds.view_rays(w, h, fov)
    return r / np.linalg.norm(rays, axis=-1)                       # radial distance -> z-depth


def control_image(z: np.ndarray) -> np.ndarray:
    """Inverse depth, 2-98 % normalised, near = white (the MiDaS-style input the depth ControlNet was trained on)."""
    inv = 1.0 / np.maximum(z, 0.1)
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    g = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat((g * 255).astype(np.uint8)[..., None], 3, -1)


@torch.inference_mode()
def rerender(pipe, crop: np.ndarray, z: np.ndarray | None, prompt: str, strength: float, seed: int,
             w: int = W, h: int = H, cn_scale: float = CN_SCALE) -> np.ndarray:
    """One re-rendered shot at w x h (RGB uint8). z: crop z-depth (ControlNet pipelines only)."""
    from PIL import Image
    init = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    kw = dict(prompt=prompt + SUFFIX, negative_prompt=NEGATIVE, image=Image.fromarray(init), strength=float(strength),
              num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen)
    if hasattr(pipe, "controlnet"):
        cz = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
        kw.update(control_image=Image.fromarray(control_image(cz)), controlnet_conditioning_scale=float(cn_scale),
                  height=h, width=w)
    out = pipe(**kw).images[0]
    out = np.asarray(out.convert("RGB"))
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def upscale(net, img: np.ndarray, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    from .shot_enhance import esrgan_shot
    return esrgan_shot(net, img, out_w, out_h)[0]


def rerender_shots(pipe, shots: list[dict], dist: np.ndarray, box: dict, prompt: str, strength: float, seed: int,
                   size: tuple[int, int] = (W, H)) -> list[np.ndarray]:
    """Re-render every shot ({'image', 'yaw', 'fov'}); returns the images at `size` (before upscaling)."""
    out = []
    for k, s in enumerate(shots):
        z = shot_depth(dist, box, s["yaw"], s["fov"], *size)
        t0 = time.perf_counter()
        out.append(rerender(pipe, s["raw"] if "raw" in s else s["image"], z, prompt, strength, seed + k, *size))
        s["rerender_s"] = time.perf_counter() - t0
    return out
