"""PLAN_V5 milestone 6: a hires pass for the Quick (single image) mode.

    512 x 512 generation with SD 1.5 + our MMIS LoRA (src/generate_2d.py, unchanged)
 -> bicubic to 1024 x 1024 -> img2img at 1024 with the same LoRA pipeline (weights shared through from_pipe, so no
    second model in memory), strength ~0.35-0.45, the same prompt + a photo suffix: the model re-draws fine texture at
    the higher resolution while the low frequencies (layout, colours) stay
 -> optional Real-ESRGAN x2 (2048 x 2048)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

SUFFIX = ", interior design photo, sharp focus, highly detailed, realistic materials"
NEGATIVE = "blurry, soft, lowres, painting, illustration, deformed, distorted, text, watermark, people"
HIRES, STRENGTH, STEPS, GUIDANCE = 1024, 0.4, 30, 6.5


def img2img_from(pipe):
    """img2img pipeline sharing the txt2img pipeline's modules (no extra VRAM / RAM)."""
    from diffusers import StableDiffusionImg2ImgPipeline
    # built from the components, NOT with from_pipe(): with diffusers 0.42 from_pipe re-casts the modules and the
    # 1024 pass ran > 10 min per image (measured); the plain constructor takes ~4 s
    p = StableDiffusionImg2ImgPipeline(**pipe.components)
    p.set_progress_bar_config(disable=True)
    return p


@torch.inference_mode()
def hires_pass(i2i, img: np.ndarray, prompt: str, seed: int, strength: float = STRENGTH, size: int = HIRES) -> np.ndarray:
    from PIL import Image
    up = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    gen = torch.Generator(i2i.device).manual_seed(int(seed) + 1)
    out = i2i(prompt=prompt + SUFFIX, negative_prompt=NEGATIVE, image=Image.fromarray(up), strength=float(strength),
              num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]
    return np.asarray(out.convert("RGB"))


def generate_hires(pipe, i2i, prompt: str, seed: int, strength: float = STRENGTH, esrgan=None) -> tuple[np.ndarray, dict]:
    """512 generation -> 1024 img2img (-> 2048 Real-ESRGAN). Returns (image, timings)."""
    from .generate_2d import generate
    t0 = time.perf_counter()
    base = generate(pipe, prompt, seed)
    t1 = time.perf_counter()
    hi = hires_pass(i2i, base, prompt, seed, strength)
    t2 = time.perf_counter()
    out = hi
    if esrgan is not None:
        from .shot_enhance import esrgan_shot
        out = esrgan_shot(esrgan, hi, 2 * HIRES, 2 * HIRES)[0]
    return out, {"base_s": t1 - t0, "hires_s": t2 - t1, "esrgan_s": time.perf_counter() - t2, "base": base, "hires": hi}
