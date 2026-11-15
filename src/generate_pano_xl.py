"""PLAN_V5 milestone 1: 360° panoramas from SDXL at 2048 x 1024 (a higher-resolution source than the 1024 x 512
SD 2.1 panorama model used up to v4).

    prompt -> SDXL 1.0 base (fp16, fp16-fix VAE) with a 360° equirectangular adapter:
                 "redmond"  artificialguybr/360Redmond LoRA (bigscience-openrail-m), trigger "360 view"
                 "pgg"      ProGamerGov/sdxl-360-diffusion: a full SDXL UNet fine-tuned on 2048 x 1024 equirectangular
                            images (MIT), trigger "equirectangular 360 view"
           -> seamless wrap: every padded Conv2d of the UNet and the VAE pads CIRCULARLY along x
              (generate_pano.install_circular), so lon = -180° and +180° are neighbours
           -> 2048 x 1024 equirectangular RGB uint8

Memory (8 GB VRAM / 16 GB RAM): fp16 weights, enable_model_cpu_offload (only the module that runs is on the GPU),
VAE slicing; the base UNet is NOT loaded when the "pgg" UNet replaces it (no two UNets in RAM).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import generate_pano as gp
from . import memory as mem

ROOT = Path(__file__).resolve().parents[1]
SDXL_BASE = ROOT / "checkpoints" / "sdxl_base"                    # stabilityai/stable-diffusion-xl-base-1.0 (fp16 parts)
SDXL_VAE = ROOT / "checkpoints" / "sdxl_vae_fp16_fix"             # madebyollin/sdxl-vae-fp16-fix
REDMOND = ROOT / "checkpoints" / "lora_360redmond" / "View360.safetensors"
PGG_UNET = ROOT / "checkpoints" / "sdxl_360_diffusion" / "unet"  # ProGamerGov/sdxl-360-diffusion

XL_W, XL_H = 2048, 1024
XL_STEPS, XL_GUIDANCE = 25, 7.0
TRIGGER = {"redmond": "360 view, 360 panorama", "pgg": "equirectangular 360 view, 360 panorama"}
XL_TEMPLATE = ("{trigger}, interior of a single enclosed {room}, four walls around the camera, view from the centre "
               "of the room, interior design photo, natural light, sharp focus, high quality")
XL_NEGATIVE = gp.ONE_ROOM_NEGATIVE + ", lowres, painting, illustration, cartoon, fisheye circle"


def xl_prompt(room: str, variant: str, template: str = XL_TEMPLATE) -> str:
    return template.format(trigger=TRIGGER[variant], room=room)


def load_xl_pipeline(variant: str = "redmond", lora_scale: float = 1.0, offload: bool = True):
    """SDXL text-to-image pipeline for 360° panoramas. variant: 'redmond' (base + LoRA) or 'pgg' (360 UNet)."""
    from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline,
                           UNet2DConditionModel)
    mem.wait_for_ram(7.5, label="load SDXL")
    vae = AutoencoderKL.from_pretrained(str(SDXL_VAE), dtype=torch.float16)
    kw = {}
    if variant == "pgg":
        kw["unet"] = UNet2DConditionModel.from_pretrained(str(PGG_UNET), dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(str(SDXL_BASE), vae=vae, dtype=torch.float16, variant="fp16",
                                                     use_safetensors=True, add_watermarker=False, **kw)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if variant == "redmond":
        pipe.load_lora_weights(str(REDMOND.parent), weight_name=REDMOND.name, adapter_name="pano360")
        pipe.fuse_lora(lora_scale=float(lora_scale))
        pipe.unload_lora_weights()
    pipe.set_progress_bar_config(disable=True)
    if offload and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    gp.install_circular(pipe.unet, pipe.vae)
    pipe._pano_variant = variant
    return pipe


@torch.inference_mode()
def generate_pano_xl(pipe, room: str, seed: int, steps: int = XL_STEPS, guidance: float = XL_GUIDANCE,
                     width: int = XL_W, height: int = XL_H, prompt: str | None = None, negative: str | None = None,
                     circular: bool = True) -> np.ndarray:
    """One equirectangular panorama (RGB uint8, height x width) for a room description ('a coastal style bedroom')."""
    variant = getattr(pipe, "_pano_variant", "redmond")
    gp.set_circular(circular, pipe.unet, pipe.vae)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    try:
        img = pipe(prompt or xl_prompt(room, variant), negative_prompt=negative or XL_NEGATIVE,
                   num_inference_steps=int(steps), guidance_scale=float(guidance), width=width, height=height,
                   generator=gen, output_type="np").images[0]
    finally:
        gp.set_circular(False, pipe.unet, pipe.vae)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
