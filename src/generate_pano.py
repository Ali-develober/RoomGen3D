"""360 degree panorama generation: SD v1.5 + panorama LoRA (+ our MMIS LoRA), 1024 x 512 equirectangular.

    prompt  ->  "qxj, 360 panorama, interior, <room description>, photo"
            ->  SD v1.5 with two LoRA adapters:
                  pano  ProGamerGov/360-Diffusion-LoRA-sd-v1-5 (trigger word "qxj"), weight 1
                  mmis  our interior LoRA (checkpoints/lora_mmis/step_03000), weight w (swept)
            ->  seamless wrap: every padded convolution of the UNet and the VAE decoder pads
                CIRCULARLY along x (and with zeros along y), so the left and right borders of the
                latent / image are neighbours, exactly like lon = -180 and lon = +180 deg.
                Optionally only from a fraction of the denoising steps onwards.
            ->  equirectangular RGB uint8 (512 x 1024)

Super-resolution to 2048 x 1024: Real-ESRGAN x2 (RRDBNet, own minimal implementation so that
basicsr is not needed) run in overlapping tiles on a horizontally WRAP-padded image, compared
with bicubic upsampling + unsharp masking.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generate_2d import BASE_MODEL, DEFAULT_LORA

ROOT = Path(__file__).resolve().parents[1]
PANO_LORA = ROOT / "checkpoints" / "pano_lora" / "360Diffusion_v1.safetensors"
ALT_MODEL = ROOT / "checkpoints" / "pano_sd_base" / "sd-base"     # archerfmy0831/sd-t2i-360panoimage (SD 2.1 base fine-tune)
ALT_TEMPLATE = "360 panorama, interior, {room}, photo, high quality"
ESRGAN_X2 = ROOT / "checkpoints" / "realesrgan" / "RealESRGAN_x2plus.pth"
PANO_W, PANO_H = 1024, 512
PANO_STEPS, PANO_GUIDANCE = 30, 7.5
PANO_TEMPLATE = "qxj, 360 panorama, interior, {room}, photo, high quality"
PANO_NEGATIVE = "blurry, distorted, text, watermark, people, frame, border"


def pano_prompt(room: str) -> str:
    """'a coastal style bedroom' -> panorama prompt with the LoRA trigger word."""
    return PANO_TEMPLATE.format(room=room)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_pano_pipeline(device, mmis_lora: str | Path | None = DEFAULT_LORA):
    """SD v1.5 fp16 (DDIM) with the panorama LoRA and, if given, our MMIS LoRA as named adapters.

    Adapters are kept unfused so the MMIS weight can be changed per call with set_weights().
    """
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, dtype=torch.float16, variant="fp16",
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(PANO_LORA.parent), weight_name=PANO_LORA.name, adapter_name="pano")
    if mmis_lora is not None:
        pipe.load_lora_weights(str(mmis_lora), adapter_name="mmis")
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    set_weights(pipe, 0.0)
    return pipe


def load_alt_pipeline(device, path: Path = ALT_MODEL):
    """Alternative generator: the SD 2.1-base panorama fine-tune of SD-T2I-360PanoImage, fp16, DDIM.
    Used as a plain text-to-panorama model (with our circular padding); the MMIS LoRA (SD 1.5) does
    not apply to it.

    The downloaded weights are fp32 .bin files (slow to load, ~70 s); on first use an fp16
    safetensors copy is written next to them (sd-base-fp16, ~2.6 GB) and used from then on."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    cache = path.parent / (path.name + "-fp16")
    if (cache / "model_index.json").exists():
        pipe = StableDiffusionPipeline.from_pretrained(str(cache), dtype=torch.float16, safety_checker=None,
                                                       requires_safety_checker=False)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(str(path), dtype=torch.float16, safety_checker=None,
                                                       requires_safety_checker=False)
        pipe.save_pretrained(str(cache), safe_serialization=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def set_weights(pipe, mmis_weight: float, pano_weight: float = 1.0) -> None:
    """Activate the adapters with the given weights (MMIS weight 0 = panorama LoRA only)."""
    if not getattr(pipe, "peft_config", None) and not pipe.get_list_adapters():
        return                                          # pipeline without adapters (alternative model)
    names = pipe.get_list_adapters()
    has_mmis = any("mmis" in v for v in names.values())
    if has_mmis:
        pipe.set_adapters(["pano", "mmis"], adapter_weights=[pano_weight, float(mmis_weight)])
    else:
        pipe.set_adapters(["pano"], adapter_weights=[pano_weight])


# --------------------------------------------------------------------------- #
# Circular padding along x
# --------------------------------------------------------------------------- #
def _circular_forward(m: nn.Conv2d):
    def forward(x):
        ph, pw = m._pad_orig
        if m._circular:
            x = F.pad(x, (pw, pw, 0, 0), mode="circular")
        else:
            x = F.pad(x, (pw, pw, 0, 0))
        x = F.pad(x, (0, 0, ph, ph))
        return m._conv_forward(x, m.weight, m.bias)
    return forward


def install_circular(*modules) -> None:
    """Patch every padded Conv2d so that it pads x circularly when enabled (zeros otherwise)."""
    for mod in modules:
        for m in mod.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, "_pad_orig") and isinstance(m.padding, tuple) \
                    and max(m.padding) > 0 and m.padding_mode == "zeros":
                m._pad_orig = m.padding
                m.padding = (0, 0)
                m._circular = False
                m.forward = _circular_forward(m)


def set_circular(enabled: bool, *modules) -> None:
    for mod in modules:
        for m in mod.modules():
            if hasattr(m, "_pad_orig"):
                m._circular = bool(enabled)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_pano(pipe, room: str, seed: int, mmis_weight: float = 0.0, circular: bool = True,
                  circular_from: float = 0.0, steps: int = PANO_STEPS, guidance: float = PANO_GUIDANCE,
                  width: int = PANO_W, height: int = PANO_H, prompt: str | None = None) -> np.ndarray:
    """One equirectangular panorama (RGB uint8, height x width).

    circular_from: fraction of the denoising steps after which circular padding is switched on
    (0 = from the start). The VAE decoder always pads circularly when `circular` is True.
    """
    install_circular(pipe.unet, pipe.vae)
    set_weights(pipe, mmis_weight)
    start = int(round(circular_from * steps))
    set_circular(circular and start == 0, pipe.unet)

    def on_step(p, i, t, kw):
        if circular and i + 1 == start:
            set_circular(True, pipe.unet)
        return kw

    gen = torch.Generator(pipe.device).manual_seed(int(seed))
    lat = pipe(prompt or pano_prompt(room), negative_prompt=PANO_NEGATIVE, num_inference_steps=int(steps),
               guidance_scale=float(guidance), width=width, height=height, generator=gen,
               output_type="latent", callback_on_step_end=on_step).images
    set_circular(circular, pipe.vae)
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor, return_dict=False)[0]
    set_circular(False, pipe.unet, pipe.vae)
    img = (img.float().clamp(-1, 1) + 1) / 2
    return (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Layout-conditioned panoramas (milestone 6): ControlNet depth for SD 1.5 + panorama LoRA
# --------------------------------------------------------------------------- #
CONTROLNET_DEPTH = ROOT / "checkpoints" / "controlnet_depth"     # lllyasviel/control_v11f1p_sd15_depth (fp16)


def load_controlnet_pipeline(device, mmis_lora: str | Path | None = None):
    """SD v1.5 + ControlNet depth + the panorama LoRA (and optionally the MMIS LoRA), fp16, DDIM."""
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
    cn = ControlNetModel.from_pretrained(str(CONTROLNET_DEPTH), dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL, controlnet=cn, dtype=torch.float16,
                                                             variant="fp16", safety_checker=None,
                                                             requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(PANO_LORA.parent), weight_name=PANO_LORA.name, adapter_name="pano")
    if mmis_lora is not None:
        pipe.load_lora_weights(str(mmis_lora), adapter_name="mmis")
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    set_weights(pipe, 0.0)
    return pipe


@torch.inference_mode()
def generate_pano_controlled(pipe, room: str, seed: int, control: np.ndarray, scale: float = 1.0,
                             mmis_weight: float = 0.0, steps: int = PANO_STEPS, guidance: float = PANO_GUIDANCE) -> np.ndarray:
    """Panorama that follows a depth condition (control: H x W x 3 uint8, near = white).
    scale = 0 gives the unconditioned panorama of the same pipeline and seed (the baseline)."""
    from PIL import Image
    install_circular(pipe.unet, pipe.vae, pipe.controlnet)
    set_weights(pipe, mmis_weight)
    set_circular(True, pipe.unet, pipe.controlnet)
    h, w = control.shape[:2]
    gen = torch.Generator(pipe.device).manual_seed(int(seed))
    lat = pipe(pano_prompt(room), image=Image.fromarray(control), negative_prompt=PANO_NEGATIVE,
               num_inference_steps=int(steps), guidance_scale=float(guidance), width=w, height=h,
               controlnet_conditioning_scale=float(scale), generator=gen, output_type="latent").images
    set_circular(True, pipe.vae)
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor, return_dict=False)[0]
    set_circular(False, pipe.unet, pipe.vae, pipe.controlnet)
    img = (img.float().clamp(-1, 1) + 1) / 2
    return (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Super-resolution
# --------------------------------------------------------------------------- #
class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = _RDB(nf), _RDB(nf), _RDB(nf)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator (Wang et al., 2018) as used by Real-ESRGAN; x2 works on a pixel-unshuffled input."""

    def __init__(self, scale: int = 2, nf: int = 64, nb: int = 23):
        super().__init__()
        self.scale = scale
        in_ch = 3 * (4 if scale == 2 else 1)
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.scale == 2:
            x = F.pixel_unshuffle(x, 2)
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_esrgan(device, path: Path = ESRGAN_X2) -> RRDBNet:
    net = RRDBNet(scale=2)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)
    return net.half().to(device).eval()


@torch.inference_mode()
def esrgan_upscale(net: RRDBNet, img: np.ndarray, tile: int = 256, overlap: int = 16, wrap_pad: int = 16) -> np.ndarray:
    """x2 Real-ESRGAN, tiled. The panorama is first padded by `wrap_pad` columns copied from the
    opposite border, so the network sees continuous content across the seam; the padding is cut off."""
    dev = next(net.parameters()).device
    ext = cv2.copyMakeBorder(img, 0, 0, wrap_pad, wrap_pad, cv2.BORDER_WRAP)
    x = torch.from_numpy(ext).permute(2, 0, 1)[None].to(dev).half() / 255.0
    _, _, h, w = x.shape
    s = 2
    out = torch.zeros(1, 3, h * s, w * s, device=dev)
    weight = torch.zeros_like(out)
    step = tile - 2 * overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            ya, xa = max(y0 - overlap, 0), max(x0 - overlap, 0)
            yb, xb = min(y0 + step + overlap, h), min(x0 + step + overlap, w)
            y = net(x[..., ya:yb, xa:xb]).float()
            out[..., ya * s:yb * s, xa * s:xb * s] += y
            weight[..., ya * s:yb * s, xa * s:xb * s] += 1
    out = (out / weight).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    out = (out * 255).round().astype(np.uint8)
    return out[:, wrap_pad * s: (w - wrap_pad) * s]


def bicubic_unsharp(img: np.ndarray, scale: int = 2, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
    """Baseline: bicubic x2 + unsharp masking (wrap-aware: the blur wraps around horizontally)."""
    p0 = 4
    ext = cv2.copyMakeBorder(img, 0, 0, p0, p0, cv2.BORDER_WRAP)
    up = cv2.resize(ext, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)[:, p0 * scale:-p0 * scale]
    pad = int(4 * sigma) + 1
    ext = cv2.copyMakeBorder(up, 0, 0, pad, pad, cv2.BORDER_WRAP)
    blur = cv2.GaussianBlur(ext, (0, 0), sigma)[:, pad:-pad]
    return np.clip(up.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount, 0, 255).astype(np.uint8)


def timed(fn, *a, **k):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn(*a, **k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return r, time.perf_counter() - t0
