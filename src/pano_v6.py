"""v6 panorama on the proxy scene (PLAN_V6 milestone 2): SDXL + 360Redmond LoRA + depth conditioning from the
proxy equirectangular depth (Ctrl-Room order: layout -> proxy scene -> depth-conditioned generation).

    condition   inverse distance of the proxy scene seen from the layout camera (1.5 m, max clearance),
                normalised 0-255 (near = white), 2048 x 1024
    model       SDXL base fp16 + fp16-fix VAE + 360Redmond LoRA (fused) +
                  "cn_small"  diffusers/controlnet-depth-sdxl-1.0-small (openrail++)
                  "cn_mid"    diffusers/controlnet-depth-sdxl-1.0-mid   (openrail++)
                  "t2i"       TencentARC/t2i-adapter-depth-midas-sdxl-1.0 (apache-2.0)
                circular padding in UNet, VAE and ControlNet / adapter (seamless wrap), model CPU offload,
                VAE tiling off (wrap padding needs one decode), VAE slicing
    prompt      style word expanded into interior descriptors (materials, palette, furniture style) so
                that a style is not read as a location ("coastal" -> light oak, white linen, pale blue ...,
                never "sea view"); the furniture list of the layout; negative prompt against glass walls,
                floor-to-ceiling windows, sea / beach views, curved lines; every prompt is asserted to fit
                the 77-token CLIP window of both SDXL text encoders (and SD 1.5 for the shots).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import generate_pano as gp
from . import generate_pano_xl as gx
from . import memory as mem

ROOT = Path(__file__).resolve().parents[1]
CONDS = {"cn_small": ROOT / "checkpoints" / "cn_depth_sdxl_small",
         "cn_mid": ROOT / "checkpoints" / "cn_depth_sdxl_mid",
         "t2i": ROOT / "checkpoints" / "t2i_depth_sdxl"}      # mid / t2i deleted after the sweep (unused, disk)
MAX_TOKENS = 77

FLOORS = {"coastal": "whitewashed oak floor", "contemporary": "large grey stone tile floor", "rustic": "wide plank wood floor",
          "mid-century modern": "walnut parquet floor", "art deco": "black and white marble floor",
          "mediterranean": "terracotta tile floor", "scandinavian": "light ash wood floor", "farmhouse": "oak plank floor",
          "industrial": "polished concrete floor", "bohemian": "wood floor with a patterned rug", "modern": "light oak floor",
          "minimalist": "pale wood floor", "traditional": "herringbone oak floor", "japandi": "light oak floor"}
WET = {"kitchen": "tile floor", "bathroom": "tile floor"}

STYLE_WORDS = {   # style -> (materials / furniture, palette)
    "coastal": ("light oak wood, white linen, rattan and wicker furniture, striped cushions", "white, sand beige and pale blue palette"),
    "contemporary": ("sleek low furniture, matte lacquer, chrome accents, abstract art", "warm grey, white and black palette"),
    "rustic": ("reclaimed wood beams, rough timber furniture, stone and wrought iron", "warm brown and cream palette"),
    "mid-century modern": ("teak and walnut furniture with tapered legs, molded wood chairs, globe lamps", "mustard, teal and walnut palette"),
    "art deco": ("geometric patterns, brass and gold trim, velvet upholstery, lacquered wood", "black, emerald and gold palette"),
    "mediterranean": ("terracotta tiles, white plaster walls, wrought iron, carved dark wood", "terracotta, white and olive palette"),
    "scandinavian": ("pale birch wood, simple functional furniture, wool textiles", "white, light grey and natural wood palette"),
    "farmhouse": ("shiplap walls, distressed painted wood, vintage furniture", "white, cream and natural wood palette"),
    "industrial": ("exposed brick, black metal frames, concrete, leather and dark wood", "charcoal, rust and brown palette"),
    "bohemian": ("layered patterned rugs, rattan, macrame, many plants, eclectic furniture", "terracotta, mustard and green palette"),
    "modern": ("clean lines, flat-front cabinets, minimal decor", "white, grey and oak palette"),
    "minimalist": ("very few pieces, clean lines, hidden storage", "white and light grey palette"),
    "traditional": ("classic carved wood furniture, upholstered pieces, framed paintings", "burgundy, cream and mahogany palette"),
    "japandi": ("low oak furniture, paper lamps, linen, natural textures", "beige, oak and charcoal palette"),
}
GENERIC_STYLE = ("well-made furniture, natural materials", "harmonious neutral palette")
SINGULAR_PLURAL = {"chair": "chairs", "nightstand": "nightstands", "armchair": "armchairs", "counter": "counters",
                   "plant": "plants", "shelf": "shelves"}
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
NEGATIVE = ("glass walls, floor-to-ceiling windows, glass facade, sea view, ocean, beach, outdoor, landscape, "
            "carpet, wall clutter, posters, mirror, curved walls, bent lines, distorted, deformed furniture, melted, "
            "extra furniture, people, text, watermark, blurry, lowres, cartoon")


def _tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(str(gx.SDXL_BASE / "tokenizer"))


_TOK = None


def n_tokens(text: str) -> int:
    """Number of CLIP tokens incl. BOS / EOS (same BPE for SD 1.5 and both SDXL text encoders)."""
    global _TOK
    if _TOK is None:
        _TOK = _tokenizer()
        _TOK.model_max_length = 10 ** 6                  # counting only: no length warning
    return len(_TOK(text).input_ids)


def fit_tokens(text: str, limit: int = MAX_TOKENS) -> str:
    """Drop words from the end until the text fits the CLIP window (what the pipeline would cut anyway, but
    without the truncation warning and without a half-cut last word)."""
    words = text.split(" ")
    while n_tokens(" ".join(words)) > limit and len(words) > 1:
        words.pop()
    return " ".join(words).rstrip(",")


def assert_tokens(text: str, limit: int = MAX_TOKENS) -> str:
    n = n_tokens(text)
    assert n <= limit, f"prompt has {n} > {limit} CLIP tokens: {text!r}"
    return text


def furniture_phrase(lay: dict, max_items: int = 5) -> str:
    """'a dining table, eight chairs, a sideboard' from the layout (largest classes first)."""
    c = Counter(i["cls"] for i in lay["items"] if i["cls"] not in ("television", "counter"))
    area = {i["cls"]: i["w"] * i["d"] for i in lay["items"]}
    parts = []
    for cls, n in sorted(c.items(), key=lambda kv: -area[kv[0]] * kv[1])[:max_items]:
        name = cls if cls != "TV unit" else "TV unit"
        parts.append((f"an {name}" if name[0] in "aeiou" else f"a {name}") if n == 1 else f"{NUM.get(n, str(n))} {SINGULAR_PLURAL.get(cls, cls + 's')}")
    return ", ".join(parts)


def style_words(style: str) -> tuple[str, str]:
    s = style.lower()
    for k in sorted(STYLE_WORDS, key=len, reverse=True):
        if k in s or s in k:
            return STYLE_WORDS[k]
    return GENERIC_STYLE


def floor_words(lay: dict) -> str:
    s = lay["style"].lower()
    f = next((v for k, v in FLOORS.items() if k in s or s in k), "wood floor")
    if lay["room"] in WET and "wood" in f:
        f = WET[lay["room"]]
    return f


def windows_phrase(lay: dict) -> str:
    n = sum(o["kind"] == "window" for o in lay["openings"])
    return "one window with curtains" if n == 1 else f"{NUM.get(n, str(n))} windows with curtains"


def pano_prompt(lay: dict) -> str:
    """Prompt of the panorama, guaranteed to fit 77 CLIP tokens (furniture list shortened if needed)."""
    mat, pal = style_words(lay["style"])
    room = lay["room"]
    for k in (5, 4, 3, 2, 1, 0):
        furn = furniture_phrase(lay, k)
        p = (f"360 view, 360 panorama, {lay['style']} style {room} interior, {mat}, {pal}, {floor_words(lay)}"
             + (f", {furn}" if furn else "") + f", plain painted walls, {windows_phrase(lay)}, interior design photo")
        if n_tokens(p) <= MAX_TOKENS:
            return p
    return assert_tokens(p)


def load_pipeline(cond: str = "cn_small", offload: bool = True, ram_checked: bool = False):
    """SDXL + 360Redmond (fused) + depth ControlNet / T2I adapter, circular padding everywhere."""
    from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionXLAdapterPipeline, StableDiffusionXLControlNetPipeline, T2IAdapter)
    if not ram_checked:
        mem.wait_for_ram(7.5, label="load SDXL v6", reuse=True)
    vae = AutoencoderKL.from_pretrained(str(gx.SDXL_VAE), dtype=torch.float16)
    if cond.startswith("cn"):
        net = ControlNetModel.from_pretrained(str(CONDS[cond]), dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            str(gx.SDXL_BASE), vae=vae, controlnet=net, dtype=torch.float16, variant="fp16", use_safetensors=True,
            add_watermarker=False)
    else:
        net = T2IAdapter.from_pretrained(str(CONDS[cond]), dtype=torch.float16)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            str(gx.SDXL_BASE), vae=vae, adapter=net, dtype=torch.float16, variant="fp16", use_safetensors=True,
            add_watermarker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.load_lora_weights(str(gx.REDMOND.parent), weight_name=gx.REDMOND.name, adapter_name="pano360")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe.set_progress_bar_config(disable=True)
    if offload and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    elif torch.cuda.is_available():
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    gp.install_circular(pipe.unet, pipe.vae, net)
    pipe._cond, pipe._net = cond, net
    return pipe


@torch.inference_mode()
def generate(pipe, control: np.ndarray, prompt: str, seed: int, scale: float = 0.7, end: float = 1.0,
             steps: int = gx.XL_STEPS, guidance: float = gx.XL_GUIDANCE, negative: str = NEGATIVE) -> np.ndarray:
    """2048 x 1024 panorama conditioned on `control` (H x W x 3 uint8 depth image)."""
    from PIL import Image
    assert_tokens(prompt)
    assert_tokens(negative)
    h, w = control.shape[:2]
    gp.set_circular(True, pipe.unet, pipe.vae, pipe._net)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    kw = dict(prompt=prompt, negative_prompt=negative, num_inference_steps=int(steps), guidance_scale=float(guidance),
              width=w, height=h, generator=gen, output_type="np", image=Image.fromarray(control))
    if pipe._cond.startswith("cn"):
        kw.update(controlnet_conditioning_scale=float(scale), control_guidance_end=float(end))
    else:
        kw.update(adapter_conditioning_scale=float(scale), adapter_conditioning_factor=float(end))
    try:
        img = pipe(**kw).images[0]
    finally:
        gp.set_circular(False, pipe.unet, pipe.vae, pipe._net)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
