"""2D generation: Stable Diffusion v1.5 + trained LoRA inference.

The LoRA was trained on MMIS captions of the form "a <style> style <room>" (e.g. "a coastal
style bedroom"), so prompts in that form give the most MMIS-like interiors. Free text also works.
"""

from pathlib import Path

import numpy as np
import torch

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_LORA = Path(__file__).resolve().parents[1] / "checkpoints" / "lora_mmis" / "step_03000"
GEN_STEPS, GEN_GUIDANCE, GEN_SIZE = 30, 7.5, 512     # same sampler settings as the 2D evaluation

STYLES = ["Art Deco", "Bohemian", "Coastal", "Contemporary", "Eclectic", "Farmhouse", "Mediterranean",
          "Mid-Century Modern", "Rustic", "Traditional", "Transitional"]
ROOMS = ["Bathroom", "Bedroom", "Dining Room", "Kitchen", "Living Room"]


def load_pipeline(lora_dir: str | Path | None, device, base_model: str = BASE_MODEL):
    """SD v1.5 fp16 pipeline with a DDIM scheduler; the LoRA (if any) is fused into the UNet."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(base_model, dtype=torch.float16, variant="fp16",
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir))
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()   # weights are fused; drop the adapter layers (faster inference)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.inference_mode()
def generate(pipe, prompt: str, seed: int, steps: int = GEN_STEPS, guidance: float = GEN_GUIDANCE,
             size: int = GEN_SIZE, negative_prompt: str | None = None) -> np.ndarray:
    """One image for one prompt and seed (reproducible). Returns RGB uint8 (size x size x 3)."""
    gen = torch.Generator(pipe.device).manual_seed(int(seed))
    img = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=int(steps),
               guidance_scale=float(guidance), height=size, width=size, generator=gen).images[0]
    return np.asarray(img.convert("RGB"))
