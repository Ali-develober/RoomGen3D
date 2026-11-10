"""LoRA training on the cleaned MMIS subset (Diffusers + PEFT, fp16, 8-bit Adam, batch 1 + grad accumulation).

Only small LoRA adapters (rank 8) on the UNet attention projections are trained;
the SD v1.5 UNet, VAE and text encoder stay frozen in fp16. The LoRA weights are
kept in fp32 and the forward pass runs under fp16 autocast with a GradScaler.

Training objective (standard DDPM / SD v1.5 epsilon-prediction):
    latents z = VAE(image) * scaling_factor
    pick timestep t, noise e;  z_t = sqrt(a_t) z + sqrt(1 - a_t) e
    loss = || UNet(z_t, t, CLIP(caption)) - e ||^2

Reproducible and resumable: data order is a pure function of (seed, sample counter)
and the noise generator state is saved in every checkpoint.
The entry point is scripts/train_lora.py.
"""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessing import load_rgb, preprocess_for_training, to_model_tensor

# Fixed validation prompts (label-style captions like the training data + one
# free-form prompt to check the model still follows extra detail) and fixed seeds.
VAL_PROMPTS = [
    "an art deco style living room",
    "a rustic style kitchen",
    "a coastal style bedroom",
    "a mid-century modern style dining room",
    "a bohemian style bathroom",
    "a contemporary style living room with large windows and a grey sofa",
]
VAL_SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


@dataclass
class TrainConfig:
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    train_csv: str = "data/mmis/train.csv"
    image_root: str = "data/mmis"
    output_dir: str = "checkpoints/lora_mmis"
    run_name: str = "lora_mmis"          # prefix of figures / tables in results/
    rank: int = 8
    alpha: int = 8                       # LoRA scale = alpha / rank = 1
    lr: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    max_steps: int = 3000                # optimizer steps (each = grad_accum images)
    grad_accum: int = 4
    resolution: int = 512
    checkpoint_every: int = 500
    seed: int = 42
    extra_text_prob: float = 0.0         # chance to append the MMIS caption (optional extra text)
    max_grad_norm: float = 1.0
    val_steps: int = 25
    val_guidance: float = 7.5
    fig_dir: str = "results/figures"
    table_dir: str = "results/tables"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class MMISDataset(Dataset):
    """Cleaned MMIS images -> (pixel tensor in [-1, 1], caption)."""

    def __init__(self, csv_path, image_root, resolution=512, extra_text_prob=0.0, seed=42):
        self.df = pd.read_csv(csv_path)
        self.root = Path(image_root)
        self.resolution = resolution
        self.extra_text_prob = extra_text_prob
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def caption(self, i: int) -> str:
        """Label caption; optionally + MMIS text (decided by a per-image fixed random draw)."""
        row = self.df.iloc[i]
        cap = row["caption"]
        if self.extra_text_prob > 0 and isinstance(row.get("extra_text"), str):
            if np.random.default_rng([self.seed, i]).random() < self.extra_text_prob:
                cap = f"{cap}, {row['extra_text']}"
        return cap

    def __getitem__(self, i: int) -> dict:
        img = preprocess_for_training(load_rgb(self.root / self.df.iloc[i]["path"]), self.resolution)
        return {"pixel_values": to_model_tensor(img), "caption": self.caption(i)}


def sample_index(sample_counter: int, n: int, seed: int) -> int:
    """Which image to use for the k-th training sample.

    Each epoch is a fresh permutation seeded by (seed, epoch), so the order only
    depends on the counter: resuming at counter k reproduces the same sequence.
    """
    epoch, pos = divmod(sample_counter, n)
    return int(np.random.default_rng([seed, epoch]).permutation(n)[pos])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_sd_components(base_model: str, device, dtype=torch.float16) -> dict:
    """Tokenizer, text encoder, VAE, UNet (fp16 variant weights) and the DDPM noise scheduler."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    kw = {"variant": "fp16", "dtype": dtype}
    return {
        "tokenizer": CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer"),
        "text_encoder": CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", **kw).to(device),
        "vae": AutoencoderKL.from_pretrained(base_model, subfolder="vae", **kw).to(device),
        "unet": UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", **kw).to(device),
        "noise_scheduler": DDPMScheduler.from_pretrained(base_model, subfolder="scheduler"),
    }


def add_lora(unet, rank: int, alpha: int, gradient_checkpointing: bool = True) -> list:
    """Freeze the UNet, attach LoRA adapters, keep only LoRA params trainable (in fp32)."""
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                                target_modules=LORA_TARGETS))
    cast_training_params(unet, dtype=torch.float32)
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    return [p for p in unet.parameters() if p.requires_grad]


def build_optimizer(params, lr: float, eight_bit: bool = True):
    """8-bit AdamW (bitsandbytes, GPU only) to save optimizer memory; plain AdamW for CPU tests."""
    if eight_bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-2)
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)


def build_lr_scheduler(optimizer, cfg: TrainConfig):
    from diffusers.optimization import get_scheduler
    return get_scheduler(cfg.lr_scheduler, optimizer, num_warmup_steps=cfg.warmup_steps,
                         num_training_steps=cfg.max_steps)


# --------------------------------------------------------------------------- #
# One micro-step (one image)
# --------------------------------------------------------------------------- #
def encode_prompt(tokenizer, text_encoder, captions: list[str], device) -> torch.Tensor:
    ids = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


def diffusion_loss(unet, vae, text_emb, pixel_values, noise_scheduler, gen: torch.Generator,
                   amp: bool) -> torch.Tensor:
    """Epsilon-prediction MSE for one batch. `gen` drives all randomness (latent sample, noise, t)."""
    device = pixel_values.device
    with torch.no_grad():
        latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample(generator=gen)
        latents = latents * vae.config.scaling_factor
    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                      generator=gen, device=device).long()
    noisy = noise_scheduler.add_noise(latents, noise, t)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        pred = unet(noisy, t, text_emb.to(noisy.dtype)).sample
    return F.mse_loss(pred.float(), noise.float())


# --------------------------------------------------------------------------- #
# Checkpoints (resumable)
# --------------------------------------------------------------------------- #
def lora_state_dict(unet) -> dict:
    from peft.utils import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(unet).items()}


def save_checkpoint(out_dir: Path, step: int, unet, optimizer, lr_sched, scaler, gen, cfg: TrainConfig) -> Path:
    """Save LoRA weights (diffusers format, loadable with pipe.load_lora_weights) + full training state.

    Written to a temp folder first and renamed, so a crash never leaves a half checkpoint.
    """
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers

    final = Path(out_dir) / f"step_{step:05d}"
    tmp = Path(out_dir) / f".tmp_step_{step:05d}"
    tmp.mkdir(parents=True, exist_ok=True)
    lora = lora_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(tmp, unet_lora_layers=convert_state_dict_to_diffusers(lora),
                                              safe_serialization=True)
    torch.save({
        "step": step,
        "lora_peft": lora,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "generator": gen.get_state(),
        "config": asdict(cfg),
    }, tmp / "training_state.pt")
    if final.exists():
        import shutil
        shutil.rmtree(final)
    tmp.rename(final)
    (Path(out_dir) / "latest.txt").write_text(final.name)
    return final


def latest_checkpoint(out_dir) -> Path | None:
    f = Path(out_dir) / "latest.txt"
    if f.exists() and (Path(out_dir) / f.read_text().strip() / "training_state.pt").exists():
        return Path(out_dir) / f.read_text().strip()
    return None


def load_checkpoint(ckpt: Path, unet, optimizer, lr_sched, scaler, gen) -> int:
    """Restore LoRA weights + optimizer/scheduler/scaler/noise-generator state. Returns the step."""
    from peft.utils import set_peft_model_state_dict

    state = torch.load(Path(ckpt) / "training_state.pt", map_location="cpu", weights_only=False)
    result = set_peft_model_state_dict(unet, state["lora_peft"])
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"LoRA keys did not match the model: {unexpected[:3]}")
    optimizer.load_state_dict(state["optimizer"])
    lr_sched.load_state_dict(state["lr_scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    gen.set_state(state["generator"])
    return int(state["step"])


# --------------------------------------------------------------------------- #
# Loss log
# --------------------------------------------------------------------------- #
LOSS_FIELDS = ["step", "loss", "lr", "sec_per_step", "vram_peak_gb"]


def init_loss_log(path: Path, resume_step: int) -> None:
    """Create the CSV, or on resume drop rows logged after the checkpoint (they will be redone)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume_step > 0 and path.exists():
        df = pd.read_csv(path)
        df[df["step"] <= resume_step].to_csv(path, index=False)
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, LOSS_FIELDS).writeheader()


def append_loss(path: Path, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, LOSS_FIELDS).writerow(row)


def plot_loss(csv_path: Path, out_path: Path, window: int = 50) -> None:
    """Raw per-step loss (light) + moving average (dark) + learning rate on a second axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["step"], df["loss"], color="#4C72B0", alpha=0.25, lw=0.8, label="loss (per step)")
    ax.plot(df["step"], df["loss"].rolling(min(window, len(df)), min_periods=1).mean(),
            color="#1f3b73", lw=2, label=f"moving average ({window} steps)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("MSE loss (noise prediction)")
    ax2 = ax.twinx()
    ax2.plot(df["step"], df["lr"], color="#DD8452", lw=1.2, label="learning rate")
    ax2.set_ylabel("learning rate")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper right", fontsize=8)
    ax.set_title("LoRA training loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Validation grid
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validation_grid(models: dict, cfg: TrainConfig, step: int, device, out_path: Path) -> None:
    """Generate the 6 fixed prompts with fixed seeds using the current LoRA and save a 2x3 grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    unet = models["unet"]
    was_training = unet.training
    unet.eval()
    pipe = StableDiffusionPipeline(
        vae=models["vae"], text_encoder=models["text_encoder"], tokenizer=models["tokenizer"], unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(cfg.base_model, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    images = []
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for prompt, seed in zip(VAL_PROMPTS, VAL_SEEDS):
            g = torch.Generator(device).manual_seed(seed)
            images.append(pipe(prompt, num_inference_steps=cfg.val_steps, guidance_scale=cfg.val_guidance,
                               height=cfg.resolution, width=cfg.resolution, generator=g).images[0])
    del pipe
    if was_training:
        unet.train()

    fig, ax = plt.subplots(2, 3, figsize=(12, 8.6))
    for a, img, prompt in zip(ax.ravel(), images, VAL_PROMPTS):
        a.imshow(img)
        a.set_title(prompt if len(prompt) < 45 else prompt[:42] + "...", fontsize=9)
        a.axis("off")
    fig.suptitle(f"Validation prompts at step {step} (fixed seeds{', base SD v1.5' if step == 0 else ''})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig, resume: bool = True, val_at_start: bool = True) -> None:
    from tqdm import tqdm
    from .utils import set_seed

    assert torch.cuda.is_available(), "training needs the GPU"
    device = torch.device("cuda")
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    fig_dir, table_dir = Path(cfg.fig_dir), Path(cfg.table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = load_sd_components(cfg.base_model, device)
    models["vae"].requires_grad_(False)
    models["text_encoder"].requires_grad_(False)
    assert models["noise_scheduler"].config.prediction_type == "epsilon"
    params = add_lora(models["unet"], cfg.rank, cfg.alpha)
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f} M")
    optimizer = build_optimizer(params, cfg.lr)
    lr_sched = build_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device).manual_seed(cfg.seed)

    step = 0
    ckpt = latest_checkpoint(out_dir) if resume else None
    if ckpt is not None:
        step = load_checkpoint(ckpt, models["unet"], optimizer, lr_sched, scaler, gen)
        print(f"resumed from {ckpt} (step {step})")
    loss_csv = table_dir / f"{cfg.run_name}_loss.csv"
    init_loss_log(loss_csv, step)

    if step == 0 and val_at_start:
        validation_grid(models, cfg, 0, device, fig_dir / f"{cfg.run_name}_val_step00000.png")

    ds = MMISDataset(cfg.train_csv, cfg.image_root, cfg.resolution, cfg.extra_text_prob, cfg.seed)
    unet = models["unet"]
    unet.train()
    torch.cuda.reset_peak_memory_stats()
    bar = tqdm(total=cfg.max_steps, initial=step, unit="step")
    t0, running = time.perf_counter(), 0.0
    while step < cfg.max_steps:
        for micro in range(cfg.grad_accum):
            item = ds[sample_index(step * cfg.grad_accum + micro, len(ds), cfg.seed)]
            pixels = item["pixel_values"].unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encode_prompt(models["tokenizer"], models["text_encoder"], [item["caption"]], device)
            loss = diffusion_loss(unet, models["vae"], emb, pixels, models["noise_scheduler"], gen, amp=True)
            scaler.scale(loss / cfg.grad_accum).backward()
            running += loss.item() / cfg.grad_accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_now = lr_sched.get_last_lr()[0]
        lr_sched.step()
        step += 1

        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        append_loss(loss_csv, {"step": step, "loss": round(running, 6), "lr": lr_now, "sec_per_step": round(dt, 3),
                               "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
        bar.set_postfix(loss=f"{running:.4f}", lr=f"{lr_now:.2e}")
        bar.update(1)
        running = 0.0

        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            save_checkpoint(out_dir, step, unet, optimizer, lr_sched, scaler, gen, cfg)
            validation_grid(models, cfg, step, device, fig_dir / f"{cfg.run_name}_val_step{step:05d}.png")
            plot_loss(loss_csv, fig_dir / f"{cfg.run_name}_loss.png")
            t0 = time.perf_counter()  # don't count checkpoint/validation time as step time
    bar.close()
    print(f"done: {step} steps, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def estimate_total_time(loss_csv: Path, max_steps: int, skip_first: int = 5) -> dict:
    """From a smoke-test log: median seconds/step (ignoring warm-up steps) -> projected hours."""
    df = pd.read_csv(loss_csv)
    s = float(df["sec_per_step"].iloc[skip_first:].median()) if len(df) > skip_first else float(df["sec_per_step"].median())
    return {"sec_per_step": s, "est_hours": s * max_steps / 3600,
            "vram_peak_gb": float(df["vram_peak_gb"].max()), "steps_logged": len(df),
            "note": "excludes ~6 checkpoint+validation rounds" if max_steps >= 500 else ""}


