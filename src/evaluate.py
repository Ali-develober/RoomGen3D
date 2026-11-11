"""2D evaluation: generate the held-out captions with each model, then FID, KID and CLIP Score.

Protocol (README section 8, 2D Generation):
  * Captions: the 500 held-out MMIS images (data/mmis/fid_holdout.csv), never seen in LoRA training.
  * One image per caption per model. The seed depends only on the caption's row index, so every
    model starts from the same initial noise for the same caption.
  * DDIM, 30 steps, CFG 7.5, 512x512, fp16.
  * FID / KID: Inception-v3 pool features (torchmetrics + torch-fidelity) of the generated images vs.
    the 500 real held-out images. MMIS images are 256x256, so generated images are downscaled to
    256x256 (bicubic, anti-aliased) first: both sets then reach Inception at the same resolution.
  * KID is reported as mean +- std over random subsets; it is unbiased, so it is more reliable than
    FID with only 500 images (FID needs a 2048x2048 covariance estimate).
  * CLIP Score (ViT-L/14): 100 * max(cos(image, caption), 0), the torchmetrics definition, computed
    per image so we can report mean +- std. The real held-out images get the same score as a reference.

Only one large model is on the GPU at a time: SD for generation, then Inception, then CLIP.
The entry point is scripts/eval_2d.py.
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm



def release_memory() -> None:
    """Collect garbage and return cached CUDA memory. Call AFTER `del` of the last model reference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_MODEL = "openai/clip-vit-large-patch14"
SEED_BASE = 20_000            # seed of held-out row i = SEED_BASE + i (same for every model)
GEN_STEPS = 30
GEN_GUIDANCE = 7.5
GEN_SIZE = 512
METRIC_SIZE = 256             # resolution of real MMIS images; generated images are resized to this
KID_SUBSET_SIZE = 100
KID_SUBSETS = 100

# name -> LoRA folder (None = vanilla SD v1.5). Order = order in tables and figures.
MODELS = {
    "sd15_base": None,
    "lora_step1000": "checkpoints/lora_mmis/step_01000",
    "lora_step2000": "checkpoints/lora_mmis/step_02000",
    "lora_step3000": "checkpoints/lora_mmis/step_03000",
}
MODEL_LABELS = {
    "sd15_base": "SD v1.5 (base)",
    "lora_step1000": "LoRA step 1000",
    "lora_step2000": "LoRA step 2000",
    "lora_step3000": "LoRA step 3000",
    "real_holdout": "Real MMIS (held-out)",
}
# (style, room) cells shown in the qualitative grid; the first held-out row of each cell is used.
QUAL_CELLS = [("Art Deco", "Living Room"), ("Rustic", "Kitchen"), ("Coastal", "Bedroom"),
              ("Mid-Century Modern", "Dining Room"), ("Bohemian", "Bathroom"), ("Mediterranean", "Living Room")]


# --------------------------------------------------------------------------- #
# Held-out captions
# --------------------------------------------------------------------------- #
def load_holdout(csv_path, image_root, limit: int | None = None) -> pd.DataFrame:
    """Held-out rows with a stable id, seed, real-image path and output file name.

    `limit` keeps `limit` evenly spaced rows (covers all styles; used by the smoke test).
    The ids/seeds/file names do not depend on `limit`, so a later full run reuses those images.
    """
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df["idx"] = df.index
    df["seed"] = SEED_BASE + df["idx"]
    df["real_path"] = [str(Path(image_root) / p) for p in df["path"]]
    df["file"] = [f"{i:04d}.png" for i in df["idx"]]
    if limit is not None and limit < len(df):
        keep = np.unique(np.linspace(0, len(df) - 1, limit).round().astype(int))
        df = df.iloc[keep].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def load_pipeline(lora_dir: str | None, device, base_model: str = BASE_MODEL):
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


def missing_rows(df: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Rows whose image is not generated yet (resume support)."""
    out_dir = Path(out_dir)
    return df[[not (out_dir / f).exists() for f in df["file"]]]


def generate_images(pipe, df: pd.DataFrame, out_dir, device, batch_size: int = 4, desc: str = "") -> dict:
    """Generate one image per row (its caption + its seed); skip files that already exist.

    Each image has its own torch.Generator, so the result does not depend on the batch it is in.
    Files are written to a temp name then renamed, so an interrupted run never leaves a broken PNG.
    Returns {"generated": n, "sec_per_image": s}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = missing_rows(df, out_dir)
    if todo.empty:
        return {"generated": 0, "sec_per_image": float("nan")}
    t0, n = time.perf_counter(), 0
    bar = tqdm(total=len(todo), desc=desc, unit="img")
    for start in range(0, len(todo), batch_size):
        chunk = todo.iloc[start:start + batch_size]
        gens = [torch.Generator(device).manual_seed(int(s)) for s in chunk["seed"]]
        with torch.inference_mode():
            images = pipe(list(chunk["caption"]), num_inference_steps=GEN_STEPS, guidance_scale=GEN_GUIDANCE,
                          height=GEN_SIZE, width=GEN_SIZE, generator=gens).images
        for img, f in zip(images, chunk["file"]):
            tmp = out_dir / f".tmp_{f}"
            img.save(tmp)
            tmp.replace(out_dir / f)
        n += len(chunk)
        bar.update(len(chunk))
    bar.close()
    return {"generated": n, "sec_per_image": (time.perf_counter() - t0) / n}


def generate_all(df: pd.DataFrame, out_root, device, models: dict = MODELS, batch_size: int = 4,
                 root: Path = Path(".")) -> dict:
    """Generate every model in turn, freeing the pipeline before loading the next one."""
    timing = {}
    for name, lora in models.items():
        out_dir = Path(out_root) / name
        if missing_rows(df, out_dir).empty:
            print(f"{name}: all {len(df)} images present, skipping")
            continue
        pipe = load_pipeline(None if lora is None else root / lora, device)
        timing[name] = generate_images(pipe, df, out_dir, device, batch_size, desc=name)
        print(f"{name}: {timing[name]['generated']} images, {timing[name]['sec_per_image']:.2f} s/image")
        del pipe          # drop the only reference first, then collect + release the CUDA cache
        release_memory()
    return timing


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def load_uint8(paths, size: int = METRIC_SIZE) -> torch.Tensor:
    """Images as an N x 3 x size x size uint8 tensor (bicubic + anti-aliasing when downscaling)."""
    arrs = []
    for p in paths:
        with Image.open(p) as f:
            img = f.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BICUBIC, reducing_gap=None)
        arrs.append(np.asarray(img))
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def fid_kid(real_paths, fake_sets: dict, device, batch: int = 50) -> dict:
    """FID and KID (mean, std) of each generated set against the same real set.

    The real features are computed once (reset_real_features=False) and reused for every model.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    n_min = min([len(real_paths)] + [len(p) for p in fake_sets.values()])
    subset = min(KID_SUBSET_SIZE, n_min)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
    kid = KernelInceptionDistance(feature=2048, subsets=KID_SUBSETS, subset_size=subset,
                                  reset_real_features=False).to(device)
    torch.manual_seed(0)  # KID subsets are random: fixed seed -> reproducible mean/std

    # images are read from disk one small batch at a time; only Inception features are kept
    for i in range(0, len(real_paths), batch):
        x = load_uint8(real_paths[i:i + batch]).to(device)
        fid.update(x, real=True)
        kid.update(x, real=True)

    out = {}
    for name, paths in fake_sets.items():
        for i in tqdm(range(0, len(paths), batch), desc=f"inception {name}", leave=False):
            x = load_uint8(paths[i:i + batch]).to(device)
            fid.update(x, real=False)
            kid.update(x, real=False)
        torch.manual_seed(0)
        k_mean, k_std = kid.compute()
        out[name] = {"fid": float(fid.compute()), "kid_mean": float(k_mean), "kid_std": float(k_std),
                     "kid_subset_size": subset}
        fid.reset()
        kid.reset()
    del fid, kid, x
    release_memory()
    return out


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.float16).to(device).eval()
    return model, CLIPProcessor.from_pretrained(CLIP_MODEL)


@torch.inference_mode()
def clip_scores(model, processor, paths, captions, device, batch: int = 32) -> np.ndarray:
    """Per-image CLIP Score = 100 * max(cos(image_emb, text_emb), 0)  (same formula as torchmetrics)."""
    scores = []
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            with Image.open(p) as f:
                imgs.append(f.convert("RGB"))
        inp = processor(text=list(captions[i:i + batch]), images=imgs, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        img = model.get_image_features(pixel_values=inp["pixel_values"].half())
        txt = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        img = getattr(img, "pooler_output", img)   # transformers may wrap the projected embedding
        txt = getattr(txt, "pooler_output", txt)
        cos = torch.nn.functional.cosine_similarity(img.float(), txt.float(), dim=-1)
        scores.append((100 * cos.clamp(min=0)).cpu().numpy())
    return np.concatenate(scores)


def clip_all(df: pd.DataFrame, image_sets: dict, device) -> pd.DataFrame:
    """Per-image CLIP scores for every image set (dict name -> list of paths aligned with df)."""
    model, proc = load_clip(device)
    rows = []
    for name, paths in image_sets.items():
        s = clip_scores(model, proc, paths, list(df["caption"]), device)
        rows.append(pd.DataFrame({"model": name, "idx": df["idx"].values, "caption": df["caption"].values,
                                  "clip_score": s}))
    del model, proc
    release_memory()
    return pd.concat(rows, ignore_index=True)


def summarize(fk: dict, clip_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per model (+ the real reference): FID, KID mean/std, CLIP mean/std."""
    rows = []
    for name in list(fk) + ["real_holdout"]:
        c = clip_df.loc[clip_df["model"] == name, "clip_score"]
        r = {"model": name, "label": MODEL_LABELS.get(name, name), "n_images": n,
             "fid": np.nan, "kid_mean": np.nan, "kid_std": np.nan, "kid_subset_size": np.nan,
             "clip_mean": c.mean(), "clip_std": c.std(ddof=1)}
        r.update(fk.get(name, {}))
        rows.append(r)
    return pd.DataFrame(rows)


def select_best(summary: pd.DataFrame) -> str:
    """Best generator: lowest mean rank over FID and KID; ties broken by the higher CLIP Score."""
    g = summary[summary["model"] != "real_holdout"].copy()
    g["rank"] = g["fid"].rank() + g["kid_mean"].rank()
    g = g.sort_values(["rank", "clip_mean"], ascending=[True, False])
    return g.iloc[0]["model"]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# Ordinal colours: base model in neutral grey, LoRA checkpoints in one blue ramp (light -> dark = later step).
MODEL_COLORS = {"sd15_base": "#8a8a85", "lora_step1000": "#86b6ef",
                "lora_step2000": "#2a78d6", "lora_step3000": "#184f95"}


def plot_metrics(summary: pd.DataFrame, out_path, best: str | None = None) -> None:
    """Three small panels (FID, KID, CLIP Score): each metric on its own axis, one bar per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = summary[summary["model"] != "real_holdout"].reset_index(drop=True)
    real_clip = summary.loc[summary["model"] == "real_holdout", "clip_mean"]
    labels = [MODEL_LABELS[m].replace(" (base)", "\n(base)").replace("LoRA ", "LoRA\n") for m in g["model"]]
    colors = [MODEL_COLORS.get(m, "#8a8a85") for m in g["model"]]
    x = np.arange(len(g))
    panels = [
        ("fid", None, "FID  (lower is better)", "{:.1f}"),
        ("kid_mean", "kid_std", "KID x 1000  (lower is better)", "{:.1f}"),
        ("clip_mean", "clip_std", "CLIP Score, ViT-L/14  (higher is better)", "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (col, err, title, fmt) in zip(axes, panels):
        scale = 1000 if col == "kid_mean" else 1
        y = g[col].values * scale
        e = g[err].values * scale if err else None
        top = (y + (e if e is not None else 0)).max()
        if col == "clip_mean":
            # CLIP scores sit far from zero: dots + std bars (a bar chart would need a cut y-axis)
            for xi, yi, ei, c in zip(x, y, e, colors):
                ax.errorbar(xi, yi, yerr=ei, fmt="o", ms=9, color=c, ecolor="#555", elinewidth=1, capsize=3,
                            mec="white", mew=1.5, zorder=3)
                ax.text(xi + 0.12, yi, fmt.format(yi), ha="left", va="center", fontsize=9, color="#222")
            if len(real_clip):
                ax.axhline(real_clip.iloc[0], color="#555", ls="--", lw=1.2,
                           label=f"real held-out images ({real_clip.iloc[0]:.1f})")
                ax.legend(loc="lower right", fontsize=8.5, frameon=False)
            lo = min(y.min() - e.max(), real_clip.iloc[0] if len(real_clip) else np.inf)
            ax.set_ylim(lo - 4, top + 2)
            ax.set_xlim(-0.5, len(g) - 0.5)
        else:
            ax.bar(x, y, width=0.62, color=colors, edgecolor="white", linewidth=2,
                   yerr=e, error_kw={"ecolor": "#555", "elinewidth": 1, "capsize": 3})
            for xi, yi, ei in zip(x, y, e if e is not None else np.zeros_like(y)):
                ax.text(xi, yi + ei + top * 0.015, fmt.format(yi), ha="center", va="bottom", fontsize=9,
                        color="#222")
            ax.set_ylim(0, top * 1.12)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        if best is not None:
            bi = list(g["model"]).index(best)
            ax.get_xticklabels()[bi].set_fontweight("bold")
    n = int(g["n_images"].iloc[0])
    fig.suptitle(f"2D generation on {n} held-out MMIS captions (DDIM 30 steps, CFG 7.5; error bars = std)"
                 + (f"; best: {MODEL_LABELS[best]}" if best else ""), fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def qualitative_rows(df: pd.DataFrame, cells=QUAL_CELLS, n: int = 6) -> pd.DataFrame:
    """First held-out row of each chosen (style, room) cell; falls back to evenly spaced rows."""
    picks = []
    for style, room in cells:
        m = df[(df["style"] == style) & (df["room"] == room)]
        if len(m):
            picks.append(m.index[0])
    if len(picks) < n:
        extra = [i for i in np.linspace(0, len(df) - 1, n).round().astype(int) if i not in picks]
        picks += extra[: n - len(picks)]
    return df.loc[picks[:n]]


def plot_qualitative(rows: pd.DataFrame, out_root, out_path, models=tuple(MODELS)) -> None:
    """Grid: one row per caption; columns = real held-out image, then each model (same seed per row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["real_holdout"] + list(models)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.75 * len(rows)), squeeze=False)
    for r, (_, row) in enumerate(rows.iterrows()):
        for c, name in enumerate(cols):
            ax = axes[r, c]
            path = row["real_path"] if name == "real_holdout" else Path(out_root) / name / row["file"]
            ax.imshow(Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC, reducing_gap=None))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(MODEL_LABELS[name], fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(row["caption"].replace(" style ", " style\n"), fontsize=9)
    fig.suptitle("Held-out captions: real MMIS image vs. generated (same seed per row across models)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
