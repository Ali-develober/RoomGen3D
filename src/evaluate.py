"""Evaluation. Part 1 (below): 2D generation - FID, KID, CLIP Score. Part 2 (end of file): depth on NYU -
AbsRel, RMSE, delta < 1.25 with scale-and-shift alignment, the preprocessing ablation and robustness tests.

2D evaluation: generate the held-out captions with each model, then FID, KID and CLIP Score.

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


# =========================================================================== #
# Depth evaluation on NYU Depth V2 (README section 8, Depth / 3D + ablation)
# =========================================================================== #
# Both depth models predict RELATIVE inverse depth (disparity up to an unknown scale and
# shift), so each prediction is first aligned to the ground truth:
#     find s, t minimising  sum_valid (s * pred + t - 1/gt)^2      (closed-form least squares)
#     depth = 1 / (s * pred + t)
# This is the standard protocol for affine-invariant depth (MiDaS, Depth Anything).
# Valid pixels: inside the Eigen crop, with 0 < gt <= 10 m. Predictions are capped at 10 m.
NYU_MIN_DEPTH, NYU_MAX_DEPTH = 1e-3, 10.0
EIGEN_CROP = (45, 471, 41, 601)      # rows 45:471, cols 41:601 of the 480x640 frame (Eigen et al., 2014)
NYU_FOCAL, NYU_CENTER = 518.86, (325.58, 253.74)   # Kinect RGB camera intrinsics (NYU toolbox), in px

# Synthetic degradations: a moderate and a severe level of each kind.
#   noise = std of additive Gaussian noise (grey levels), blur = length of a linear motion-blur
#   kernel (px, at 30 degrees), gamma = darkening exponent (applied before the noise).
DEGRADATION_PARAMS = {
    "noise15": {"noise": 15},
    "noise25": {"noise": 25},
    "noise50": {"noise": 50},
    "motion_blur": {"blur": 15},
    "motion_blur25": {"blur": 25},
    "low_light": {"gamma": 2.2, "noise": 10},
    "low_light_severe": {"gamma": 3.0, "noise": 20},
}
DEGRADATIONS = ["clean", *DEGRADATION_PARAMS]
DEGRADATION_LABELS = {"clean": "Clean", "noise15": "Gaussian noise σ=15", "noise25": "Gaussian noise σ=25",
                      "noise50": "Gaussian noise σ=50 (severe)", "motion_blur": "Motion blur (15 px)",
                      "motion_blur25": "Motion blur (25 px, severe)", "low_light": "Low light (γ=2.2 + noise 10)",
                      "low_light_severe": "Low light (γ=3.0 + noise 20, severe)"}
NOISE_LEVELS = [15, 25, 50]          # levels of the denoising PSNR / SSIM table

# Input preprocessing variants (one forward pass each, see depth.prepare_input).
INPUT_VARIANTS = ["raw", "denoise", "clahe", "full", "full_plain"]
REFINE_VARIANTS = ["raw", "full", "full_plain"]     # also scored after depth refinement
# Ablation: name -> (input preprocessing, refine the predicted depth?)
ABLATION = {
    "raw": ("raw", False),
    "denoise": ("denoise", False),
    "clahe": ("clahe", False),
    "full": ("full", False),                    # noise-aware pipeline (noise_aware=True)
    "full_plain": ("full_plain", False),        # plain denoise -> CLAHE -> unsharp (noise_aware=False)
    "full_refine": ("full", True),
    "full_plain_refine": ("full_plain", True),
    "raw_refine": ("raw", True),                # extra row: isolates the effect of refinement alone
}
ABLATION_LABELS = {"raw": "Raw image", "denoise": "+ Denoising", "clahe": "+ CLAHE",
                   "full": "+ Full preprocessing (noise-aware)",
                   "full_plain": "+ Full preprocessing (plain 3 steps)",
                   "full_refine": "+ Full preprocessing (noise-aware) + depth refinement",
                   "full_plain_refine": "+ Full preprocessing (plain 3 steps) + depth refinement",
                   "raw_refine": "Raw image + depth refinement"}
DEPTH_METRICS = ["abs_rel", "rmse", "delta1"]
# Boundary metrics: what matters for the 3D model is the depth at object boundaries.
BOUNDARY_METRICS = ["edge_abs_rel", "edge_rmse", "flying_pct"]
ALL_DEPTH_METRICS = DEPTH_METRICS + BOUNDARY_METRICS
# All four constants below were fixed before looking at any result (nothing is tuned on NYU).
EDGE_REL_STEP = 0.05      # neighbouring GT pixels differing by > 5% in depth = depth discontinuity
EDGE_DILATE = 7           # px: the edge mask is that discontinuity dilated by a 7-px ellipse (+-3 px)
EDGE_MIN_PIXELS = 200     # images with fewer edge pixels get NaN for the edge metrics
FLYING_REL_GAP = 0.05     # a 3D point is "flying" if both neighbours are > 5% of its depth away


def degrade(img: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Synthetic degradation of an RGB uint8 image; `seed` (the image index) makes it reproducible.

    See DEGRADATION_PARAMS. Order inside one degradation: motion blur, then gamma darkening
    out = 255 (in/255)^gamma, then additive Gaussian noise (dark photos are noisy because the
    sensor gain is turned up). Motion blur is a linear kernel at 30 degrees (camera shake).
    """
    import cv2
    if kind == "clean":
        return img
    p = DEGRADATION_PARAMS[kind]
    rng = np.random.default_rng(1234 + seed)
    f = img.astype(np.float32)
    if "blur" in p:
        n = p["blur"]
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), 30, 1.0), (n, n))
        f = cv2.filter2D(f, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)
    if "gamma" in p:
        f = 255.0 * (f / 255.0) ** p["gamma"]
    if "noise" in p:
        f = f + rng.normal(0, p["noise"], f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def nyu_valid_mask(gt: np.ndarray) -> np.ndarray:
    """Pixels used for the metrics: inside the Eigen crop and with a valid depth <= 10 m."""
    r0, r1, c0, c1 = EIGEN_CROP
    mask = np.zeros(gt.shape, bool)
    mask[r0:r1, c0:c1] = True
    return mask & (gt > NYU_MIN_DEPTH) & (gt <= NYU_MAX_DEPTH)


def align_scale_shift(pred_disp: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares scale + shift alignment in disparity space; returns metric depth (capped at 10 m)."""
    x = pred_disp[mask].astype(np.float64)
    y = 1.0 / gt_depth[mask].astype(np.float64)
    # normal equations of  min || s x + t - y ||^2
    a = np.array([[np.sum(x * x), np.sum(x)], [np.sum(x), x.size]])
    b = np.array([np.sum(x * y), np.sum(y)])
    s, t = np.linalg.lstsq(a, b, rcond=None)[0]
    disp = s * pred_disp.astype(np.float64) + t
    return (1.0 / np.maximum(disp, 1.0 / NYU_MAX_DEPTH)).astype(np.float32)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """AbsRel = mean |p-g|/g,  RMSE = sqrt(mean (p-g)^2) in metres,  delta1 = share with max(p/g, g/p) < 1.25."""
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "delta1": float(np.mean(ratio < 1.25))}


def depth_edge_mask(gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels near a depth discontinuity of the ground truth (object boundaries).

    The GT depth gradient is taken as the relative step between 4-neighbours,
    max(d1/d2, d2/d1) - 1. A step above EDGE_REL_STEP (5%) cannot be a smooth surface (at the
    NYU focal length that would be a surface tilted more than 87 degrees), so it is an
    occlusion boundary. Those pixels are dilated by EDGE_DILATE px and intersected with `mask`.
    """
    import cv2
    valid = gt > NYU_MIN_DEPTH
    d = np.where(valid, gt, 1.0).astype(np.float64)
    edge = np.zeros(gt.shape, bool)
    for ax in (0, 1):
        a, v, e = np.moveaxis(d, ax, 0), np.moveaxis(valid, ax, 0), np.moveaxis(edge, ax, 0)
        step = (np.maximum(a[1:] / a[:-1], a[:-1] / a[1:]) > 1 + EDGE_REL_STEP) & v[1:] & v[:-1]
        e[1:] |= step           # e is a view: writes go into `edge`
        e[:-1] |= step
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE_DILATE, EDGE_DILATE))
    return (cv2.dilate(edge.astype(np.uint8), k) > 0) & mask


def backproject(depth: np.ndarray) -> np.ndarray:
    """Depth map (m) -> HxWx3 point cloud in camera coordinates (pinhole model, NYU intrinsics)."""
    h, w = depth.shape
    z = depth.astype(np.float64)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - NYU_CENTER[0]) / NYU_FOCAL * z, (v - NYU_CENTER[1]) / NYU_FOCAL * z, z], axis=-1)


def flying_pixel_mask(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flying pixels: 3D points hanging in empty space between a foreground and a background surface.

    A depth map that blurs an object boundary puts intermediate depths there; back-projected,
    these become a "curtain" of points joining the object to the wall behind it. A point counts
    as flying if, along its image row or its column, the 3D points on BOTH sides are farther
    away than FLYING_REL_GAP x its depth (15 cm at 3 m) and one is nearer, the other farther.
    A perfectly sharp step has none (each side has a close neighbour on its own surface).
    """
    pts, z = backproject(depth), depth.astype(np.float64)
    fly = np.zeros(depth.shape, bool)
    for ax in (0, 1):
        p, zz, m = np.moveaxis(pts, ax, 0), np.moveaxis(z, ax, 0), np.moveaxis(mask, ax, 0)
        gap = np.linalg.norm(p[1:] - p[:-1], axis=-1)           # 3D distance between neighbours
        lim = FLYING_REL_GAP * zz[1:-1]
        between = (zz[2:] - zz[1:-1]) * (zz[1:-1] - zz[:-2]) > 0
        np.moveaxis(fly, ax, 0)[1:-1] |= (gap[:-1] > lim) & (gap[1:] > lim) & between & m[2:] & m[:-2]
    return fly & mask


def boundary_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, edge: np.ndarray) -> dict:
    """AbsRel / RMSE restricted to the GT depth-edge band, and the flying-pixel share (% of valid pixels)."""
    out = {"edge_abs_rel": np.nan, "edge_rmse": np.nan}
    if int(edge.sum()) >= EDGE_MIN_PIXELS:
        m = depth_metrics(pred, gt, edge)
        out = {"edge_abs_rel": m["abs_rel"], "edge_rmse": m["rmse"]}
    out["flying_pct"] = float(100.0 * flying_pixel_mask(pred, mask)[mask].mean())
    return out


def load_nyu_index(nyu_dir, limit: int | None = None) -> pd.DataFrame:
    """NYU test index with rgb / depth paths; `limit` keeps evenly spaced images (smoke test)."""
    nyu_dir = Path(nyu_dir)
    df = pd.read_csv(nyu_dir / "index.csv", dtype={"name": str})
    df["rgb_path"] = [str(nyu_dir / "rgb" / f"{n}.png") for n in df["name"]]
    df["depth_path"] = [str(nyu_dir / "depth" / f"{n}.npy") for n in df["name"]]
    if limit is not None and limit < len(df):
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, limit).round().astype(int))]
    return df.reset_index(drop=True)


def eval_depth_run(model, processor, df: pd.DataFrame, degradation: str, inputs: list[str],
                   refine_for: list[str], device, desc: str = "") -> pd.DataFrame:
    """Run one depth model over the NYU images for one degradation.

    For every image: degrade -> for each input variant in `inputs` preprocess, predict, align,
    score (global + boundary metrics). Variants listed in `refine_for` are scored a second time
    after depth refinement (same forward pass). If two variants give the identical image (the
    noise-aware and the plain pipeline agree on clean inputs) the prediction is reused.
    One image is in memory at a time. Returns one row per (image, input variant, refined?).
    """
    from . import depth as dp
    from .preprocessing import load_rgb

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=desc, unit="img"):
        gt = np.load(r["depth_path"])
        mask = nyu_valid_mask(gt)
        edge = depth_edge_mask(gt, mask)
        img = degrade(load_rgb(r["rgb_path"]), degradation, seed=int(r["mat_index"]))
        done = {}                                   # variant -> (input image, disparity)
        for variant in inputs:
            x = dp.prepare_input(img, variant)
            twin = next((v for v, (xv, _) in done.items() if np.array_equal(xv, x)), None)
            disp = done[twin][1] if twin else dp.predict_disparity(model, processor, x, device)
            done[variant] = (x, disp)
            preds = {False: disp}
            if variant in refine_for:
                preds[True] = dp.refine_depth(disp, x)
            for refined, d in preds.items():
                depth = align_scale_shift(d, gt, mask)
                rows.append({"name": r["name"], "degradation": degradation, "input": variant,
                             "refined": refined, **depth_metrics(depth, gt, mask),
                             **boundary_metrics(depth, gt, mask, edge)})
    return pd.DataFrame(rows)


def summarize_depth(per_image: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean of each metric over images (the usual NYU protocol: per-image metric, then average)."""
    g = per_image.groupby(by, sort=False)
    out = g[ALL_DEPTH_METRICS].mean().reset_index()      # NaN edge metrics (no edges) are skipped
    out.insert(len(by), "n_images", g["name"].nunique().values)
    return out


def recovery_table(robust: pd.DataFrame) -> pd.DataFrame:
    """How much of the damage done by a degradation each ablation step wins back (one model).

    recovery % = (m[degraded, raw] - m[degraded, step]) / (m[degraded, raw] - m[clean, raw]) x 100
    100% = back to clean-image accuracy, 0% = no help, negative = made it worse.
    The same formula works for delta1 (higher is better) because both differences flip sign.
    """
    clean = robust[(robust["degradation"] == "clean") & (robust["condition"] == "raw")].iloc[0]
    rows = []
    for deg in [d for d in robust["degradation"].unique() if d != "clean"]:
        sub = robust[robust["degradation"] == deg]
        raw = sub[sub["condition"] == "raw"].iloc[0]
        for _, r in sub.iterrows():
            row = {"degradation": deg, "condition": r["condition"]}
            for m in DEPTH_METRICS:
                damage = raw[m] - clean[m]
                row[f"{m}_recovery_pct"] = 100 * (raw[m] - r[m]) / damage if abs(damage) > 1e-9 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def refinement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Before / after depth refinement, per model x degradation x input variant, with the change."""
    idx = ["model", "degradation"]
    rows = []
    for inp in REFINE_VARIANTS:
        before = summary[summary["condition"] == inp].set_index(idx)
        after = summary[summary["condition"] == f"{inp}_refine"].set_index(idx)
        for key in before.index.intersection(after.index):
            row = {"model": key[0], "degradation": key[1], "input": inp}
            for m in BOUNDARY_METRICS + ["abs_rel", "rmse"]:
                b, a = float(before.loc[key, m]), float(after.loc[key, m])
                row[m], row[f"{m}_refined"], row[f"{m}_change"] = b, a, a - b
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Denoising quality (PSNR / SSIM against the clean image)
# --------------------------------------------------------------------------- #
def denoise_quality_run(df: pd.DataFrame, levels=NOISE_LEVELS) -> pd.DataFrame:
    """PSNR and SSIM of each denoiser against the clean image, per image and noise level (CPU only).

    Methods: "noisy" (no filtering), "gaussian" (sigma 1), "bilateral_fixed" (sigma_color 40),
    "bilateral_auto" (sigma_color from the estimated noise = what the pipeline uses).
    """
    from . import preprocessing as pp
    methods = {"noisy": lambda x: x, "gaussian": pp.denoise_gaussian, "bilateral_fixed": pp.denoise_bilateral,
               "bilateral_auto": pp.denoise}
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="denoising PSNR / SSIM", unit="img"):
        clean = pp.load_rgb(r["rgb_path"])
        for sigma in levels:
            noisy = degrade(clean, f"noise{sigma}", seed=int(r["mat_index"]))
            for name, fn in methods.items():
                y = fn(noisy)
                rows.append({"name": r["name"], "noise_sigma": sigma, "method": name,
                             "psnr": pp.psnr(clean, y), "ssim": pp.ssim(clean, y),
                             "noise_est": pp.estimate_noise_sigma(y)})
    return pd.DataFrame(rows)


def summarize_denoise_quality(per_image: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over images per (noise level, method), plus the gain over the unfiltered image."""
    g = per_image.groupby(["noise_sigma", "method"], sort=False)
    out = g.agg(psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"), ssim_mean=("ssim", "mean"),
                ssim_std=("ssim", "std"), noise_est_mean=("noise_est", "mean"),
                n_images=("name", "nunique")).reset_index()
    base = out[out["method"] == "noisy"].set_index("noise_sigma")
    out["psnr_gain_db"] = out["psnr_mean"] - out["noise_sigma"].map(base["psnr_mean"])
    out["ssim_gain"] = out["ssim_mean"] - out["noise_sigma"].map(base["ssim_mean"])
    return out


# --------------------------------------------------------------------------- #
# Depth figures
# --------------------------------------------------------------------------- #
# One fixed colour per input variant (raw = neutral grey); "+ refinement" = same colour, hatched.
CONDITION_COLORS = {"raw": "#8a8a85", "denoise": "#2a78d6", "clahe": "#eb6834", "full": "#1baf7a",
                    "full_plain": "#8b5fd6", "full_refine": "#1baf7a", "full_plain_refine": "#8b5fd6"}
CONDITION_HATCH = {"full_refine": "///", "full_plain_refine": "///"}
CONDITION_SHORT = {"raw": "Raw", "denoise": "+Denoise", "clahe": "+CLAHE", "full": "+Full\n(aware)",
                   "full_plain": "+Full\n(plain)", "full_refine": "+Full aware\n+Refine",
                   "full_plain_refine": "+Full plain\n+Refine"}
METRIC_TITLES = {"abs_rel": "AbsRel  (lower is better)", "rmse": "RMSE in m  (lower is better)",
                 "delta1": "δ < 1.25  (higher is better)", "edge_abs_rel": "AbsRel at depth edges  (lower is better)",
                 "edge_rmse": "RMSE at depth edges, m  (lower is better)",
                 "flying_pct": "Flying pixels, %  (lower is better)"}


def _bar_panel(ax, labels, values, colors, fmt="{:.3f}", ref=None, ref_label=None, hatches=None):
    """Bars with value labels; optional dashed reference line."""
    x = np.arange(len(values))
    values = np.asarray(values, float)
    bars = ax.bar(x, values, width=0.64, color=colors, edgecolor="white", linewidth=1.5)
    for b, h in zip(bars, hatches or []):
        b.set_hatch(h)
    top = max(values.max(), ref or 0)
    ax.set_ylim(0, top * 1.16)
    for xi, v in zip(x, values):
        ax.text(xi, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8, color="#222")
    if ref is not None:
        ax.axhline(ref, color="#555", ls="--", lw=1.1, label=ref_label)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)


def _cond_style(conds):
    return ([CONDITION_SHORT[c] for c in conds], [CONDITION_COLORS[c] for c in conds],
            [CONDITION_HATCH.get(c, "") for c in conds])


def plot_depth_ablation(ablation: pd.DataFrame, out_path, model_labels: dict,
                        metrics=("abs_rel", "rmse", "delta1")) -> None:
    """Clean NYU images: one row per depth model, one panel per metric, one bar per ablation step.

    Panels of the same metric share the y-axis, so the two models can be compared directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(dict.fromkeys(ablation["model"]))
    conds = [c for c in CONDITION_COLORS if c in set(ablation["condition"])]
    labels, cols, hatches = _cond_style(conds)
    fig, axes = plt.subplots(len(model_names), len(metrics), figsize=(5.6 * len(metrics), 3.9 * len(model_names)),
                             squeeze=False, sharey="col")
    for r, name in enumerate(model_names):
        ab = ablation[ablation["model"] == name].set_index("condition").loc[conds]
        for c, m in enumerate(metrics):
            _bar_panel(axes[r, c], labels, list(ab[m]), cols, hatches=hatches)
            axes[r, c].set_ylim(0, float(ablation[ablation["condition"].isin(conds)][m].max()) * 1.16)
            axes[r, c].set_title(f"{model_labels.get(name, name)}: {METRIC_TITLES[m]}", fontsize=10)
    n = int(ablation["n_images"].iloc[0])
    fig.suptitle(f"Depth on NYU Depth V2 ({n} test images, clean): effect of each preprocessing step "
                 "(hatched = with depth refinement)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_robustness(robust: pd.DataFrame, out_path, model_label: str, metric: str = "abs_rel") -> None:
    """One panel per degradation (one model): the metric per ablation step; dashed line = clean raw image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degs = [d for d in DEGRADATIONS if d in set(robust["degradation"])]
    conds = [c for c in CONDITION_COLORS if c in set(robust["condition"])]
    labels, cols, hatches = _cond_style(conds)
    clean_raw = float(robust[(robust["degradation"] == "clean") & (robust["condition"] == "raw")][metric].iloc[0])
    top = float(robust[robust["condition"].isin(conds)][metric].max())
    ncol = 4
    nrow = int(np.ceil(len(degs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False, sharey=True)
    for ax in axes.ravel()[len(degs):]:
        ax.set_visible(False)
    for ax, deg in zip(axes.ravel(), degs):
        sub = robust[robust["degradation"] == deg].set_index("condition").loc[conds]
        _bar_panel(ax, labels, list(sub[metric]), cols, ref=clean_raw,
                   ref_label=f"clean raw image ({clean_raw:.3f})", hatches=hatches)
        ax.set_ylim(0, top * 1.16)
        ax.set_title(DEGRADATION_LABELS[deg], fontsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_TITLES[metric])
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    n = int(robust["n_images"].iloc[0])
    fig.suptitle(f"Robustness of {model_label} on degraded NYU images ({n} test images): effect of each "
                 "preprocessing step (hatched = with depth refinement)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
