"""Milestone 2 evaluation: 360 degree panorama generation and super-resolution.

Prompts: 5 rooms x 6 styles ("a <style> style <room>", the LoRA caption form) x 2 seeds = 60 panoramas
per configuration, 1024 x 512, DDIM 30 steps, CFG 7.5, fixed seeds.

Configurations (panorama LoRA always at weight 1):
  w0_nocirc   MMIS LoRA 0,   no circular padding                (seam baseline)
  w0_circ50   MMIS LoRA 0,   circular padding from 50% of the steps
  w0_circ     MMIS LoRA 0,   circular padding from the start
  w05_circ / w08_circ / w10_circ   MMIS LoRA 0.5 / 0.8 / 1.0, circular from the start
  alt_sd21_circ  alternative generator: SD 2.1-base panorama fine-tune (archerfmy0831/sd-t2i-360panoimage),
                 circular from the start, prompt "360 panorama, interior, <caption>, photo, high quality"

Metrics:
  seam ratio     mean |grey difference| across the wrap / mean between neighbouring columns (1 = no seam)
  CLIP score     ViT-L/14, caption without the trigger word, mean over 4 perspective crops (90 deg FOV,
                 yaw 0/90/180/270, pitch 0, 512 px)
  FID / KID      the 240 crops of a configuration (resized to 256) against the 500 MMIS hold-out images
                 (same protocol as the 2D evaluation)
Selection rule (fixed before the run): among the circular-from-start configurations, lowest mean rank
of KID (lower better) and CLIP (higher better); ties -> lower KID. (The alternative generator was
added to the candidates after the first six configurations were run, but before it was run itself.)

Super-resolution (chosen configuration, first seed = 30 panoramas): Real-ESRGAN x2 (tiled, wrap-padded)
vs bicubic x2 + unsharp mask. Sharpness = Laplacian variance at 2048 x 1024; fidelity = PSNR of the
result downscaled back to 1024 x 512 against the input; seam ratio; seconds per panorama.

Outputs: outputs/pano_eval/<config>/<id>.png (+ crops/), results/tables/pano_*.csv,
         results/figures/pano_*.png
Usage: python scripts/eval_pano.py [--limit 4]
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src.utils import set_seed  # noqa: E402

ROOMS = ["bedroom", "living room", "kitchen", "dining room", "bathroom"]
STYLES = ["art deco", "bohemian", "coastal", "farmhouse", "mid-century modern", "rustic"]
SEEDS = [1, 2]
CONFIGS = {  # name: (mmis weight, circular, circular_from)
    "w0_nocirc": (0.0, False, 0.0), "w0_circ50": (0.0, True, 0.5), "w0_circ": (0.0, True, 0.0),
    "w05_circ": (0.5, True, 0.0), "w08_circ": (0.8, True, 0.0), "w10_circ": (1.0, True, 0.0),
    "alt_sd21_circ": (None, True, 0.0),        # alternative generator (SD 2.1 panorama fine-tune), no LoRA
}
SWEEP = ["w0_circ", "w05_circ", "w08_circ", "w10_circ", "alt_sd21_circ"]
OUT = ROOT / "outputs" / "pano_eval"


def prompt_table(limit=None) -> pd.DataFrame:
    rows = []
    for s in STYLES:
        for r in ROOMS:
            art = "an" if s[0] in "aeiou" else "a"
            for seed in SEEDS:
                rows.append({"caption": f"{art} {s} style {r}", "style": s, "room": r, "seed": seed})
    df = pd.DataFrame(rows)
    df["id"] = [f"{i:03d}" for i in range(len(df))]
    if limit:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, limit).round().astype(int))].reset_index(drop=True)
    return df


def generate_all(df, device) -> pd.DataFrame:
    times = []
    for alt in (False, True):
        todo = [(c, r) for c in CONFIGS if (CONFIGS[c][0] is None) == alt
                for _, r in df.iterrows() if not (OUT / c / f"{r['id']}.png").exists()]
        if not todo:
            continue
        pipe = gp.load_alt_pipeline(device) if alt else gp.load_pano_pipeline(device)
        for c, r in tqdm(todo, desc="panoramas" + (" (alt)" if alt else ""), unit="pano"):
            w, circ, frm = CONFIGS[c]
            img, t = gp.timed(gp.generate_pano, pipe, r["caption"], 1000 + int(r["seed"]) * 7919 + int(r["id"]),
                              mmis_weight=w or 0.0, circular=circ, circular_from=frm,
                              prompt=gp.ALT_TEMPLATE.format(room=r["caption"]) if alt else None)
            (OUT / c / "crops").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / c / f"{r['id']}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            for k, v in enumerate(eq.horizontal_views(img)):
                cv2.imwrite(str(OUT / c / "crops" / f"{r['id']}_{k}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            times.append({"config": c, "id": r["id"], "seconds": t, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
        del pipe
        ev.release_memory()
    return pd.DataFrame(times)


def load(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def metrics(df, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = []
    for c in CONFIGS:
        for _, r in df.iterrows():
            per.append({"config": c, "id": r["id"], "caption": r["caption"],
                        "seam_ratio": eq.seam_ratio(load(OUT / c / f"{r['id']}.png"))})
    per = pd.DataFrame(per)
    crops = {c: [str(OUT / c / "crops" / f"{i}_{k}.png") for i in df["id"] for k in range(4)] for c in CONFIGS}
    caps = [cap for cap in df["caption"] for _ in range(4)]
    model, proc = ev.load_clip(device)
    clip = {c: ev.clip_scores(model, proc, p, caps, device).reshape(-1, 4).mean(1) for c, p in crops.items()}
    del model, proc
    ev.release_memory()
    per["clip"] = np.concatenate([clip[c] for c in CONFIGS])
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    fk = ev.fid_kid(list(hold["real_path"]), crops, device)
    summ = per.groupby("config", sort=False).agg(seam_mean=("seam_ratio", "mean"), seam_median=("seam_ratio", "median"),
                                                 clip_mean=("clip", "mean"), clip_std=("clip", "std")).reset_index()
    summ["fid"] = [fk[c]["fid"] for c in summ["config"]]
    summ["kid_x1e3"] = [1e3 * fk[c]["kid_mean"] for c in summ["config"]]
    summ["kid_std_x1e3"] = [1e3 * fk[c]["kid_std"] for c in summ["config"]]
    summ["n_panos"], summ["n_crops"] = len(df), 4 * len(df)
    return per, summ


def select(summ: pd.DataFrame) -> str:
    s = summ[summ["config"].isin(SWEEP)].copy()
    s["rank"] = s["kid_x1e3"].rank() + s["clip_mean"].rank(ascending=False)
    return s.sort_values(["rank", "kid_x1e3"]).iloc[0]["config"]


def super_resolution(df, cfg: str, device) -> tuple[pd.DataFrame, dict]:
    net = gp.load_esrgan(device)
    rows, example = [], {}
    sub = df[df["seed"] == SEEDS[0]]
    (OUT / "sr").mkdir(parents=True, exist_ok=True)
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc="super-resolution", unit="pano"):
        img = load(OUT / cfg / f"{r['id']}.png")
        for name, fn in (("realesrgan_x2", lambda x: gp.esrgan_upscale(net, x)), ("bicubic_unsharp", gp.bicubic_unsharp)):
            up, t = gp.timed(fn, img)
            down = cv2.resize(up, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            rows.append({"method": name, "id": r["id"], "seconds": t, "laplacian_var": eq.laplacian_variance(up),
                         "psnr_down_db": cv2.PSNR(down, img), "seam_ratio": eq.seam_ratio(up),
                         "width": up.shape[1], "height": up.shape[0]})
            if r["id"] == sub["id"].iloc[0]:
                example[name] = up
                cv2.imwrite(str(OUT / "sr" / f"{r['id']}_{name}.png"), cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
        if r["id"] == sub["id"].iloc[0]:
            example["input"] = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    del net
    ev.release_memory()
    return pd.DataFrame(rows), example


def fig_grid(df, path: Path, n: int = 4) -> None:
    ids = df["id"].iloc[np.linspace(0, len(df) - 1, n).round().astype(int)].tolist()
    cfgs = list(CONFIGS)
    fig, axes = plt.subplots(len(ids), len(cfgs), figsize=(4.2 * len(cfgs), 2.4 * len(ids)), dpi=100)
    for i, pid in enumerate(ids):
        for j, c in enumerate(cfgs):
            img = load(OUT / c / f"{pid}.png")
            img = np.roll(img, img.shape[1] // 2, axis=1)       # seam in the middle
            ax = axes[i, j]
            ax.imshow(img)
            ax.axvline(img.shape[1] / 2, color="red", lw=0.5, ls=":")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                ax.set_ylabel(df[df["id"] == pid]["caption"].iloc[0], fontsize=8)
    fig.suptitle("Panoramas rolled by half a width: the wrap seam is at the dotted red line", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_metrics(summ: pd.DataFrame, best: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), dpi=110)
    for ax, (col, title, err) in zip(axes, [("seam_median", "Seam ratio (median, 1 = seamless)", None),
                                            ("clip_mean", "CLIP score (crops) ↑", "clip_std"),
                                            ("kid_x1e3", "KID ×10³ vs MMIS ↓", "kid_std_x1e3"),
                                            ("fid", "FID vs MMIS ↓", None)]):
        colors = ["#1baf7a" if c == best else "#8a8a85" for c in summ["config"]]
        ax.bar(summ["config"], summ[col], color=colors, yerr=summ[err] if err else None, capsize=3)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        for k, v in enumerate(summ[col]):
            ax.text(k, v, f"{v:.2f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Panorama configurations (green = selected: {best})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_sr(ex: dict, path: Path) -> None:
    h, w = ex["input"].shape[:2]
    y0, x0, s = h // 2 - 160, w // 2 - 240, 320
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=110)
    for ax, (k, t) in zip(axes, [("input", "Input 1024×512 (nearest ×2)"), ("bicubic_unsharp", "Bicubic ×2 + unsharp"),
                                 ("realesrgan_x2", "Real-ESRGAN ×2 (wrap-padded tiles)")]):
        ax.imshow(ex[k][y0:y0 + s, x0:x0 + int(1.5 * s)])
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    tag = "" if args.limit is None else "_smoke"
    df = prompt_table(args.limit)
    times = generate_all(df, device)
    per, summ = metrics(df, device)
    best = select(summ)
    sr, ex = super_resolution(df, best, device)
    sr_sum = sr.groupby("method", sort=False)[["seconds", "laplacian_var", "psnr_down_db", "seam_ratio"]].mean().reset_index()
    tables, figs = ROOT / "results/tables", ROOT / "results/figures"
    if len(times):
        summ = summ.merge(times.groupby("config")["seconds"].mean().rename("sec_per_pano").reset_index(), on="config", how="left")
    summ["selected"] = summ["config"] == best
    per.to_csv(tables / f"pano_per_image{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(tables / f"pano_configs{tag}.csv", index=False, float_format="%.4f")
    sr.to_csv(tables / f"pano_sr_per_image{tag}.csv", index=False, float_format="%.4f")
    sr_sum.to_csv(tables / f"pano_sr{tag}.csv", index=False, float_format="%.4f")
    fig_grid(df, figs / f"pano_configs_grid{tag}.png")
    fig_metrics(summ, best, figs / f"pano_configs_metrics{tag}.png")
    fig_sr(ex, figs / f"pano_sr_compare{tag}.png")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print("selected:", best)
    print(sr_sum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
