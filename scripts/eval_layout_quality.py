"""PLAN_V3 milestone 7: layout-first image quality vs. layout adherence.

Rooms: the 40 bench layouts (scripts/make_bench.py; same prompts and generation seeds); layout seeds 1-4 = DEV,
5-8 = HELD-OUT. Configurations (the condition = inverse layout depth, layout.control_image):
    m6                 boxes, ControlNet scale 1.0 over all steps, SD 1.5 + 360 LoRA (= the bench panoramas)
    shapes_s{S}_e{E}   shaped proxies + door / window recesses (layout.render_full(shapes=True)),
                       ControlNet scale S in {1.0, 0.75, 0.5}, applied during the first E in {1.0, 0.6} of the steps
    sd21_s{S}_e{E}     the SD 2.1 panorama model (milestone 2's selected generator) + SD 2.1 depth ControlNet
                       (thibaud/controlnet-sd21-depth-diffusers), shaped proxies
    {sel}+i2i          the selected configuration followed by img2img at strength 0.3 without ControlNet
                       (texture pass; same model and seed)
Per panorama, against its own layout rendering:
    adherence  DAP distance (+ wrap fix) vs. layout distance: AbsRel and delta<1.25 as predicted, AbsRel after one
               median scale; SegFormer (14 views) vs. layout labels: IoU wall / floor / ceiling / furniture, mIoU,
               door and window IoU
    quality    CLIP score (8 horizontal 90 deg crops vs. the prompt); KID / FID of all crops of a configuration
               against the 500 MMIS hold-out images (as in milestone 2 of PLAN_V2)
Selection rule (fixed before running): among the DEV configurations with mean depth AbsRel (as predicted) <= 0.10,
the highest CLIP; ties within 0.1 CLIP -> lower KID. The selected configuration (and the best SD 2.1 one, and the
img2img pass) are then run on the HELD-OUT layouts.
Outputs: outputs/layout_quality/ (intermediate), results/tables/lq_*.csv, results/figures/lq_*.png
Usage: python scripts/eval_layout_quality.py [--limit N]
"""

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))
from src import analysis as an  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import layout as ly  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src.utils import set_seed  # noqa: E402
import make_bench as mb  # noqa: E402

OUT = ROOT / "outputs" / "layout_quality"
DEV_CONFIGS = {"m6": dict(shapes=False, scale=1.0, end=1.0, model="sd15")}
for s_ in (1.0, 0.75, 0.5):
    for e_ in (1.0, 0.6):
        DEV_CONFIGS[f"shapes_s{s_}_e{e_}"] = dict(shapes=True, scale=s_, end=e_, model="sd15")
for s_, e_ in ((1.0, 1.0), (0.75, 0.6)):
    DEV_CONFIGS[f"sd21_s{s_}_e{e_}"] = dict(shapes=True, scale=s_, end=e_, model="sd21")
I2I_STRENGTH = 0.3
GROUPS = {"wall": [0], "floor": [3, 28], "ceiling": [5], "door": [14, 58], "window": [8]}


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def crops8(pano):
    return [eq.e2p(pano, 90, 45 * i, 0, 384) for i in range(8)]


def generate(cfgs: dict, rooms: pd.DataFrame, device):
    """Panoramas of every (config, room) not yet on disk, one model at a time."""
    for model in ("sd15", "sd21"):
        todo = []
        for name, c in cfgs.items():
            if c["model"] != model:
                continue
            for r in rooms.itertuples():
                d = OUT / name / r.id
                if name == "m6" or (d / "pano.png").exists():
                    continue
                todo.append((name, c, r, d))
        if not todo:
            continue
        pipe = gp.load_controlnet_pipeline(device) if model == "sd15" else gp.load_controlnet_pipeline_sd21(device)
        i2i = None
        for name, c, r, d in tqdm(todo, desc=f"generate {model}", unit="pano"):
            d.mkdir(parents=True, exist_ok=True)
            L = json.loads((mb.OUT / r.id / "layout.json").read_text())
            dist, _, _ = ly.render_full(L, shapes=c["shapes"])
            ctrl = ly.control_image(dist)
            kw = {} if model == "sd15" else {"prompt": gp.ALT_TEMPLATE.format(room=r.prompt), "negative": gp.PANO_NEGATIVE}
            img = gp.generate_pano_controlled(pipe, r.prompt, r.gen_seed, ctrl, c["scale"], end=c["end"], **kw)
            if c.get("i2i"):
                if i2i is None:
                    from diffusers import StableDiffusionImg2ImgPipeline
                    i2i = StableDiffusionImg2ImgPipeline(**{k: v for k, v in pipe.components.items()
                                                            if k in ("vae", "text_encoder", "tokenizer", "unet",
                                                                     "scheduler", "feature_extractor")},
                                                         safety_checker=None, requires_safety_checker=False)
                    i2i.set_progress_bar_config(disable=True)
                img = img2img(i2i, img, gp.pano_prompt(r.prompt) if model == "sd15" else kw["prompt"], r.gen_seed)
            cv2.imwrite(str(d / "pano.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        del pipe, i2i
        ev.release_memory()


@torch.inference_mode()
def img2img(pipe, img: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    """Texture pass: img2img at I2I_STRENGTH without ControlNet, circular padding on."""
    from PIL import Image
    gp.install_circular(pipe.unet, pipe.vae)
    gp.set_circular(True, pipe.unet, pipe.vae)
    gen = torch.Generator(pipe.device).manual_seed(int(seed))
    out = pipe(prompt, image=Image.fromarray(img), strength=I2I_STRENGTH, negative_prompt=gp.PANO_NEGATIVE,
               num_inference_steps=gp.PANO_STEPS, guidance_scale=gp.PANO_GUIDANCE, generator=gen).images[0]
    gp.set_circular(False, pipe.unet, pipe.vae)
    return np.asarray(out)


def pano_path(name, rid):
    return mb.OUT / rid / "pano.png" if name == "m6" else OUT / name / rid / "pano.png"


def analyse(cfgs: dict, rooms: pd.DataFrame, device) -> pd.DataFrame:
    """DAP + SegFormer + CLIP per panorama (cached), crops for KID."""
    todo = [(n, r) for n in cfgs for r in rooms.itertuples() if not (OUT / n / r.id / "dist.npy").exists()]
    if todo:
        dap, _ = pdp.load_dap(device)
        for n, r in tqdm(todo, desc="DAP", unit="pano"):
            (OUT / n / r.id).mkdir(parents=True, exist_ok=True)
            np.save(OUT / n / r.id / "dist.npy", pdp.dap_depth_wrap(dap, mb.rgb(pano_path(n, r.id)), device))
        del dap
        ev.release_memory()
    seg = an.load_segmenter(device)
    clip = ev.load_clip(device)
    from src.instances import family_of
    furn_ids = sorted(family_of({int(k): v for k, v in seg[0].config.id2label.items()}))   # all furniture classes
    rows = []
    for n, c in cfgs.items():
        for r in tqdm(list(rooms.itertuples()), desc=f"metrics {n}", unit="pano"):
            d = OUT / n / r.id
            pano = mb.rgb(pano_path(n, r.id))
            if not (d / "seg.png").exists():
                cv2.imwrite(str(d / "seg.png"), pdp.segment_pano(*seg, pano, device).astype(np.uint8))
            cd = d / "crops"
            if not cd.exists():
                cd.mkdir()
                for i, cimg in enumerate(crops8(pano)):
                    cv2.imwrite(str(cd / f"{i}.png"), cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR))
            L = json.loads((mb.OUT / r.id / "layout.json").read_text())
            gt_d, gt_l, _ = ly.render_full(L, shapes=c["shapes"])
            pred = np.load(d / "dist.npy")
            m = gt_d > 0.1
            p, g = pred[m], gt_d[m]
            ps = p * np.median(g / p)
            lab = cv2.imread(str(d / "seg.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
            ious = {k: iou(np.isin(lab, ids), np.isin(gt_l, ids)) for k, ids in GROUPS.items()}
            ious["furniture"] = iou(np.isin(lab, furn_ids), np.isin(gt_l, furn_ids))
            crops = [str(cd / f"{i}.png") for i in range(8)]
            rows.append({"config": n, "id": r.id, "split": r.split, "room": r.room,
                         "absrel_raw": float(np.mean(np.abs(p - g) / g)),
                         "delta1_raw": float(np.mean(np.maximum(p / g, g / p) < 1.25)),
                         "absrel_scaled": float(np.mean(np.abs(ps - g) / g)),
                         **{f"iou_{k}": v for k, v in ious.items()},
                         "miou": float(np.nanmean([ious[k] for k in ("wall", "floor", "ceiling", "furniture")])),
                         "clip": float(ev.clip_scores(*clip, crops, [r.prompt] * 8, device).mean())})
    del seg, clip
    ev.release_memory()
    return pd.DataFrame(rows)


def kid_table(cfgs, rooms, device, split) -> pd.DataFrame:
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    sets = {n: [str(OUT / n / r.id / "crops" / f"{i}.png") for r in rooms.itertuples() for i in range(8)] for n in cfgs}
    fk = ev.fid_kid(list(hold["real_path"]), sets, device)
    return pd.DataFrame([{"config": n, "split": split, "kid_x1e3": 1e3 * v["kid_mean"], "kid_std_x1e3": 1e3 * v["kid_std"],
                          "fid": v["fid"], "crops": len(sets[n])} for n, v in fk.items()])


def summarize(per: pd.DataFrame, kid: pd.DataFrame) -> pd.DataFrame:
    cols = ["absrel_raw", "delta1_raw", "absrel_scaled", "iou_furniture", "iou_door", "iou_window", "miou", "clip"]
    s = per.groupby(["split", "config"])[cols].mean().reset_index()
    return s.merge(kid, on=["split", "config"], how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    tag = "" if args.limit is None else "_smoke"
    rooms = mb.rooms()
    if args.limit:
        rooms = rooms.groupby("split").head(args.limit)
    dev, test = rooms[rooms["split"] == "dev"], rooms[rooms["split"] == "test"]

    generate(DEV_CONFIGS, dev, device)
    per_dev = analyse(DEV_CONFIGS, dev, device)
    s_dev = summarize(per_dev, kid_table(DEV_CONFIGS, dev, device, "dev"))
    ok = s_dev[s_dev["absrel_raw"] <= 0.10].sort_values("clip", ascending=False)
    top = ok[ok["clip"] >= ok["clip"].max() - 0.1].sort_values("kid_x1e3") if len(ok) else s_dev.sort_values("absrel_raw")
    sel = str(top.iloc[0]["config"])
    sd21 = s_dev[s_dev["config"].str.startswith("sd21")].sort_values("clip", ascending=False)
    best21 = str(sd21.iloc[0]["config"]) if len(sd21) else None
    print(s_dev.round(3).to_string(index=False))
    print("selected on dev:", sel, "| best SD 2.1:", best21)

    i2i_name = f"{sel}+i2i"
    test_cfgs = {"m6": DEV_CONFIGS["m6"], "shapes_s1.0_e1.0": DEV_CONFIGS["shapes_s1.0_e1.0"], sel: DEV_CONFIGS[sel],
                 i2i_name: {**DEV_CONFIGS[sel], "i2i": True}}
    if best21:
        test_cfgs[best21] = DEV_CONFIGS[best21]
    # the img2img pass is also measured on dev (decision: keep it if dev CLIP rises and dev AbsRel stays <= 0.10)
    generate({i2i_name: test_cfgs[i2i_name]}, dev, device)
    per_dev_i2i = analyse({i2i_name: test_cfgs[i2i_name]}, dev, device)
    s_dev = pd.concat([s_dev, summarize(per_dev_i2i, kid_table({i2i_name: 0}, dev, device, "dev"))])
    generate(test_cfgs, test, device)
    per_test = analyse(test_cfgs, test, device)
    s_test = summarize(per_test, kid_table(test_cfgs, test, device, "test"))
    summ = pd.concat([s_dev, s_test])
    T, F = ROOT / "results/tables", ROOT / "results/figures"
    pd.concat([per_dev, per_dev_i2i, per_test]).to_csv(T / f"lq_per_pano{tag}.csv", index=False, float_format="%.4f")
    summ.to_csv(T / f"lq_summary{tag}.csv", index=False, float_format="%.4f")
    (T / f"lq_decision{tag}.json").write_text(json.dumps({"selected": sel, "best_sd21": best21}))
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))

    # figure: held-out rooms, condition + panoramas of each test configuration
    names = list(test_cfgs)
    ex = test.groupby("room").head(1)
    fig, axes = plt.subplots(len(ex), len(names) + 1, figsize=(5.2 * (len(names) + 1), 2.9 * len(ex)), dpi=70)
    axes = np.atleast_2d(axes)
    for i, r in enumerate(ex.itertuples()):
        L = json.loads((mb.OUT / r.id / "layout.json").read_text())
        axes[i, 0].imshow(ly.control_image(ly.render_full(L, shapes=True)[0]))
        axes[i, 0].set_title(f"shaped condition — {r.prompt}" if i == 0 else r.prompt, fontsize=10)
        for j, n in enumerate(names):
            axes[i, j + 1].imshow(mb.rgb(pano_path(n, r.id)))
            if i == 0:
                axes[i, j + 1].set_title(n, fontsize=11)
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(F / f"lq_examples{tag}.png", facecolor="white")
    plt.close(fig)
    # trade-off plot (dev)
    d = s_dev.copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=110)
    for _, row in d.iterrows():
        col = "#2a78d6" if row["config"].startswith("shapes") else "#eb6834" if row["config"].startswith("sd21") else "#555555"
        ax.scatter(row["absrel_raw"], row["clip"], s=60, color=col, edgecolor="k" if row["config"] == sel else "none", lw=1.5)
        ax.annotate(row["config"], (row["absrel_raw"], row["clip"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.10, color="#999999", ls="--", lw=1)
    ax.set_xlabel("Depth AbsRel vs. layout (lower = follows the layout)")
    ax.set_ylabel("CLIP score (higher = matches the prompt)")
    ax.set_title("Layout-first, dev layouts: adherence vs. image-text alignment")
    fig.tight_layout()
    fig.savefig(F / f"lq_tradeoff{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
