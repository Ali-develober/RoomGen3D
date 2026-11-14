"""PLAN_V4 milestone 2: a sharper panorama - comparison of refinement configurations.

Rooms: DEVELOPMENT = 10 bench rooms (layout seeds 1-2 of each room type) for the choice; REPORT = 10 held-out
bench rooms (seeds 5-6) + the six demo rooms (SD 2.1 panoramas of the 360° mode) for the selected configuration
against the baselines. Input everywhere: the Real-ESRGAN texture (2048 x 1024) that v3 uses.
Configurations (src/pano_refine.py):
    v3_texture        the 2048 x 1024 texture itself (no refinement; what v3 / milestone 1 used)
    bicubic           26 views, bicubic x2, blended to 4096 x 2048
    esrgan            26 views, Real-ESRGAN x2, blended
    i2i_{base,lora}_{0.2,0.3}  img2img on the 512 views (SD 1.5 without / with the MMIS LoRA, strength), then
                      Real-ESRGAN x2, blended (low-frequency lock on)
Measured per room (8 level design shots, 80° HFOV, 1024 x 768, yaw 0, 45, ..., in the panorama frame):
    sharpness   Laplacian variance of the shots (higher = more detail)
    brisque     BRISQUE no-reference quality of the shots (piq; lower = better)
    kid / fid   shots vs. the 500 MMIS hold-out photos (Inception at 256 px, so it sees content / style, not
                fine detail), CLIP score of shots vs. the room prompt
    drift       structural edge F1 (Canny after 1.5 px blur at 1024 x 512, 2 px tolerance) and DAP depth AbsRel against the input texture
    seams       overlap disagreement of neighbouring views (before blending), boundary gradient ratio and
                wrap seam ratio of the result (after blending)
    cost        seconds per room, peak VRAM
Selection rule (fixed before running): among configurations whose drift stays within the guard (edge F1 >= 0.9 x
the esrgan configuration's, depth AbsRel <= 2 x esrgan's + 0.01), the best mean rank of sharpness, BRISQUE, KID
and CLIP on the development rooms.
Outputs: outputs/pano_refine/<config>/<room>.png (+ shots) [deleted after the tables / figures are saved];
results/tables/pano_refine_per_room.csv, pano_refine_dev.csv, pano_refine_report.csv, pano_refine_decision.json;
figure pano_refine_crops.png.
Usage: python scripts/eval_pano_refine.py [--stage gen|metrics|report|all] [--limit N]
"""

import argparse
import json
import sys
import time
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
from src import design_shots as ds  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_2d as g2  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
import make_bench as mb  # noqa: E402

OUT = ROOT / "outputs" / "pano_refine"
TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEV_CONFIGS = ["v3_texture", "bicubic", "esrgan", "i2i_base_0.2", "i2i_base_0.3", "i2i_lora_0.2", "i2i_lora_0.3"]
SHOT_YAWS = [45.0 * k for k in range(8)]
DEVICE = torch.device("cuda")


def rooms(limit=None) -> pd.DataFrame:
    b = mb.rooms()
    rows = []
    for r in b.itertuples():
        if r.layout_seed in (1, 2):
            rows.append({"id": r.id, "set": "dev", "prompt": r.prompt, "tex": mb.OUT / r.id / "texture.png",
                         "dist": mb.OUT / r.id / "dist.npy", "seed": int(r.gen_seed)})
        elif r.layout_seed in (5, 6):
            rows.append({"id": r.id, "set": "report", "prompt": r.prompt, "tex": mb.OUT / r.id / "texture.png",
                         "dist": mb.OUT / r.id / "dist.npy", "seed": int(r.gen_seed)})
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        rows.append({"id": "demo_" + pl.slug(prompt, seed), "set": "report", "prompt": prompt,
                     "tex": d / "texture.png", "dist": d / "distance.npy", "seed": int(seed)})
    df = pd.DataFrame(rows)
    if limit:
        df = df.groupby("set").head(limit)
    return df


def parse(cfg: str):
    if not cfg.startswith("i2i"):
        return None, None
    _, kind, s = cfg.split("_")
    return kind, float(s)


def shots(pano: np.ndarray) -> list[np.ndarray]:
    I = {"R": np.eye(3), "Q": np.eye(3)}
    return [ds.render_view(pano, ds.view_to_cam(I, y), ds.SHOT_W, ds.SHOT_H, ds.HERO_FOV) for y in SHOT_YAWS]


def stage_gen(df: pd.DataFrame, configs_for: dict):
    """Refined panoramas + their shots + per-room cost / seam numbers (resumable)."""
    net = gp.load_esrgan(DEVICE)
    log_p = OUT / "gen_log.csv"
    done = pd.read_csv(log_p) if log_p.exists() else pd.DataFrame(columns=["config", "id"])
    by_model = {}
    for cfg in sorted({c for cs in configs_for.values() for c in cs}):
        by_model.setdefault(parse(cfg)[0], []).append(cfg)
    for kind, cfgs in by_model.items():                  # one SD model on the GPU at a time
        todo = [(c, r) for c in cfgs for r in df.itertuples() if c in configs_for[r.set]
                and not ((done["config"] == c) & (done["id"] == r.id)).any()]
        if not todo:
            continue
        pipe = pr.load_img2img(DEVICE, g2.DEFAULT_LORA if kind == "lora" else None) if kind else None
        for cfg, r in tqdm(todo, desc=f"refine ({kind or 'no sd'})"):
            tex = mb.rgb(r.tex)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            row = {"config": cfg, "id": r.id}
            if cfg == "v3_texture":
                pano, info = tex, None
            else:
                _, s = parse(cfg)
                pano, info = pr.refine(tex, "bicubic" if cfg == "bicubic" else net, pipe, r.prompt, s or 0.0, r.seed)
                row.update({f"t_{k}": v for k, v in info["seconds"].items()})
                row["overlap_diff"] = pr.overlap_disagreement(info["views"])
                row["boundary_ratio"] = pr.boundary_ratio(pano, info["argmax_view"])
            row["seconds"] = time.perf_counter() - t0
            row["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            row["seam_ratio"] = eq.seam_ratio(pano)
            d = OUT / cfg
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{r.id}.png"), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
            for k, sh in enumerate(shots(pano)):
                cv2.imwrite(str(d / f"{r.id}_shot{k}.png"), cv2.cvtColor(sh, cv2.COLOR_RGB2BGR))
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(log_p, index=False)
        del pipe
        ev.release_memory()
    del net
    ev.release_memory()


def stage_metrics(df: pd.DataFrame):
    """Per room x config: sharpness, BRISQUE, edge F1, DAP depth AbsRel, CLIP (one model at a time)."""
    import piq
    gen = pd.read_csv(OUT / "gen_log.csv")
    met_p = OUT / "metrics_room.csv"
    met = pd.read_csv(met_p) if met_p.exists() else pd.DataFrame(columns=["config", "id"])
    todo = [r for r in gen.itertuples() if not ((met["config"] == r.config) & (met["id"] == r.id)).any()]
    rows = []
    rmap = df.set_index("id")
    for g in tqdm(todo, desc="image metrics"):
        r = rmap.loc[g.id]
        pano = mb.rgb(OUT / g.config / f"{g.id}.png")
        tex = mb.rgb(r.tex)
        sh = [mb.rgb(OUT / g.config / f"{g.id}_shot{k}.png") for k in range(len(SHOT_YAWS))]
        x = torch.from_numpy(np.stack(sh)).permute(0, 3, 1, 2).float().div(255)
        bq = float(np.mean([float(piq.brisque(x[k:k + 1], data_range=1.0)) for k in range(len(sh))]))
        rows.append({"config": g.config, "id": g.id, "sharpness": float(np.mean([eq.laplacian_variance(s) for s in sh])),
                     "brisque": bq, "edge_f1": pr.edge_f1(tex, pano)})
    if rows:
        new = pd.DataFrame(rows)
        dap = pdp.load_dap(DEVICE)[0]
        ab = []
        for r in tqdm(new.itertuples(), total=len(new), desc="DAP drift"):
            ref = np.load(rmap.loc[r.id, "dist"])
            dd = pdp.dap_depth_wrap(dap, mb.rgb(OUT / r.config / f"{r.id}.png"), DEVICE)
            dd = cv2.resize(dd, ref.shape[::-1], interpolation=cv2.INTER_AREA)
            ab.append(float(np.median(np.abs(dd - ref) / ref)))
        new["depth_absrel"] = ab
        del dap
        ev.release_memory()
        met = pd.concat([met, new], ignore_index=True)
        met.to_csv(met_p, index=False)
    # CLIP per shot (cheap, recomputed)
    clip_m, proc = ev.load_clip(DEVICE)
    cl = []
    for r in met.itertuples():
        paths = [str(OUT / r.config / f"{r.id}_shot{k}.png") for k in range(len(SHOT_YAWS))]
        cl.append(float(ev.clip_scores(clip_m, proc, paths, [rmap.loc[r.id, "prompt"]] * len(paths), DEVICE).mean()))
    met["clip"] = cl
    del clip_m
    ev.release_memory()
    out = met.merge(gen, on=["config", "id"]).merge(df[["id", "set"]], on="id")
    out.to_csv(TAB / "pano_refine_per_room.csv", index=False)
    return out


def kid_table(per: pd.DataFrame, subset: str) -> pd.DataFrame:
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    sets = {}
    for cfg, g in per[per["set"] == subset].groupby("config"):
        sets[cfg] = [str(OUT / cfg / f"{i}_shot{k}.png") for i in g["id"] for k in range(len(SHOT_YAWS))]
    fk = ev.fid_kid(list(hold["real_path"]), sets, DEVICE)
    return pd.DataFrame([{"config": k, "kid_x1e3": v["kid_mean"] * 1e3, "kid_std_x1e3": v["kid_std"] * 1e3,
                          "fid": v["fid"]} for k, v in fk.items()])


def summarize(per: pd.DataFrame, subset: str) -> pd.DataFrame:
    cols = ["sharpness", "brisque", "clip", "edge_f1", "depth_absrel", "overlap_diff", "boundary_ratio",
            "seam_ratio", "seconds", "peak_vram_gb"]
    s = per[per["set"] == subset].groupby("config")[cols].mean().reset_index()
    return s.merge(kid_table(per, subset), on="config")


def select(dev: pd.DataFrame) -> dict:
    e = dev.set_index("config").loc["esrgan"]
    ok = dev[(dev["edge_f1"] >= 0.9 * e["edge_f1"]) & (dev["depth_absrel"] <= 2 * e["depth_absrel"] + 0.01)].copy()
    ranks = pd.DataFrame({"sharpness": ok["sharpness"].rank(ascending=False), "brisque": ok["brisque"].rank(),
                          "kid": ok["kid_x1e3"].rank(), "clip": ok["clip"].rank(ascending=False)})
    ok["mean_rank"] = ranks.mean(1)
    best = ok.sort_values("mean_rank").iloc[0]
    return {"selected": best["config"], "passed_guard": list(ok["config"]),
            "failed_guard": sorted(set(dev["config"]) - set(ok["config"])),
            "mean_rank": dict(zip(ok["config"], ok["mean_rank"]))}


def figures(per: pd.DataFrame, configs: list[str], room_ids: list[str]):
    fig, axes = plt.subplots(len(room_ids), len(configs), figsize=(3.6 * len(configs), 3.0 * len(room_ids)), dpi=80,
                             squeeze=False)
    for i, rid in enumerate(room_ids):
        for j, c in enumerate(configs):
            sh = mb.rgb(OUT / c / f"{rid}_shot0.png")
            h, w = sh.shape[:2]
            crop = sh[h // 2 - 150:h // 2 + 150, w // 2 - 200:w // 2 + 200]          # 1:1 pixels, centre
            axes[i, j].imshow(crop)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(c, fontsize=11)
        axes[i, 0].text(-10, 150, rid, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("Centre crops (400 x 300 px at 1:1) of the 1024 x 768 design shot, by panorama refinement", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "pano_refine_crops.png", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = rooms(a.limit)
    dec_p = TAB / "pano_refine_decision.json"
    if a.stage in ("gen", "all"):
        stage_gen(df, {"dev": DEV_CONFIGS, "report": []})
    if a.stage in ("metrics", "all"):
        per = stage_metrics(df)
        dev = summarize(per, "dev")
        dev.to_csv(TAB / "pano_refine_dev.csv", index=False)
        print(dev.round(3).to_string())
        dec = select(dev)
        dec_p.write_text(json.dumps(dec, indent=1))
        print(dec)
    if a.stage in ("report", "all"):
        dec = json.loads(dec_p.read_text())
        rep_cfgs = sorted({"v3_texture", "esrgan", dec["selected"]})
        stage_gen(df, {"dev": DEV_CONFIGS, "report": rep_cfgs})
        per = stage_metrics(df)
        rep = summarize(per, "report")
        rep.to_csv(TAB / "pano_refine_report.csv", index=False)
        print(rep.round(3).to_string())
        rep_ids = list(df[df["set"] == "report"]["id"])
        ids = [rep_ids[k] for k in sorted({int(x) for x in np.linspace(0, len(rep_ids) - 1, min(4, len(rep_ids)))})]
        figures(per, rep_cfgs if len(rep_cfgs) > 2 else rep_cfgs, ids)


if __name__ == "__main__":
    main()
