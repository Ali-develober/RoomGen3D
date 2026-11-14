"""Polish round, step 2: sharper design shots (perspective shots only).

Rooms: the six demo rooms (both 4096 panoramas exist: Real-ESRGAN views = the app's fast path, and the img2img
detail pass) + 10 held-out bench rooms (layout seeds 5-6; their ESRGAN-views panorama is recomputed into the
scratch cache; no detail-pass panorama there). Shots: hero + four wall views (demo), hero + wall A (bench), 1536 x 1152.
Methods:
    fast          the shot cut directly at 1536 from the ESRGAN-views panorama (current fast path)
    detail_pass   the shot cut at 1536 from the img2img + MMIS LoRA panorama (current detail option; demo only)
    esrgan_shot   Real-ESRGAN x2 on the 1024 shot (from the fast panorama), area-resized to 1536
    tile_0.35 / tile_0.5   SD 1.5 + ControlNet tile on the 1024 shot (src/shot_enhance.tile_upscale)
Measures: sharpness (Laplacian variance at 1536), BRISQUE, edge alignment = structural edge F1 against the fast
shot (Canny after a 1.5 px blur at 1024 x 768, 2 px tolerance: geometry must not move), seconds per shot.
"Clearly better" (fixed before running): best sharpness AND best BRISQUE among the methods with edge F1 >= 0.85,
and BRISQUE at least 3 points below the fast path.
Outputs: results/tables/shot_enhance_per_shot.csv, shot_enhance.csv, shot_enhance_selection.json;
figure results/figures/shot_enhance_crops.png
Usage: python scripts/eval_shot_enhance.py [--cache DIR]
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
from src import design_shots as ds  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import shot_enhance as se  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEVICE = torch.device("cuda")
W, H = se.OUT_W, se.OUT_H


def rooms(cache: Path):
    out = []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        out.append({"id": "demo_" + pl.slug(prompt, seed), "set": "demo", "prompt": prompt, "seed": seed, "ana": d / "v3",
                    "dist": d / "distance.npy", "fast": d / "v4/pano_4096_esrgan.png", "detail": d / "v4/pano_4096_detail.png",
                    "tex": d / "texture.png"})
    for r in mb.rooms().itertuples():
        if r.layout_seed in (5, 6):
            d = mb.OUT / r.id
            out.append({"id": r.id, "set": "bench", "prompt": r.prompt, "seed": int(r.gen_seed), "ana": d,
                        "dist": d / "dist.npy", "fast": cache / f"{r.id}_4096.jpg", "detail": None, "tex": d / "texture.png"})
    return out


def shots_for(r, id2label, pano):
    dist = np.load(r["dist"])
    cm = cv2.imread(str(r["ana"] / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    inst = cv2.imread(str(r["ana"] / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
    insts = json.loads((r["ana"] / "inst_v3.json").read_text())
    box = rb.fit_box(dist, cm, id2label)
    res = ds.design_shots(pano, box, cm, inst, insts, id2label, dist=dist, room=ds.room_type_of(r["prompt"]), n_hero=1)
    return box, [(s["letter"], s["yaw"], s["fov"]) for s in res["shots"]]


def main():
    import piq
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / "AppData/Local/Temp/roomgen_shot_cache"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    id2label = erb.id2label_ade()
    R = rooms(cache)
    net = gp.load_esrgan(DEVICE)
    for r in tqdm([r for r in R if not r["fast"].exists()], desc="bench panoramas (ESRGAN views)"):
        p, _ = pr.refine(mb.rgb(r["tex"]), net)
        cv2.imwrite(str(r["fast"]), cv2.cvtColor(p, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    # shots + non-SD methods
    jobs = []
    for r in tqdm(R, desc="shots"):
        fast = mb.rgb(r["fast"])
        box, poses = shots_for(r, id2label, fast)
        det = mb.rgb(r["detail"]) if r["detail"] else None
        for letter, yaw, fov in (poses if r["set"] == "demo" else [p for p in poses if p[0] in ("H", "A")]):
            M = ds.view_to_cam(box, yaw)
            s1024 = ds.render_view(fast, M, 1024, 768, fov)
            outs = {"fast": (ds.render_view(fast, M, W, H, fov), 0.0)}
            if det is not None:
                outs["detail_pass"] = (ds.render_view(det, M, W, H, fov), 0.0)
            outs["esrgan_shot"] = se.esrgan_shot(net, s1024)
            jobs.append({"r": r, "letter": letter, "s1024": s1024, "outs": outs})
    del net
    torch.cuda.empty_cache()
    pipe = se.load_tile_pipeline(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    for k, j in enumerate(tqdm(jobs, desc="tile ControlNet")):
        for s in (0.35, 0.5):
            j["outs"][f"tile_{s}"] = se.tile_upscale(pipe, j["s1024"], j["r"]["prompt"], s, seed=1000 + k)
    vram = torch.cuda.max_memory_allocated() / 2 ** 30
    del pipe
    torch.cuda.empty_cache()
    rows, ex = [], []
    for j in tqdm(jobs, desc="metrics"):
        ref = j["outs"]["fast"][0]
        for m, (im, sec) in j["outs"].items():
            x = torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255
            rows.append({"id": j["r"]["id"], "set": j["r"]["set"], "shot": j["letter"], "method": m,
                         "sharpness": eq.laplacian_variance(im), "brisque": float(piq.brisque(x, data_range=1.0)),
                         "edge_f1": pr.edge_f1(ref, im, size=(1024, 768)), "seconds": sec})
        if j["r"]["set"] == "demo" and j["letter"] == "H" and len(ex) < 3:
            ex.append(j)
    per = pd.DataFrame(rows)
    per.to_csv(TAB / "shot_enhance_per_shot.csv", index=False)
    summ = per.groupby(["set", "method"])[["sharpness", "brisque", "edge_f1", "seconds"]].mean().reset_index()
    summ["peak_vram_gb_tile"] = vram
    summ.to_csv(TAB / "shot_enhance.csv", index=False)
    print(summ.round(3).to_string())
    allm = per.groupby("method")[["sharpness", "brisque", "edge_f1", "seconds"]].mean()
    ok = allm[allm["edge_f1"] >= 0.85]
    best_s, best_b = ok["sharpness"].idxmax(), ok["brisque"].idxmin()
    clear = best_s == best_b and ok.loc[best_b, "brisque"] <= allm.loc["fast", "brisque"] - 3 and best_b != "fast"
    sel = {"clearly_better": bool(clear), "method": best_b if clear else "fast", "best_sharpness": best_s,
           "best_brisque": best_b, "passed_edge_guard": list(ok.index),
           "rule": "best sharpness AND best BRISQUE among methods with edge F1 >= 0.85, BRISQUE >= 3 below fast"}
    (TAB / "shot_enhance_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    meths = list(ex[0]["outs"])
    fig, axes = plt.subplots(len(ex), len(meths), figsize=(4.2 * len(meths), 3.6 * len(ex)), dpi=80, squeeze=False)
    for i, j in enumerate(ex):
        for c, m in enumerate(meths):
            im = j["outs"][m][0]
            crop = im[H // 2 - 180:H // 2 + 180, W // 2 - 240:W // 2 + 240]
            axes[i, c].imshow(crop)
            axes[i, c].set_title(m + (f" - {j['r']['id'][5:]}" if c == 0 else ""), fontsize=10)
            axes[i, c].axis("off")
    fig.suptitle("Centre crops (480 x 360 px at 1:1) of the 1536 x 1152 hero shot", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "shot_enhance_crops.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
