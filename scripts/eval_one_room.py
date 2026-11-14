"""PLAN_V3 milestone 5: panoramas that are actually one room.

Generator: the selected 360 model (SD 2.1-base panorama fine-tune, circular padding), as in the 360 mode.
Prompts: 5 room types x 2 styles; the first style ("scandinavian") is the DEV set, the second ("industrial")
the HELD-OUT set. 4 seeds per prompt (the candidates), each generated with
    A  the current template (generate_pano.ALT_TEMPLATE + PANO_NEGATIVE)
    B  the one-room template (ONE_ROOM_TEMPLATE: "single enclosed ..., four walls around the camera, view from
       the centre"; ONE_ROOM_NEGATIVE adds corridor, hallway, open plan, multiple rooms, archway, ...)
on the same seeds. Every panorama: DAP distance (+ wrap fix), SegFormer classes, box fit, boxiness score
(src/boxiness.py: wall fit, enclosed share, detected sides), CLIP score of 4 horizontal crops vs. the prompt.
Selection: best-of-k by boxiness for k = 1, 2, 4 (expected value over all k-subsets of the 4 candidates).
Decision rule (fixed before running): use template B if its mean dev score is higher and its dev CLIP is not
lower by more than 0.5.
Outputs: outputs/one_room/ (intermediate), results/tables/oneroom_*.csv, results/figures/oneroom_*.png
Usage: python scripts/eval_one_room.py
"""

import itertools
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
from src import analysis as an  # noqa: E402
from src import boxiness as bx  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import pano_depth as pdp  # noqa: E402
from src import room_box as rb  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "one_room"
ROOMS = ["bedroom", "living room", "kitchen", "dining room", "bathroom"]
STYLES = {"scandinavian": "dev", "industrial": "test"}
SEEDS = [501, 502, 503, 504]
TEMPLATES = {"A": (gp.ALT_TEMPLATE, gp.PANO_NEGATIVE), "B": (gp.ONE_ROOM_TEMPLATE, gp.ONE_ROOM_NEGATIVE)}


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def main():
    set_seed(42)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for style, split in STYLES.items():
        for room in ROOMS:
            prompt = f"a {style} style {room}"
            for t in TEMPLATES:
                for s in SEEDS:
                    items.append({"prompt": prompt, "room": room, "style": style, "split": split, "template": t,
                                  "seed": s, "path": OUT / f"{prompt.replace(' ', '_')}_{t}_s{s}"})
    todo = [it for it in items if not (it["path"].with_suffix(".png")).exists()]
    if todo:
        pipe = gp.load_alt_pipeline(device)
        for it in tqdm(todo, desc="panoramas", unit="pano"):
            tpl, neg = TEMPLATES[it["template"]]
            t0 = time.perf_counter()
            img = gp.generate_pano(pipe, it["prompt"], it["seed"], prompt=tpl.format(room=it["prompt"]), negative=neg)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            cv2.imwrite(str(it["path"].with_suffix(".png")), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            it["path"].with_suffix(".json").write_text(json.dumps({"generate_s": dt}))
        del pipe
        ev.release_memory()
    todo = [it for it in items if not Path(str(it["path"]) + "_dist.npy").exists()]
    if todo:
        dap, _ = pdp.load_dap(device)
        for it in tqdm(todo, desc="DAP", unit="pano"):
            t0 = time.perf_counter()
            np.save(str(it["path"]) + "_dist.npy", pdp.dap_depth_wrap(dap, rgb(it["path"].with_suffix(".png")), device))
            j = json.loads(it["path"].with_suffix(".json").read_text())
            j["dap_s"] = time.perf_counter() - t0
            it["path"].with_suffix(".json").write_text(json.dumps(j))
        del dap
        ev.release_memory()
    seg = an.load_segmenter(device)
    id2label = {int(k): v for k, v in seg[0].config.id2label.items()}
    clip = ev.load_clip(device)
    rows = []
    for it in tqdm(items, desc="score", unit="pano"):
        p = Path(str(it["path"]) + "_seg.png")
        pano = rgb(it["path"].with_suffix(".png"))
        j = json.loads(it["path"].with_suffix(".json").read_text())
        if not p.exists():
            t0 = time.perf_counter()
            cv2.imwrite(str(p), pdp.segment_pano(*seg, pano, device).astype(np.uint8))
            j["seg_s"] = time.perf_counter() - t0
            it["path"].with_suffix(".json").write_text(json.dumps(j))
        cm = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.int32)
        dist = np.load(str(it["path"]) + "_dist.npy")
        t0 = time.perf_counter()
        sc = bx.score(dist, cm, id2label)
        j["score_s"] = time.perf_counter() - t0
        crops = []
        for k, c in enumerate(eq.horizontal_views(pano)):
            cp = OUT / f"_crop{k}.png"
            cv2.imwrite(str(cp), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            crops.append(str(cp))
        cs = float(ev.clip_scores(*clip, crops, [it["prompt"]] * 4, device).mean())
        rows.append({k: v for k, v in it.items() if k != "path"} | sc | {"clip": cs} | j)
    del seg, clip
    ev.release_memory()
    df = pd.DataFrame(rows)

    per_t = df.groupby(["split", "template"]).agg(
        n=("score", "size"), score=("score", "mean"), score_q25=("score", lambda s: s.quantile(0.25)),
        score_q75=("score", lambda s: s.quantile(0.75)), wall_fit=("wall_fit", "mean"), enclosed=("enclosed", "mean"),
        beyond_share=("beyond_share", "mean"), sides=("sides", "mean"), area_m2=("area_m2", "median"),
        clip=("clip", "mean"), generate_s=("generate_s", "mean")).reset_index()
    dev = per_t[per_t["split"] == "dev"].set_index("template")
    use_b = bool(dev.loc["B", "score"] > dev.loc["A", "score"] and dev.loc["B", "clip"] >= dev.loc["A", "clip"] - 0.5)
    print(per_t.round(3).to_string(index=False))
    print("decision on dev: use template", "B (one room)" if use_b else "A (current)")

    sel = []
    for (split, t, prompt), g in df.groupby(["split", "template", "prompt"]):
        for k in (1, 2, 4):
            vals = []
            for sub in itertools.combinations(range(len(g)), k):
                best = g.iloc[list(sub)].sort_values("score", ascending=False).iloc[0]
                vals.append(best[["score", "enclosed", "wall_fit", "sides", "clip"]].astype(float))
            m = pd.DataFrame(vals).mean()
            sel.append({"split": split, "template": t, "prompt": prompt, "k": k, **m.to_dict()})
    sel = pd.DataFrame(sel).groupby(["split", "template", "k"])[["score", "enclosed", "wall_fit", "sides", "clip"]].mean().reset_index()
    t_gen = df["generate_s"].mean()
    t_rest = (df["dap_s"].mean() + df["seg_s"].mean() + df["score_s"].mean())
    sel["extra_time_s"] = (sel["k"] - 1) * (t_gen + t_rest)
    print(sel.round(3).to_string(index=False))
    print(f"per candidate: generate {t_gen:.1f} s + DAP/SegFormer/score {t_rest:.1f} s")

    T, F = ROOT / "results/tables", ROOT / "results/figures"
    df.to_csv(T / "oneroom_per_pano.csv", index=False, float_format="%.4f")
    per_t.to_csv(T / "oneroom_templates.csv", index=False, float_format="%.4f")
    sel.to_csv(T / "oneroom_selection.csv", index=False, float_format="%.4f")
    (T / "oneroom_decision.json").write_text(json.dumps({"use_one_room_template": use_b}))

    # figure: held-out prompts, both templates, 4 candidates each, best marked
    ex = df[(df["split"] == "test") & df["room"].isin(["living room", "kitchen", "dining room"])]
    fig, axes = plt.subplots(6, 4, figsize=(24, 15), dpi=70)
    for i, ((prompt, t), g) in enumerate(ex.groupby(["prompt", "template"], sort=False)):
        best = g["score"].idxmax()
        for j, (idx, row) in enumerate(g.iterrows()):
            ax = axes[i, j]
            ax.imshow(rgb(OUT / f"{prompt.replace(' ', '_')}_{t}_s{row['seed']}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            col = "#1baf7a" if idx == best else "#999999"
            for sp in ax.spines.values():
                sp.set_color(col)
                sp.set_linewidth(5 if idx == best else 1)
            ax.set_title(f"{'one-room' if t == 'B' else 'current'} template, seed {row['seed']}: score {row['score']:.2f} "
                         f"(enclosed {row['enclosed']:.2f}, sides {int(row['sides'] * 4)})", fontsize=11)
            if j == 0:
                ax.set_ylabel(prompt, fontsize=11)
    fig.suptitle("Held-out prompts: 4 candidates per template, the one kept by the boxiness score framed green", fontsize=14)
    fig.tight_layout()
    fig.savefig(F / "oneroom_candidates.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
