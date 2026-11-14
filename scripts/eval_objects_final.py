"""Polish round, step 3: one last attempt at real furniture - only the single LARGEST piece per room.

Input: the instance-centred crop (objects3d.instance_view, 512 px from the 4096 panorama) ENHANCED with the step-2
method (results/tables/shot_enhance_selection.json; tile ControlNet or Real-ESRGAN, to 1024 px), a clean cut-out
from the instance mask (upsampled, cleaned, GrabCut-refined at 1024 px, holes filled, occluded hull parts inpainted,
feathered), then TripoSR and the same render-and-compare placement as milestone 4. Compared per instance with the
v3 block and with the milestone-4 object of the same instance (results/tables/objects_per_instance.csv).
Rooms: the 40 bench rooms (dev / held-out) + the six demo rooms.
Decision (fixed before running): if the enhanced object's silhouette IoU is not above the block's on the held-out
rooms (mean, and in at least half of the instances), this is recorded as the final negative result.
Outputs: results/tables/objects_final_per_instance.csv, objects_final.csv; figure objects_final_inputs.png
Usage: python scripts/eval_objects_final.py [--cache DIR]
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
from src import furniture as fu  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import objects3d as o3  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import shot_enhance as se  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEVICE = torch.device("cuda")


def room_list(cache: Path):
    out = []
    for r in mb.rooms().itertuples():
        d = mb.OUT / r.id
        out.append({"id": r.id, "set": "dev" if r.split == "dev" else "test", "prompt": r.prompt, "ana": d,
                    "dist": d / "dist.npy", "tex": d / "texture.png", "hires": cache / f"{r.id}_4096.jpg"})
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        out.append({"id": "demo_" + pl.slug(prompt, seed), "set": "demo", "prompt": prompt, "ana": d / "v3",
                    "dist": d / "distance.npy", "tex": d / "texture.png", "hires": d / "v4/pano_4096_detail.png"})
    return out


def enhanced_input(v: dict, enhance, prompt: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced 1024 crop + clean cut-out -> TripoSR input (512)."""
    big = enhance(v["crop"], prompt, seed)
    S = big.shape[0]
    up = lambda m: cv2.resize(m.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    v2 = {**v, "crop": big, "mask": up(v["mask"]), "occ": up(v["occ"]), "hull": up(v["hull"])}
    img, _ = o3.prepare_input(v2, size=o3.IN_SIZE)
    return img, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    id2label = erb.id2label_ade()
    method = json.loads((TAB / "shot_enhance_selection.json").read_text())
    use_tile = method["best_brisque"].startswith("tile")             # the step-2 enhancer with the best BRISQUE
    R = room_list(cache)
    net = gp.load_esrgan(DEVICE)
    for r in tqdm([r for r in R if not r["hires"].exists()], desc="4096 panoramas"):
        p, _ = pr.refine(mb.rgb(r["tex"]), net)
        cv2.imwrite(str(r["hires"]), cv2.cvtColor(p, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    # 1) largest piece per room + its enhanced input (one SD model on the GPU at a time)
    jobs = []
    if use_tile:
        tpipe = se.load_tile_pipeline(DEVICE)
        strength = float(method["best_brisque"].split("_")[1])
        enhance = lambda img, prompt, seed: se.tile_upscale(tpipe, img, prompt, strength, seed, 1024, 1024)[0]  # noqa: E731
    else:
        enhance = lambda img, prompt, seed: se.esrgan_shot(net, img, 1024, 1024)[0]  # noqa: E731
    for k, r in enumerate(tqdm(R, desc="largest piece: enhanced inputs")):
        dist = np.load(r["dist"])
        ana = {"class_map": cv2.imread(str(r["ana"] / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
               "inst_map": cv2.imread(str(r["ana"] / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((r["ana"] / "inst_v3.json").read_text())}
        res = p3v.build(dist, mb.rgb(r["tex"]), ana, id2label)
        sol = [s for s in res["solids"] if int(s.get("instance", -1)) >= 0]
        if not sol:
            continue
        s = max(sol, key=lambda s: s["e"][0] * s["e"][1])
        inst_map, _ = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)
        iid = int(s["instance"])
        if not (inst_map == iid).any():
            continue
        hires = mb.rgb(r["hires"])
        v = o3.instance_view(hires, inst_map, dist, iid)
        t0 = time.perf_counter()
        img, big = enhanced_input(v, enhance, r["prompt"], 500 + k)
        base, _ = o3.prepare_input(v)
        jobs.append({"r": r, "solid": s, "box": res["box"], "v": v, "img": img, "base": base, "dist": dist,
                     "inst_map": inst_map, "enh_s": time.perf_counter() - t0, "iid": iid})
    del net
    if use_tile:
        del tpipe
    torch.cuda.empty_cache()
    # 2) TripoSR + placement for the enhanced input (and the plain milestone-4 input, for a paired comparison)
    model = o3.load_triposr(DEVICE)
    rows = []
    for j in tqdm(jobs, desc="TripoSR"):
        out = {}
        for name, img in (("enhanced", j["img"]), ("m4_input", j["base"])):
            t0 = time.perf_counter()
            obj = o3.simplify(o3.generate(model, img, DEVICE))
            if len(obj["faces"]) < 20:
                out[name] = (0.0, np.nan, time.perf_counter() - t0)
                continue
            o = o3.orient(obj, j["solid"], j["box"], j["v"], j["inst_map"], j["dist"], j["iid"])
            out[name] = (o["iou"], o["iou_block"], time.perf_counter() - t0)
        rows.append({"id": j["r"]["id"], "set": j["r"]["set"], "label": j["solid"]["label"], "instance": j["iid"],
                     "iou_block": out["m4_input"][1] if not np.isnan(out["m4_input"][1]) else out["enhanced"][1],
                     "iou_m4_input": out["m4_input"][0], "iou_enhanced": out["enhanced"][0],
                     "enhance_s": j["enh_s"], "triposr_s": out["enhanced"][2]})
    del model
    torch.cuda.empty_cache()
    per = pd.DataFrame(rows)
    per.to_csv(TAB / "objects_final_per_instance.csv", index=False)
    per["enh_beats_block"] = per["iou_enhanced"] > per["iou_block"]
    per["enh_beats_m4"] = per["iou_enhanced"] > per["iou_m4_input"]
    summ = per.groupby("set").agg(n=("id", "size"), iou_block=("iou_block", "mean"), iou_m4_input=("iou_m4_input", "mean"),
                                  iou_enhanced=("iou_enhanced", "mean"), share_beats_block=("enh_beats_block", "mean"),
                                  share_beats_m4=("enh_beats_m4", "mean"), enhance_s=("enhance_s", "mean"),
                                  triposr_s=("triposr_s", "mean")).reset_index()
    summ["enhancer"] = method["best_brisque"] if use_tile else "esrgan_shot"
    summ.to_csv(TAB / "objects_final.csv", index=False)
    print(summ.round(3).to_string())
    t = summ.set_index("set").loc["test"]
    verdict = "positive" if t["iou_enhanced"] > t["iou_block"] and t["share_beats_block"] >= 0.5 else "final negative result"
    (TAB / "objects_final_verdict.json").write_text(json.dumps({"verdict": verdict, "held_out": t.to_dict()}, indent=1, default=float))
    print(verdict)
    ex = [j for j in jobs if j["r"]["set"] == "demo"][:6]
    fig, axes = plt.subplots(2, len(ex), figsize=(3.6 * len(ex), 7.4), dpi=70, squeeze=False)
    for c, j in enumerate(ex):
        axes[0, c].imshow(j["base"]); axes[0, c].set_title(f"milestone-4 input: {j['solid']['label']}", fontsize=10)
        axes[1, c].imshow(j["img"]); axes[1, c].set_title("enhanced + clean cut-out", fontsize=10)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "objects_final_inputs.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
