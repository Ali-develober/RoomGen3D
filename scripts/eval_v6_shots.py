"""PLAN_V6 milestone 3: design shots from cameras placed in the scene — sweep and selection.

Per room (panoramas of the selected milestone-2 config in outputs/v6_pano/<cfg>/): cameras (src/cameras_v6.py),
reprojection of the panorama through the proxy (z-buffer), LaMa hole fill, then per config
    lora1_s{0.35,0.45,0.55}   SD 1.5 + MMIS LoRA + depth ControlNet img2img (exact proxy depth)
    lora0_s{0.35,0.45,0.55}   the same without our LoRA
    init                      the filled reprojection itself (no re-rendering), reference
Metrics per shot (768 x 576, before Real-ESRGAN):
    consistency  colour: mean |ΔLab| (CIE76) of the low-passed (σ = 3 px) shot vs the reprojected panorama on
                 co-visible pixels; LPIPS (SqueezeNet, torchmetrics) between the shot and the shot with its
                 co-visible pixels replaced by the reprojection (so only co-visible pixels count)
    geometry     depth AbsRel of Depth Anything (relative, scale/shift aligned) vs the proxy depth; edge recall:
                 share of proxy instance boundaries (furniture / walls / openings) with a Canny edge within 3 px
    quality      BRISQUE, KID vs the 500 MMIS hold-out photos (pooled per config and set), CLIP score (prompt)
Selection rule (fixed before the run): on dev, configs must pass the consistency / geometry constraints
    LPIPS <= 0.30, ΔLab <= 12, depth AbsRel <= 0.15, edge recall >= 0.50 (mean over shots);
    among those, best mean rank of BRISQUE, KID, CLIP; if none passes, the config with the lowest LPIPS.
Outputs: outputs/v6_shots/<cfg>/<room>_<shot>.jpg, results/tables/v6_shots_per_shot.csv, v6_shots_summary.csv,
         v6_shots_decision.json (read by src/pipeline_v6.py)
Usage: python scripts/eval_v6_shots.py --pano cn_small_s0.7_e1.0 --set dev --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import cameras_v6 as CV  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import layout_v6 as LV  # noqa: E402
from src import memory as mem  # noqa: E402
from src import scene_v6 as S  # noqa: E402
from src import shot_render as SR  # noqa: E402
from src import shots_v6 as SH  # noqa: E402
import v5_common as C  # noqa: E402
from eval_v6_pano import an_align, room_df  # noqa: E402

OUT = ROOT / "outputs" / "v6_shots"
PANO = ROOT / "outputs" / "v6_pano"
DEVICE = "cuda"
CONFIGS = ["init", "lora1_s0.35", "lora1_s0.45", "lora1_s0.55", "lora0_s0.35", "lora0_s0.45", "lora0_s0.55"]
RULE = {"lpips_max": 0.30, "dlab_max": 12.0, "absrel_max": 0.15, "edge_min": 0.50}


def prep(df, pano_cfg):
    """Cameras, reprojections and LaMa-filled inits per room (cached as .npz)."""
    d = OUT / "_prep"
    d.mkdir(parents=True, exist_ok=True)
    from src import dollhouse_v5 as dh
    for r in df.itertuples():
        f = d / f"{r.id}.npz"
        src = PANO / pano_cfg / f"{r.id}.jpg"
        if f.exists() or not src.exists():
            continue
        pano = C.rgb(src)
        sc = S.build(LV.generate(r.prompt, r.seed))
        rd = S.Renderer(sc)
        lay = sc["layout"]
        cams = CV.choose_cameras(rd, lay)
        arrs, meta = {}, {}
        for n, cam in cams.items():
            if cam is None:
                continue
            raw, vis, rr = SH.reproject(rd, pano, lay["camera"], cam)
            init = SH.fill_holes(raw, vis, DEVICE)
            arrs[f"{n}_raw"], arrs[f"{n}_vis"], arrs[f"{n}_init"] = raw, vis, init
            arrs[f"{n}_z"], arrs[f"{n}_inst"] = rr["z"].astype(np.float32), rr["inst"].astype(np.int16)
            meta[n] = cam
        np.savez_compressed(f, **arrs)
        (d / f"{r.id}.json").write_text(json.dumps({"cams": meta, "prompt": SH.shot_prompt(lay, "H")}, default=float))
        print("prep", r.id, flush=True)
    dh.free_lama()
    mem.free()


def load_prep(rid):
    z = np.load(OUT / "_prep" / f"{rid}.npz")
    meta = json.loads((OUT / "_prep" / f"{rid}.json").read_text())
    return z, meta


def render(df, configs):
    for lora in (1, 0):
        cs = [c for c in configs if c.startswith(f"lora{lora}")]
        todo = [(c, r) for c in cs for r in df.itertuples()
                if (OUT / "_prep" / f"{r.id}.npz").exists() and not (OUT / c / f"{r.id}_done").exists()]
        if not todo:
            continue
        pipe = SR.load_cn_img2img(DEVICE, lora=bool(lora))
        for c, r in todo:
            s = float(c.split("_s")[1])
            z, meta = load_prep(r.id)
            (OUT / c).mkdir(parents=True, exist_ok=True)
            for k, n in enumerate(meta["cams"]):
                img = SH.rerender(pipe, z[f"{n}_init"], z[f"{n}_z"], meta["prompt"], s, r.seed + k)
                Image.fromarray(img).save(OUT / c / f"{r.id}_{n}.jpg", quality=95)
            (OUT / c / f"{r.id}_done").write_text("ok")
            print(c, r.id, flush=True)
        mem.free(pipe)
        del pipe
        mem.free()
    (OUT / "init").mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        if not (OUT / "_prep" / f"{r.id}.npz").exists():
            continue
        z, meta = load_prep(r.id)
        for n in meta["cams"]:
            p = OUT / "init" / f"{r.id}_{n}.jpg"
            if not p.exists():
                Image.fromarray(z[f"{n}_init"]).save(p, quality=95)


def inst_edges(inst):
    e = np.zeros(inst.shape, bool)
    e[:, 1:] |= inst[:, 1:] != inst[:, :-1]
    e[1:, :] |= inst[1:, :] != inst[:-1, :]
    return e


def metrics(df, configs):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(DEVICE)
    dm, dproc = dp.load_depth_model("depth_anything", DEVICE)
    rows = []
    out = C.TAB / "v6_shots_per_shot.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["config", "id", "shot"])
    done = set(zip(have.config, have.id, have.shot))
    for r in df.itertuples():
        if not (OUT / "_prep" / f"{r.id}.npz").exists():
            continue
        z, meta = load_prep(r.id)
        for c in configs:
            for n in meta["cams"]:
                p = OUT / c / f"{r.id}_{n}.jpg"
                if (c, r.id, n) in done or not p.exists():
                    continue
                img = C.rgb(p)
                raw, vis = z[f"{n}_raw"], z[f"{n}_vis"].astype(bool)
                vis_e = cv2.erode(vis.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                lab_a = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 3), cv2.COLOR_RGB2LAB).astype(np.float32)
                lab_b = cv2.cvtColor(cv2.GaussianBlur(raw, (0, 0), 3), cv2.COLOR_RGB2LAB).astype(np.float32)
                scale = np.array([100 / 255, 1, 1], np.float32)
                dlab = float(np.linalg.norm((lab_a - lab_b) * scale, axis=-1)[vis_e].mean()) if vis_e.any() else np.nan
                comp = np.where(vis_e[..., None], raw, img)
                ta = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
                tb = torch.from_numpy(comp).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
                lpv = float(lp(ta, tb))
                zz = z[f"{n}_z"]
                ok = np.isfinite(zz) & (zz > 0.1)
                disp = dp.predict_disparity(dm, dproc, img, DEVICE)
                pred = an_align(disp, zz, ok)
                absrel = float((np.abs(pred[ok] - zz[ok]) / zz[ok]).mean())
                g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                can = cv2.dilate((cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 40, 110) > 0).astype(np.uint8),
                                 np.ones((7, 7), np.uint8)) > 0
                ie = inst_edges(z[f"{n}_inst"])
                rows.append({"config": c, "id": r.id, "set": r.set, "shot": n, "dlab": dlab, "lpips": lpv,
                             "depth_absrel": absrel, "edge_recall": float(can[ie].mean()),
                             "brisque": C.brisque([img]), "sharpness": C.sharpness([img]),
                             "valid": bool(meta["cams"][n]["valid"]), "covis": float(vis.mean())})
        print("metrics", r.id, flush=True)
    mem.free(lp, dm, dproc)
    new = pd.DataFrame(rows)
    pd.concat([have, new], ignore_index=True).to_csv(out, index=False)


def clip_kid(df, configs) -> pd.DataFrame:
    per = pd.read_csv(C.TAB / "v6_shots_per_shot.csv")
    model, proc = ev.load_clip(DEVICE)
    prompts = dict(zip(df.id, df.prompt))
    clips = []
    for c in configs:
        sub = per[(per.config == c) & per.id.isin(df.id)]
        paths = [OUT / c / f"{a}_{b}.jpg" for a, b in zip(sub.id, sub.shot)]
        sc = ev.clip_scores(model, proc, paths, [prompts[a] for a in sub.id], DEVICE)
        clips.append(pd.DataFrame({"config": c, "id": sub.id.values, "shot": sub.shot.values, "clip": sc}))
    mem.free(model, proc)
    per = per.drop(columns=["clip"], errors="ignore").merge(pd.concat(clips), on=["config", "id", "shot"], how="left")
    per.to_csv(C.TAB / "v6_shots_per_shot.csv", index=False)
    hold = ev.load_holdout(ROOT / "data/mmis/fid_holdout.csv", ROOT / "data/mmis")
    sets = {}
    for c in configs:
        for split in sorted(df.set.unique()):
            sub = per[(per.config == c) & (per.set == split)]
            if len(sub) >= 16:
                sets[f"{c}|{split}"] = [str(OUT / c / f"{a}_{b}.jpg") for a, b in zip(sub.id, sub.shot)]
    fk = ev.fid_kid(list(hold["real_path"]), sets, DEVICE)
    return pd.DataFrame([{"config": k.split("|")[0], "set": k.split("|")[1], "kid_x1e3": v["kid_mean"] * 1e3}
                         for k, v in fk.items()])


def summarize(df, configs, kid: pd.DataFrame | None):
    per = pd.read_csv(C.TAB / "v6_shots_per_shot.csv")
    per = per[per.id.isin(df.id) & per.config.isin(configs)]
    cols = ["dlab", "lpips", "depth_absrel", "edge_recall", "brisque", "sharpness", "clip", "covis"]
    summ = per.groupby(["config", "set"])[[c for c in cols if c in per]].mean().reset_index()
    if kid is not None:
        summ = summ.merge(kid, on=["config", "set"], how="left")
    summ.to_csv(C.TAB / "v6_shots_summary.csv", index=False)
    dev = summ[(summ.set == "dev") & (summ.config != "init")].copy()
    if len(dev):
        ok = dev[(dev.lpips <= RULE["lpips_max"]) & (dev.dlab <= RULE["dlab_max"]) &
                 (dev.depth_absrel <= RULE["absrel_max"]) & (dev.edge_recall >= RULE["edge_min"])]
        if len(ok):
            rk = pd.DataFrame({"brisque": ok.brisque.rank(), "clip": ok.clip.rank(ascending=False)})
            if "kid_x1e3" in ok:
                rk["kid"] = ok.kid_x1e3.rank()
            ok = ok.assign(mean_rank=rk.mean(1).values)
            sel = ok.sort_values("mean_rank").iloc[0]
            why = "passes the consistency constraints; best mean rank of BRISQUE, KID, CLIP"
        else:
            sel = dev.sort_values("lpips").iloc[0]
            why = "no config passes the constraints: lowest LPIPS"
        c = sel.config
        dec = {"selected": c, "why": why, "rule": RULE,
               "selected_params": {"lora": c.startswith("lora1"), "strength": float(c.split("_s")[1])},
               "dev": dev.round(4).to_dict("records")}
        C.write_json(C.TAB / "v6_shots_decision.json", dec)
        print("selected", c, why)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--set", default="dev")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    df = room_df(a.set)
    cfgs = a.configs.split(",")
    st = a.stage.split(",")
    if "prep" in st or "all" in st:
        prep(df, a.pano)
    if "render" in st or "all" in st:
        render(df, cfgs)
    if "metrics" in st or "all" in st:
        metrics(df, cfgs)
    kid = None
    if "kid" in st or "all" in st:
        kid = clip_kid(df, cfgs)
        kid.to_csv(C.TAB / "v6_shots_kid.csv", index=False)
    elif (C.TAB / "v6_shots_kid.csv").exists():
        kid = pd.read_csv(C.TAB / "v6_shots_kid.csv")
    if "report" in st or "all" in st or "kid" in st:
        summarize(df, cfgs, kid)


if __name__ == "__main__":
    main()
