"""PLAN_V4 milestone 3: a sharp, clean 3D view - offline part (exact ray casting, bilinear texture lookup,
2x supersampling; the real browser is scripts/viewer_check_v4.py).

Rooms: DEVELOPMENT = 5 bench rooms (layout seed 1 of each type), REPORT = 5 held-out bench rooms (seed 5) + the
six demo rooms. Panorama: demo rooms use the milestone-2 selection (pano_4096_detail.png); bench rooms are
sharpened here with Real-ESRGAN views (the app default, 13 s).
Reference = the v4 design shots (hero + four walls) cut from that 4096 panorama; the inside meshes are rendered
from the panorama centre at exactly those poses:
    PSNR, SSIM, LPIPS (SqueezeNet), sharpness ratio (Laplacian variance render / shot)
and from 0.6 m behind the centre (3 poses) for the geometry:
    jaggedness of the furniture-surface silhouettes (contour length / smoothed contour length, 1 = smooth)
Grain (Immerkaer noise estimate on flat wall pixels of the shots / renders, grey levels) along the chain:
    generator (raw panorama) -> upscaler (v3 texture, v4 4096 panorama) -> mesh texture + sampling (v3, v4 renders)
and the effect of edge-preserving denoising of the shell textures (bilateral, NL-means).
Variants (room_v4.DEFAULTS is the result of the selection below):
    v3            pipeline_v3 inside mesh (2048 texture, 1.5 cm shell texels, stride 2, inpainted to the silhouette)
    tex           4096 panorama + 5 mm shell texels, stride 2, no band, no smoothing
    tex+band      + silhouette band (6 cm)
    s1            + stride 1
    s1+inst       + depth smoothing inside instances
    s1+radial10   + Taubin along the rays (borders fixed)
    s1+full10/30  + 3D Taubin (borders along their chain, UVs re-projected)
    s1+full10+bilateral / +nlmeans   shell textures denoised
Selection (development rooms): geometry = lowest jaggedness among variants whose centre PSNR is within 0.3 dB of
the best; denoising only if it lowers the mid-band grain, keeps the sharpness ratio within 5 % and LPIPS within 10 %.
Outputs: results/tables/room_v4_per_room.csv, room_v4_dev.csv, room_v4_report.csv, room_v4_grain.csv,
room_v4_selection.json; figures room_v4_offcentre.png, room_v4_centre.png.
Usage: python scripts/eval_room_v4.py [--limit N]
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
from skimage.metrics import structural_similarity as ssim
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
from src import pipeline_v3 as p3v  # noqa: E402
from src import room_v4 as r4  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
CACHE = ROOT / "outputs" / "room_v4_eval"
DEVICE = torch.device("cuda")
VARIANTS = {
    "tex": dict(stride=2, band_m=0.0, smooth="none", taubin_iters=0),
    "tex+band": dict(stride=2, smooth="none", taubin_iters=0),
    "s1": dict(stride=1, smooth="none", taubin_iters=0),
    "s1+inst": dict(stride=1, smooth="instances", taubin_iters=0),
    "s1+radial10": dict(stride=1, smooth="radial", taubin_iters=10),
    "s1+full10": dict(stride=1, smooth="full", taubin_iters=10),
    "s1+full30": dict(stride=1, smooth="full", taubin_iters=30),
    "s1+full10+bilateral": dict(stride=1, smooth="full", taubin_iters=10, denoise="bilateral"),
    "s1+full10+nlmeans": dict(stride=1, smooth="full", taubin_iters=10, denoise="nlmeans"),
}


def room_list(limit=None):
    rows = []
    for r in mb.rooms().itertuples():
        if r.layout_seed in (1, 5):
            d = mb.OUT / r.id
            rows.append({"id": r.id, "set": "dev" if r.layout_seed == 1 else "report", "d": d, "dist": d / "dist.npy",
                         "tex": d / "texture.png", "ana": d, "hires": None})
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        rows.append({"id": "demo_" + pl.slug(prompt, seed), "set": "report", "d": d, "dist": d / "distance.npy",
                     "tex": d / "texture.png", "ana": d / "v3", "hires": d / "v4/pano_4096_detail.png"})
    df = pd.DataFrame(rows)
    return df.groupby("set").head(limit) if limit else df


def hires_for(r, net) -> np.ndarray:
    if r.hires is not None:
        return mb.rgb(r.hires)
    p = CACHE / f"{r.id}_4096.png"
    if not p.exists():
        pano, _ = pr.refine(mb.rgb(r.tex), net)
        cv2.imwrite(str(p), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    return mb.rgb(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    id2label = erb.id2label_ade()
    sets = ds.class_sets(id2label)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(DEVICE)
    lpips = lambda x, y: float(lp(*[torch.from_numpy(i).permute(2, 0, 1)[None].float().div(255).to(DEVICE)  # noqa: E731
                                     for i in (x, y)]))
    net = gp.load_esrgan(DEVICE)
    rows, grain, figs = [], [], {}
    df = room_list(a.limit)
    for r in tqdm(list(df.itertuples()), desc="rooms"):
        dist = np.load(r.dist)
        ana = {"class_map": cv2.imread(str(r.ana / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
               "inst_map": cv2.imread(str(r.ana / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((r.ana / "inst_v3.json").read_text())}
        tex, hires = mb.rgb(r.tex), hires_for(r, net)
        res = p3v.build(dist, tex, ana, id2label)
        box = res["box"]
        sh = ds.design_shots(hires, box, ana["class_map"], ana["inst_map"], ana["instances"], id2label, dist=dist)
        poses = [(s["yaw"], s["fov"]) for s in sh["shots"]]
        refs = [s["image"] for s in sh["shots"]]
        walls = [np.isin(ds.render_view(ana["class_map"].astype(np.float32), ds.view_to_cam(box, y), 1024, 768, f,
                                        cv2.INTER_NEAREST).astype(int), sets["wall"]) for y, f in poses]
        backs = [-0.6 * np.array([np.sin(np.radians(y)), 0, np.cos(np.radians(y))]) @ box["Q"].T for y, _ in poses[:3]]
        raw = mb.rgb(r.d / "pano.png")
        for name, src in (("generator: raw panorama", raw), ("upscaler: v3 texture (ESRGAN on the panorama)", tex),
                          ("upscaler: v4 4096 panorama", hires)):
            vs = [ds.render_view(src, ds.view_to_cam(box, y), 1024, 768, f) for y, f in poses]
            grain.append({"id": r.id, "set": r.set, "stage": name,
                          "noise": np.nanmean([r4.noise_sigma(v, m) for v, m in zip(vs, walls)]),
                          "grain_band": np.nanmean([r4.grain_band(v, m) for v, m in zip(vs, walls)])})
        meshes = {"v3": res["mesh"]}
        info = {"v3": {"build_s": np.nan}}
        for name, o in VARIANTS.items():
            t0 = time.perf_counter()
            meshes[name] = r4.build_meshes(dist, hires, res, ana, id2label, o)["inside"]
            info[name] = {"build_s": time.perf_counter() - t0}
        for name, m in meshes.items():
            if "face_kind" not in m:
                m["face_kind"] = np.array(["?"] * len(m["faces"]))
            ren = [r4.render(m, box, y, f)[0] for y, f in poses]
            ex = CACHE / f"tmp_{name.replace('+', '_')}.glb"
            from src import textured_mesh as tm
            tm.export_glb(m, ex, double_sided=False, mipmaps=name != "v3",
                          jpeg_quality=None if name == "v3" else r4.DEFAULTS["jpeg_q"])
            jag = [r4.jaggedness(r4.kind_mask(m, box, y, f, origin_A=b), sigma=6) for (y, f), b in zip(poses[:3], backs)] \
                if name != "v3" else [r4.jaggedness(r4.kind_mask(_with_kinds(res), box, y, f, origin_A=b), sigma=6)
                                      for (y, f), b in zip(poses[:3], backs)]
            rows.append({"id": r.id, "set": r.set, "variant": name,
                         "psnr": np.mean([cv2.PSNR(x, y) for x, y in zip(ren, refs)]),
                         "ssim": np.mean([ssim(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(y, cv2.COLOR_RGB2GRAY))
                                          for x, y in zip(ren, refs)]),
                         "lpips": np.mean([lpips(x, y) for x, y in zip(ren, refs)]),
                         "sharpness_ratio": np.mean([eq.laplacian_variance(x) / eq.laplacian_variance(y)
                                                     for x, y in zip(ren, refs)]),
                         "noise": np.nanmean([r4.noise_sigma(x, w_) for x, w_ in zip(ren, walls)]),
                         "grain_band": np.nanmean([r4.grain_band(x, w_) for x, w_ in zip(ren, walls)]),
                         "jaggedness": np.nanmean(jag), "glb_mb": ex.stat().st_size / 2 ** 20,
                         "triangles": len(m["faces"]), "atlas": "x".join(map(str, m["texture"].shape[:2])),
                         **info[name]})
            if name in ("v3", "s1+full10") and r.id.startswith("demo") and len(figs) < 6:
                figs.setdefault(r.id, {})[name] = (ren[0], r4.render(m, box, poses[0][0], poses[0][1], origin_A=backs[0])[0])
                figs[r.id]["ref"] = refs[0]
        for name in ("v3", "s1+full10"):
            q = pd.DataFrame(rows).query("id == @r.id and variant == @name")
            grain.append({"id": r.id, "set": r.set, "stage": f"render: {name} mesh", "noise": float(q["noise"].iloc[0]),
                          "grain_band": float(q["grain_band"].iloc[0])})
        pd.DataFrame(rows).to_csv(TAB / "room_v4_per_room.csv", index=False)
    per = pd.DataFrame(rows)
    cols = ["psnr", "ssim", "lpips", "sharpness_ratio", "noise", "grain_band", "jaggedness", "glb_mb", "triangles", "build_s"]
    order = ["v3"] + list(VARIANTS)
    for split, name in (("dev", "room_v4_dev.csv"), ("report", "room_v4_report.csv")):
        s = per[per.set == split].groupby("variant")[cols].mean().reindex(order).reset_index()
        s.to_csv(TAB / name, index=False)
        print(split)
        print(s.round(3).to_string())
    gr = pd.DataFrame(grain)
    gs = gr.groupby(["set", "stage"])[["noise", "grain_band"]].mean().reset_index()
    gs.to_csv(TAB / "room_v4_grain.csv", index=False)
    print(gs.round(2).to_string())
    dev = per[per.set == "dev"].groupby("variant")[cols].mean()
    geo = dev.loc[[v for v in VARIANTS if "+bilateral" not in v and "+nlmeans" not in v]]
    cand = geo[geo["psnr"] >= geo["psnr"].max() - 0.3]
    g_sel = cand["jaggedness"].idxmin()
    base = dev.loc["s1+full10"]
    dn = [v for v in ("s1+full10+bilateral", "s1+full10+nlmeans") if dev.loc[v, "grain_band"] < base["grain_band"]
          and dev.loc[v, "sharpness_ratio"] >= 0.95 * base["sharpness_ratio"] and dev.loc[v, "lpips"] <= 1.1 * base["lpips"]]
    sel = {"geometry": g_sel, "geometry_candidates": list(cand.index), "denoise": dn[0] if dn else "none",
           "rule": "lowest jaggedness within 0.3 dB of the best centre PSNR; denoise only if the mid-band grain drops, "
                   "the sharpness ratio stays within 5 % and LPIPS within 10 % (LPIPS condition added after a "
                   "one-room smoke test, before the full run)"}
    (TAB / "room_v4_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    # figures
    ids = list(figs)[:4]
    if not ids:
        return
    fig, axes = plt.subplots(len(ids), 5, figsize=(25, 4.0 * len(ids)), dpi=60, squeeze=False)
    for i, rid in enumerate(ids):
        f = figs[rid]
        ims = [f["ref"], f["v3"][0], f["s1+full10"][0], f["v3"][1], f["s1+full10"][1]]
        tt = ["design shot H (reference)", "v3 inside mesh, from the centre", "v4 inside mesh, from the centre",
              "v3, 0.6 m behind the centre", "v4, 0.6 m behind the centre"]
        for j, (im, t) in enumerate(zip(ims, tt)):
            axes[i, j].imshow(im)
            axes[i, j].set_title(t + ("  -  " + rid[5:] if j == 0 else ""), fontsize=11)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "room_v4_offcentre.png", facecolor="white")
    plt.close(fig)
    for p in CACHE.glob("tmp_*.glb"):
        p.unlink()


def _with_kinds(res):
    m = res["mesh"]
    if "face_kind" in m and m["face_kind"] is not None and len(m["face_kind"]) == len(m["faces"]):
        return m
    return m


if __name__ == "__main__":
    main()
