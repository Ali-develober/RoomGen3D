"""Old vs. new 3D stage on the demo prompts: Poisson mesh vs. textured grid mesh.

For every prompt (image cached by scripts/run_pipeline.py in outputs/pipeline/<slug>/image.png):
    Poisson (old)              point cloud -> normals -> Poisson -> vertex colours
    Grid only                  textured grid mesh cut at depth discontinuities (layer 0)
    Grid + inpainting + planes layers 0-2 (src/textured_mesh.py)
each rendered from the original camera and from views rotated by -25 / +25 degrees around the
scene. White = nothing there (a hole). Timing of each method is recorded.

Outputs (full run):
  results/figures/mesh_old_vs_new.png        Poisson vs. new, all prompts
  results/figures/mesh_layers.png            what each step adds, 2 prompts
  results/tables/mesh_old_vs_new.csv         time, triangles, planes, hole share per view
Smoke test (figure -> outputs/mesh_compare_smoke.png, nothing written to results/):
  python scripts/compare_mesh.py --limit 1
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import depth as dp  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src import reconstruct_3d as r3  # noqa: E402
from src import textured_mesh as tm  # noqa: E402
from src.utils import set_seed  # noqa: E402

OUT = ROOT / "outputs" / "pipeline"
YAWS = [-25, 0, 25]
METHODS = {"poisson": "Poisson mesh (old)", "grid": "Textured grid, cuts only",
           "full": "Textured grid + inpainting + planes (new)"}


def hole_share(img: np.ndarray) -> float:
    """Share of pure-white (empty) pixels in a render."""
    return float((img.min(axis=2) == 255).mean())


def run_one(img, seg, dm, device) -> dict:
    """Depth + the three meshes + renders for one image."""
    pre, _ = pl.preprocess(img)
    ana = pl.analyse(pre, *seg, device)
    depth = dp.refine_metric_depth(dp.predict_depth(*dm, pre, device), pre)
    pivot = float(np.median(depth))
    out = {"image": img, "depth": depth, "renders": {}, "seconds": {}, "stats": {}, "planes": []}

    t0 = time.perf_counter()
    rec = r3.reconstruct(depth, pre)
    out["seconds"]["poisson"] = time.perf_counter() - t0
    out["stats"]["poisson"] = {"triangles": rec["stats"]["triangles"]}
    samples = {"poisson": tm.vertex_color_samples(rec["mesh"])}

    for name, kw in (("grid", {"fill": False}), ("full", {})):
        t0 = time.perf_counter()
        mesh = tm.build_textured_mesh(depth, pre, ana["edges"], ana["masks"], **kw)
        out["seconds"][name] = time.perf_counter() - t0
        out["stats"][name] = mesh["stats"]
        samples[name] = tm.textured_samples(mesh)
        if name == "full":
            out["planes"], out["mesh"] = mesh["planes"], mesh
    for name, (pts, cols) in samples.items():
        out["renders"][name] = [tm.render_view(pts, cols, yaw, pivot_z=pivot) for yaw in YAWS]
    return out


def figure(results, prompts, methods, path, title):
    rows = [(r, m) for r in range(len(results)) for m in methods]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13.2, 3.25 * len(rows)), squeeze=False)
    heads = ["Generated image"] + [("Camera view" if y == 0 else f"Rotated {y:+d}°") for y in YAWS]
    for i, (r, m) in enumerate(rows):
        res = results[r]
        panels = [res["image"]] + res["renders"][m]
        for c, p in enumerate(panels):
            ax = axes[i, c]
            ax.imshow(p)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(heads[c], fontsize=10.5)
            if c > 0:
                ax.set_xlabel(f"empty: {100 * hole_share(p):.1f}%", fontsize=8.5)
        axes[i, 0].set_ylabel(f"{METHODS[m]}\n{prompts[r][0]}".replace(" style ", " style\n"), fontsize=8.5)
        axes[i, 0].set_xlabel(f"{res['seconds'][m]:.1f} s, {res['stats'][m]['triangles']:,} triangles", fontsize=8.5)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N prompts")
    ap.add_argument("--start", type=int, default=0, help="index of the first prompt")
    args = ap.parse_args()
    set_seed(42)
    device = torch.device("cuda")
    prompts = pl.DEMO_PROMPTS[args.start:][: args.limit] if args.limit else pl.DEMO_PROMPTS
    images = [pp.load_rgb(OUT / pl.slug(p, s) / "image.png") for p, s in prompts]

    seg = an.load_segmenter(device)
    dm = dp.load_depth_model(pl.DEPTH_MODEL_3D, device)     # both are small (0.3 GB each): verified to fit together
    results = [run_one(im, seg, dm, device) for im in images]
    del seg, dm
    ev.release_memory()

    rows = []
    for (prompt, seed), res in zip(prompts, results):
        for m in METHODS:
            holes = [hole_share(x) for x in res["renders"][m]]
            rows.append({"prompt": prompt, "seed": seed, "method": m, "seconds": res["seconds"][m],
                         "triangles": res["stats"][m]["triangles"],
                         **{f"empty_pct_{y:+d}": 100 * h for y, h in zip(YAWS, holes)},
                         "planes_accepted": res["stats"][m].get("planes_accepted", np.nan)})
        for p in res["planes"]:
            print(f"  {prompt}: {p['kind']:8s} inliers {100 * p['inlier_share']:.0f}%  rms {100 * p['rms']:.1f} cm  "
                  f"behind {100 * p['behind_share']:.0f}%  -> {'ok' if p['accepted'] else 'SKIPPED: ' + p['reason']}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(2).to_string(index=False))
    title = ("3D stage, old vs. new, rendered from the original camera and rotated ±25° around the scene "
             "(white = empty)")
    if args.limit:
        path = ROOT / "outputs" / "mesh_compare_smoke.png"
        figure(results, prompts, list(METHODS), path, title)
        tm.export_glb(results[0]["mesh"], ROOT / "outputs" / "mesh_compare_smoke.glb")
        print("smoke figure:", path)
        return
    mean = df.groupby("method", sort=False).mean(numeric_only=True).reset_index().drop(columns="seed")
    mean.insert(0, "prompt", "MEAN")
    pd.concat([df, mean], ignore_index=True).to_csv(ROOT / "results/tables/mesh_old_vs_new.csv", index=False,
                                                    float_format="%.3f")
    figure(results, prompts, ["poisson", "full"], ROOT / "results/figures/mesh_old_vs_new.png", title)
    figure(results[1:4:2], prompts[1:4:2], list(METHODS), ROOT / "results/figures/mesh_layers.png",
           "What each step of the new 3D stage adds")
    print("\n", mean.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
