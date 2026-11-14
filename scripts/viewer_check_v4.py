"""PLAN_V4 milestone 3: the inside viewer in a real browser (headed Chromium via Playwright, <model-viewer> 3.5 as
in the app), v3 against v4, for the six demo rooms.

Meshes: v3 = outputs/room360/<room>/v3/room_v3.glb (as shipped by v3); v4 = room_v4.build_meshes with the
selection of scripts/eval_room_v4.py (results/tables/room_v4_selection.json), exported to
outputs/room360/<room>/v4/room_v4.glb + dollhouse_v4.glb (mipmapped, single-sided).
Screenshots at the design-shot poses (camera at the panorama centre, 1 cm orbit radius, level, the shot's
vertical FOV, 1024 x 768 element) compared with the design shots (outputs/room360/<room>/v4 shots rebuilt from
the 4096 panorama): PSNR, SSIM, LPIPS (SqueezeNet), sharpness ratio, colour error (mean CIEDE2000), share of
the view showing no surface (magenta background).
Colour: tone mapping "neutral" (the v3 app setting) vs "none" on the v4 mesh.
Sampling: mipmapped vs not, at a pose looking 35 deg down at the floor, against an offline 4x-supersampled
render of the same mesh (PSNR; aliasing = sparkle when the camera moves).
Opening view: the app's inside viewer (app.inside_html) opens level, facing the hero; screenshot vs hero shot.
Outputs: results/tables/viewer_v4.csv, viewer_v4_summary.csv; results/figures/viewer_v4_inside.png;
outputs/viewer_check/v4_*.png
Usage: python scripts/viewer_check_v4.py [--headless]
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
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import design_shots as ds  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import pipeline_v4 as p4v  # noqa: E402
from src import room_v4 as r4  # noqa: E402
from src import textured_mesh as tm  # noqa: E402
import eval_room_box as erb  # noqa: E402
import viewer_check as vc  # noqa: E402

W, H = 1024, 768
MAGENTA = "#ff00ff"


def page_html(src: str, tone: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><script type="module" src="{vc.MV_JS}"></script>'
            f'<style>body{{margin:0;background:#fff}} model-viewer{{width:{W}px;height:{H}px;display:block}}</style>'
            f'</head><body><model-viewer id="mv" src="{src}" camera-controls interaction-prompt="none" '
            f'interpolation-decay="1000" shadow-intensity="0" exposure="1" tone-mapping="{tone}"></model-viewer>'
            '</body></html>')


def shoot(page, url, views, bg):
    page.goto(url)
    page.wait_for_function("() => { const m = document.getElementById('mv'); return m && m.loaded; }", timeout=180000)
    out = []
    for v in views:
        page.evaluate("""(v) => { const m = document.getElementById('mv');
            m.style.background = v.bg; m.maxFieldOfView = '120deg'; m.minFieldOfView = '10deg';
            m.minCameraOrbit = 'auto 0deg 0.001m'; m.maxCameraOrbit = 'auto 180deg 100m';
            m.cameraTarget = v.target; m.cameraOrbit = v.orbit; m.fieldOfView = v.fov; m.jumpCameraToGoal(); }""",
                      {**v, "bg": bg})
        page.wait_for_timeout(900)
        png = page.locator("#mv").screenshot()
        out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def lon_A(box, yaw_B):
    d = box["Q"] @ np.array([np.sin(np.radians(yaw_B)), 0.0, np.cos(np.radians(yaw_B))])
    return float(np.degrees(np.arctan2(d[0], d[2])))


def magenta(img):
    return float(((img[..., 0] > 235) & (img[..., 1] < 25) & (img[..., 2] > 235)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    sys.path.insert(0, str(ROOT / "app"))
    import app as appmod
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    lpips = lambda x, y: float(lp(*[torch.from_numpy(i).permute(2, 0, 1)[None].float().div(255) for i in (x, y)]))  # noqa: E731
    id2label = erb.id2label_ade()
    sel = json.loads((ROOT / "results/tables/room_v4_selection.json").read_text())
    import eval_room_v4 as er4
    opts = dict(er4.VARIANTS[sel["geometry"]])
    if sel["denoise"] != "none":
        opts["denoise"] = sel["denoise"].split("+")[-1]
    rooms = []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        ana = {"class_map": cv2.imread(str(d / "v3/class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
               "inst_map": cv2.imread(str(d / "v3/inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((d / "v3/inst_v3.json").read_text())}
        dist, tex, hires = np.load(d / "distance.npy"), vc.rgb(d / "texture.png"), vc.rgb(d / "v4/pano_4096_detail.png")
        res = p4v.build(dist, tex, ana, id2label, hires=hires)
        p4v.save(res, d / "v4", title=prompt, pano=hires)              # shots / plan / sphere from the 4096 panorama
        meshes = r4.build_meshes(dist, hires, res, ana, id2label, opts)
        files = r4.export(meshes, d / "v4")
        nomip = d / "v4/room_v4_nomip.glb"
        tm.export_glb(meshes["inside"], nomip, double_sided=False, mipmaps=False, jpeg_quality=r4.DEFAULTS["jpeg_q"])
        rooms.append({"prompt": prompt, "d": d, "box": res["box"], "shots": res["shots"]["shots"], "files": files,
                      "mesh": meshes["inside"], "nomip": nomip, "bg": res["bg_color"]})
    vc.OUT.mkdir(parents=True, exist_ok=True)
    srv = vc.serve()
    rows, samp, figs, opening = [], [], [], []
    mode = "headless" if a.headless else "headed"
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=a.headless, args=["--use-angle=default", "--ignore-gpu-blocklist"])
        except Exception as e:
            print("headed launch failed:", e)
            browser = p.chromium.launch(headless=True)
            mode = "headless (headed launch failed)"
        page = browser.new_page(viewport={"width": W + 40, "height": H + 40})
        page.goto(f"http://127.0.0.1:{vc.PORT}/README.md")
        print("WebGL2 max texture size:", page.evaluate(
            "() => document.createElement('canvas').getContext('webgl2').getParameter(3379)"))
        for r in rooms:
            box = r["box"]
            views = []
            for s in r["shots"]:
                _, vf = ds.view_rays(W, H, s["fov"])
                views.append({"target": "0m 0m 0m", "orbit": f"{-lon_A(box, s['yaw']):.3f}deg 90deg 0.01m",
                              "fov": f"{vf:.3f}deg"})
            refs = [s["image"] for s in r["shots"]]
            rel3 = (r["d"] / "v3/room_v3.glb").relative_to(ROOT).as_posix()
            rel4 = Path(r["files"]["inside"]).relative_to(ROOT).as_posix()
            html = vc.OUT / "page_v4check.html"
            url = f"http://127.0.0.1:{vc.PORT}/{html.relative_to(ROOT).as_posix()}"
            for ver, rel, tone in (("v3", rel3, "neutral"), ("v4", rel4, "neutral"), ("v4", rel4, "none")):
                html.write_text(page_html(f"/{rel}", tone), encoding="utf-8")
                ims = shoot(page, url, views, MAGENTA)
                for s, im, ref in zip(r["shots"], ims, refs):
                    rows.append({"room": r["prompt"], "version": ver, "tone_mapping": tone, "shot": s["letter"],
                                 "psnr": cv2.PSNR(im, ref),
                                 "ssim": ssim(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)),
                                 "lpips": lpips(im, ref),
                                 "sharpness_ratio": eq.laplacian_variance(im) / eq.laplacian_variance(ref),
                                 "delta_e": float(deltaE_ciede2000(rgb2lab(im), rgb2lab(ref)).mean()),
                                 "gap_share": magenta(im)})
                figs.append((r["prompt"], ver, tone, refs[0], ims[0]))
                if ver == "v4" and tone == "none":
                    cv2.imwrite(str(vc.OUT / f"v4_{pl.slug(r['prompt'], 0)}_H.png"), cv2.cvtColor(ims[0], cv2.COLOR_RGB2BGR))
            # sampling: floor pose, mipmaps vs not, against a 4x supersampled offline render
            y0 = r["shots"][0]["yaw"]
            fl = {"target": "0m 0m 0m", "orbit": f"{-lon_A(box, y0):.3f}deg 55deg 0.01m", "fov": "60deg"}   # polar 55 = 35 deg down
            ref_ss = _render_pitch(r["mesh"], box, y0, -35.0, 60.0, ss=4)
            for name, src in (("mipmaps", rel4), ("no mipmaps", r["nomip"].relative_to(ROOT).as_posix())):
                html.write_text(page_html(f"/{src}", "none"), encoding="utf-8")
                im = shoot(page, url, [fl], MAGENTA)[0]
                samp.append({"room": r["prompt"], "texture_filter": name, "psnr_vs_supersampled": cv2.PSNR(im, ref_ss),
                             "lpips_vs_supersampled": lpips(im, ref_ss)})
            # the app's opening view
            ins = appmod.inside_html(r["files"]["inside"], r["bg"], lon_A(box, y0))
            orbit = ins.split('camera-orbit="')[1].split('"')[0]
            fov = ins.split('field-of-view="')[1].split('"')[0]
            html.write_text(page_html(f"/{rel4}", "none"), encoding="utf-8")
            im = shoot(page, url, [{"target": "0m 0m 0m", "orbit": orbit, "fov": fov}], r["bg"])[0]
            opening.append((r["prompt"], im, refs[0]))
        browser.close()
    srv.shutdown()
    df = pd.DataFrame(rows)
    df["browser"] = mode
    df.to_csv(ROOT / "results/tables/viewer_v4.csv", index=False, float_format="%.4f")
    summ = df.groupby(["version", "tone_mapping"])[["psnr", "ssim", "lpips", "sharpness_ratio", "delta_e",
                                                    "gap_share"]].mean()
    print(mode)
    print(summ.round(4).to_string())
    summ.reset_index().to_csv(ROOT / "results/tables/viewer_v4_summary.csv", index=False, float_format="%.4f")
    sm = pd.DataFrame(samp)
    sm.to_csv(ROOT / "results/tables/viewer_v4_sampling.csv", index=False, float_format="%.4f")
    print(sm.groupby("texture_filter")[["psnr_vs_supersampled", "lpips_vs_supersampled"]].mean().round(3))
    sizes = pd.DataFrame([{"room": r["prompt"], "inside_mb": r["files"]["inside_mb"],
                           "dollhouse_mb": r["files"]["dollhouse_mb"],
                           "inside_atlas": "x".join(map(str, r["files"]["inside_atlas"])),
                           "v3_inside_mb": (r["d"] / "v3/room_v3.glb").stat().st_size / 2 ** 20} for r in rooms])
    sizes.to_csv(ROOT / "results/tables/viewer_v4_sizes.csv", index=False, float_format="%.2f")
    print(sizes.round(2).to_string())
    fig, axes = plt.subplots(3, 4, figsize=(24, 13.5), dpi=60)
    for i, r in enumerate(rooms[:3]):
        fs = [f for f in figs if f[0] == r["prompt"]]
        ims = [fs[0][3]] + [f[4] for f in fs]
        tt = ["design shot H (reference)"] + [f"{f[1]} in Chromium ({mode}), tone mapping {f[2]}" for f in fs]
        for j, (im, t) in enumerate(zip(ims[:4], tt[:4])):
            axes[i, j].imshow(im)
            axes[i, j].set_title(t + ("  -  " + r["prompt"] if j == 0 else ""), fontsize=12)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/viewer_v4_inside.png", facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(2, len(opening), figsize=(5 * len(opening), 7.5), dpi=60)
    for j, (t, im, ref) in enumerate(opening):
        axes[0, j].imshow(ref)
        axes[0, j].set_title(f"hero shot - {t}", fontsize=10)
        axes[1, j].imshow(im)
        axes[1, j].set_title("app inside viewer: opening view", fontsize=10)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/viewer_v4_opening.png", facecolor="white")
    plt.close(fig)


def _render_pitch(mesh, box, yaw_B, pitch, vfov, ss=4):
    """Offline render looking `pitch` degrees up (negative = down): the reference for the sampling check."""
    from src import room_mesh as rm
    hf = 2 * np.degrees(np.arctan(np.tan(np.radians(vfov) / 2) * W / H))
    r, _ = ds.view_rays(W * ss, H * ss, hf)
    p = np.radians(-pitch)                       # view Y is down: looking down = forward gets +Y
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    dA = (r @ rx.T @ ds.view_to_B(yaw_B).T @ box["Q"].T).reshape(-1, 3)
    dA /= np.linalg.norm(dA, axis=1, keepdims=True)
    rc = rm.Raycaster(mesh)
    ans = rc.scene.cast_rays(rc.o3d.core.Tensor(np.concatenate([np.zeros_like(dA), dA], 1).astype(np.float32)))
    hit = np.isfinite(ans["t_hit"].numpy())
    prim, b = ans["primitive_ids"].numpy(), ans["primitive_uvs"].numpy()
    img = np.zeros((len(hit), 3), np.float32)
    f = mesh["faces"][prim[hit]]
    wgt = np.stack([1 - b[hit, 0] - b[hit, 1], b[hit, 0], b[hit, 1]], 1)
    uv = (mesh["uvs"][f] * wgt[:, :, None]).sum(1)
    tex = mesh["texture"]
    img[hit] = r4.bilinear(tex, uv[:, 0] * tex.shape[1] - 0.5, uv[:, 1] * tex.shape[0] - 0.5)
    return cv2.resize(img.reshape(H * ss, W * ss, 3), (W, H), interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
