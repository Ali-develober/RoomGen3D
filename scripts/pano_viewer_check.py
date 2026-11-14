"""PLAN_V4 milestone 1: the interactive 360° panorama viewer, opened in a real browser (headed Chromium via
Playwright, <model-viewer> 3.5 as in the app).

For the six demo rooms: outputs/room360/<room>/v4/pano_sphere.glb (written by scripts/eval_design_shots.py) is
opened with the app's settings (camera at the sphere centre) facing the hero view (theta = -longitude of the
hero direction, app.pano_html) with the hero shot's vertical FOV in a 4:3 element. The screenshot is compared
with the hero design shot (normalised cross-correlation of grey images, 225 x 300): the right orientation must
match clearly better than the mirrored sign (theta = +longitude) and the opposite direction.
Outputs: results/tables/pano_viewer_check.csv, results/figures/pano_viewer_check.png
Usage: python scripts/pano_viewer_check.py [--headless]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import design_shots as ds  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import room_box as rb  # noqa: E402
import eval_room_box as erb  # noqa: E402
import viewer_check as vc  # noqa: E402


def shoot(page, url: str, views: list[dict]) -> list[np.ndarray]:
    """As viewer_check.shoot, but with the app's FOV limits (model-viewer clamps to max-field-of-view, 45 deg by
    default) and its orbit limits (radius 0.01 m)."""
    page.goto(url)
    page.wait_for_function("() => { const m = document.getElementById('mv'); return m && m.loaded; }", timeout=120000)
    out = []
    for v in views:
        page.evaluate("""(v) => { const m = document.getElementById('mv');
            m.style.background = '#000000'; m.maxFieldOfView = '100deg'; m.minFieldOfView = '30deg';
            m.minCameraOrbit = 'auto 5deg 0.01m'; m.maxCameraOrbit = 'auto 175deg 0.01m';
            m.cameraTarget = v.target; m.cameraOrbit = v.orbit; m.fieldOfView = v.fov; m.jumpCameraToGoal(); }""", v)
        page.wait_for_timeout(700)
        png = page.locator("#mv").screenshot()
        out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    g = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (300, 225), interpolation=cv2.INTER_AREA).astype(float)
         for x in (a, b)]
    g = [(x - x.mean()) / (x.std() + 1e-9) for x in g]
    return float((g[0] * g[1]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    id2label = erb.id2label_ade()
    rooms = []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        meta = json.loads((d / "v4/shots.json").read_text())
        hero = meta["shots"][0]
        cm = cv2.imread(str(d / "v3/class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        box = rb.fit_box(np.load(d / "distance.npy"), cm, id2label)
        dc = np.array([0.0, 0.0, 1.0]) @ ds.view_to_cam(box, hero["yaw_B_deg"])
        lon = float(np.degrees(np.arctan2(dc[0], dc[2])))
        _, vfov = ds.view_rays(ds.SHOT_W, ds.SHOT_H, hero["hfov_deg"])
        rooms.append({"prompt": prompt, "d": d, "lon": lon, "vfov": vfov,
                      "hero": vc.rgb(d / "v4/shots" / hero["file"])})
    vc.OUT.mkdir(parents=True, exist_ok=True)
    srv = vc.serve()
    rows, figs = [], []
    mode = "headless" if args.headless else "headed"
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=args.headless, args=["--use-angle=default", "--ignore-gpu-blocklist"])
        except Exception as e:
            print("headed launch failed:", e)
            browser = p.chromium.launch(headless=True)
            mode = "headless (headed launch failed)"
        page = browser.new_page(viewport={"width": 920, "height": 700})
        for r in rooms:
            rel = (r["d"] / "v4/pano_sphere.glb").relative_to(ROOT).as_posix()
            html = vc.OUT / "page_pano.html"
            html.write_text(vc.page_html(f"/{rel}"), encoding="utf-8")
            url = f"http://127.0.0.1:{vc.PORT}/{html.relative_to(ROOT).as_posix()}"
            cand = {"app (theta = -lon)": -r["lon"], "mirrored sign (theta = +lon)": r["lon"],
                    "opposite (theta = -lon + 180)": -r["lon"] + 180}
            views = [{"target": "0m 0m 0m", "orbit": f"{t:.2f}deg 90deg 0.01m", "fov": f"{r['vfov']:.2f}deg"}
                     for t in cand.values()]
            ims = shoot(page, url, views)
            row = {"room": r["prompt"], "hero_lon_deg": r["lon"]}
            for (name, _), im in zip(cand.items(), ims):
                row[f"ncc_{name.split(' (')[0].replace(' ', '_')}"] = ncc(im, r["hero"])
            rows.append(row)
            figs.append((r["prompt"], r["hero"], ims[0]))
            cv2.imwrite(str(vc.OUT / f"pano_viewer_{pl.slug(r['prompt'], 0)}.png"), cv2.cvtColor(ims[0], cv2.COLOR_RGB2BGR))
        browser.close()
    srv.shutdown()
    df = pd.DataFrame(rows)
    df["browser"] = mode
    df.to_csv(ROOT / "results/tables/pano_viewer_check.csv", index=False, float_format="%.4f")
    print(df.round(3).to_string())
    fig, axes = plt.subplots(len(figs), 2, figsize=(12, 4.3 * len(figs)), dpi=70)
    for i, (t, a, b) in enumerate(figs):
        axes[i, 0].imshow(a); axes[i, 0].set_title(f"hero design shot - {t}", fontsize=10)
        axes[i, 1].imshow(b); axes[i, 1].set_title(f"360° viewer in Chromium ({mode}), opened facing the hero", fontsize=10)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/pano_viewer_check.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
