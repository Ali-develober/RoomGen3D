"""PLAN_V3 milestone 6: the viewers opened in a real browser (headed Chromium via Playwright).

For the six demo rooms (outputs/room360/<room>): the v2 meshes (room.glb, dollhouse.glb, double-sided) and the
v3 meshes (rebuilt here with the milestone-6 export: single-sided, ceiling removed in the dollhouse).
A local HTTP server serves the files to <model-viewer> 3.5 (the component the app uses; loaded from its CDN).
    inside     camera at the panorama centre, 0.3 m orbit radius, 75 deg FOV, azimuth 0 / 90 / 180 / 270.
               Gap measurement: the page background is set to pure magenta; the share of magenta pixels in
               the screenshot = how much of the view shows no surface (holes, cracks). The display screenshots
               use the real backgrounds (v2 white, v3 the room's mean colour).
    dollhouse  orbit from outside and above (polar 50 deg), azimuth 30 / 120 / 210 / 300, neutral background.
Also: top-down orthographic render next to the v3 floor plan for the six rooms.
Outputs: outputs/viewer_check/ (screenshots), results/tables/viewer_gaps.csv,
         results/figures/viewer_inside.png, viewer_dollhouse.png, plan_vs_topdown.png
Usage: python scripts/viewer_check.py [--headless]
"""

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402

OUT = ROOT / "outputs" / "viewer_check"
PORT = 8765
MV_JS = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"
AZ_IN = [0, 90, 180, 270]
AZ_DH = [30, 120, 210, 300]
MAGENTA = "#ff00ff"


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def page_html(src: str) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><script type="module" src="{MV_JS}"></script>'
            '<style>body{margin:0;background:#fff} model-viewer{width:900px;height:675px;display:block}</style>'
            f'</head><body><model-viewer id="mv" src="{src}" camera-controls interaction-prompt="none" '
            'interpolation-decay="1000" shadow-intensity="0" exposure="1" tone-mapping="neutral"></model-viewer>'
            '</body></html>')


def shoot(page, url: str, views: list[dict], bg: str) -> list[np.ndarray]:
    page.goto(url)
    page.wait_for_function("() => { const m = document.getElementById('mv'); return m && m.loaded; }", timeout=120000)
    out = []
    for v in views:
        page.evaluate("""(v) => { const m = document.getElementById('mv');
            m.style.background = v.bg; m.cameraTarget = v.target; m.cameraOrbit = v.orbit;
            m.fieldOfView = v.fov; m.minCameraOrbit = 'auto 0deg 0.01m'; m.maxCameraOrbit = 'auto 180deg 100m';
            m.jumpCameraToGoal(); }""", {**v, "bg": bg})
        page.wait_for_timeout(700)
        png = page.locator("#mv").screenshot()
        out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def magenta_share(img: np.ndarray) -> float:
    m = (img[..., 0] > 235) & (img[..., 1] < 25) & (img[..., 2] > 235)
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from transformers import AutoConfig
    id2label = {int(k): v for k, v in AutoConfig.from_pretrained(str(ROOT / "checkpoints/mask2former_ade_panoptic")).id2label.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    rooms = []
    for prompt, seed in pl.DEMO_PROMPTS:                      # v3 meshes with the milestone-6 export
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        v3 = d / "v3"
        ana = {"class_map": cv2.imread(str(v3 / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
               "inst_map": cv2.imread(str(v3 / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
               "instances": json.loads((v3 / "inst_v3.json").read_text())}
        res = p3v.build(np.load(d / "distance.npy"), rgb(d / "texture.png"), ana, id2label)
        p3v.save(res, v3, title=prompt)
        rooms.append({"prompt": prompt, "d": d, "cam_h3": res["box"]["camera_height"], "bg3": res["bg_color"],
                      "plan": res["plan_image"], "top": res["topdown"]})
    srv = serve()
    rows, shots = [], {}
    mode = "headless" if args.headless else "headed"
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=args.headless, args=["--use-angle=default", "--ignore-gpu-blocklist"])
        except Exception as e:                                # no display: fall back and say so
            print("headed launch failed:", e)
            browser = p.chromium.launch(headless=True)
            mode = "headless (headed launch failed)"
        page = browser.new_page(viewport={"width": 920, "height": 700})
        for r in rooms:
            rel = r["d"].relative_to(ROOT).as_posix()
            for ver, room_glb, doll_glb, cam_h, bg in (("v2", f"{rel}/room.glb", f"{rel}/dollhouse.glb", r["cam_h3"], "#ffffff"),
                                                         ("v3", f"{rel}/v3/room_v3.glb", f"{rel}/v3/dollhouse_v3.glb", r["cam_h3"], r["bg3"])):
                html = OUT / f"page_{ver}.html"
                inside = [{"target": "0m 0m 0m", "orbit": f"{a}deg 90deg 0.3m", "fov": "75deg"} for a in AZ_IN]
                html.write_text(page_html(f"/{room_glb}"), encoding="utf-8")
                url = f"http://127.0.0.1:{PORT}/{html.relative_to(ROOT).as_posix()}"
                gaps = shoot(page, url, inside, MAGENTA)
                show = shoot(page, url, inside, bg)
                html.write_text(page_html(f"/{doll_glb}"), encoding="utf-8")
                y = -cam_h + 0.9
                doll = shoot(page, url, [{"target": f"0m {y:.2f}m 0m", "orbit": f"{a}deg 50deg auto", "fov": "45deg"}
                                         for a in AZ_DH], "#e9e8e4")
                for a, g in zip(AZ_IN, gaps):
                    rows.append({"room": r["prompt"], "version": ver, "view": "inside", "azimuth": a,
                                 "gap_share": magenta_share(g)})
                shots[(r["prompt"], ver)] = {"inside": show, "dollhouse": doll}
                for k, ims in (("inside", show), ("dollhouse", doll)):
                    for a, im in zip(AZ_IN if k == "inside" else AZ_DH, ims):
                        cv2.imwrite(str(OUT / f"{pl.slug(r['prompt'], 0)}_{ver}_{k}_{a}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        browser.close()
    srv.shutdown()
    df = pd.DataFrame(rows)
    df["browser"] = mode
    summ = df.groupby("version")["gap_share"].agg(["mean", "max"]).reset_index()
    summ["browser"] = mode
    T, F = ROOT / "results/tables", ROOT / "results/figures"
    df.to_csv(T / "viewer_gaps_per_view.csv", index=False, float_format="%.5f")
    summ.to_csv(T / "viewer_gaps.csv", index=False, float_format="%.5f")
    print(mode)
    print(summ.to_string(index=False))

    sel = [r["prompt"] for r in rooms[:3]]
    for kind, azs in (("inside", AZ_IN), ("dollhouse", AZ_DH)):
        fig, axes = plt.subplots(2 * len(sel), 4, figsize=(20, 7.5 * len(sel)), dpi=70)
        for i, pr in enumerate(sel):
            for j, ver in enumerate(("v2", "v3")):
                for k, im in enumerate(shots[(pr, ver)][kind]):
                    ax = axes[2 * i + j, k]
                    ax.imshow(im)
                    ax.axis("off")
                    ax.set_title(f"{ver} {kind}, azimuth {azs[k]}°" + (f"  —  {pr}" if k == 0 else ""), fontsize=11)
        fig.suptitle(f"<model-viewer> screenshots in Chromium ({mode})", fontsize=14)
        fig.tight_layout()
        fig.savefig(F / f"viewer_{kind}.png", facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(len(rooms), 2, figsize=(16, 6.5 * len(rooms)), dpi=60)
    for i, r in enumerate(rooms):
        axes[i, 0].imshow(r["plan"])
        axes[i, 1].imshow(r["top"])
        axes[i, 0].set_title(f"v3 floor plan — {r['prompt']}", fontsize=12)
        axes[i, 1].set_title("v3 model, orthographic top view (same orientation)", fontsize=12)
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(F / "plan_vs_topdown.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
