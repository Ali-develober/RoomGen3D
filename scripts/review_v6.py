"""PLAN_V6 review loop: headed-browser screenshots of every view the owner checks, per room.

For each room in outputs/v6_demo/<slug>/ (built by scripts/build_v6_rooms.py) a review page is written with the same
viewer markup as the app (the panorama Inside viewer page in an iframe; model-viewer 3.5 from app/static for the
dollhouse), served locally and opened in headed Chromium (Playwright). Element screenshots:
    shot_<H|H2|A|B|C|D>   every design shot at full size
    inside_<0|90|180|270> the Inside viewer turned 0 / 90 / 180 / 270° from the hero direction (level, 80° FOV)
    doll_a, doll_b        the dollhouse from two orbit angles (30° / 210° azimuth, 50° / 35° from the vertical)
    topdown, plan         the top-down render and the floor plan
-> outputs/v6_review/<slug>/r<round>_<what>.png; contact sheet results/figures/v6_review_r<round>_<slug>.png
Usage: python scripts/review_v6.py --round 1 [--rooms slug1,slug2] [--src outputs/v6_demo]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v5_common as vc  # noqa: E402
from src import pipeline as pl  # noqa: E402

PORT = 8766
OUT = ROOT / "outputs" / "v6_review"
MV_JS = "/app/static/model-viewer.min.js"
SHOTS = ["H", "H2", "A", "B", "C", "D"]


def serve():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def rel(p: Path) -> str:
    return "/" + str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def page(d: Path) -> Path:
    shots = "".join(f'<section><img id="shot_{n}" src="{rel(d / "shots" / (n + ".jpg"))}" style="width:1152px"></section>'
                    for n in SHOTS if (d / "shots" / f"{n}.jpg").exists())
    lay = json.loads((d / "layout.json").read_text())
    y = 0.9
    doll = (f'<model-viewer id="doll" src="{rel(d / "dollhouse_v6.glb")}" camera-controls disable-pan '
            f'interaction-prompt="none" loading="eager" reveal="auto" camera-target="auto {y:.2f}m auto" '
            'camera-orbit="30deg 50deg auto" min-camera-orbit="auto 0deg auto" max-camera-orbit="auto 80deg auto" '
            'field-of-view="40deg" interpolation-decay="1000" shadow-intensity="0" exposure="1" tone-mapping="neutral" '
            'style="width:960px;height:720px;display:block;background:#e9e8e4"></model-viewer>')
    html = (f'<!doctype html><html><head><meta charset="utf-8"><script type="module" src="{MV_JS}"></script>'
            '<style>body{margin:8px;font-family:sans-serif;background:#fff} section{margin:10px 0}</style></head><body>'
            f'{shots}<section><iframe id="inside" src="{rel(d / "inside_v6.html")}" style="width:960px;height:720px;'
            f'border:0"></iframe></section><section>{doll}</section>'
            f'<section><img id="topdown" src="{rel(d / "topdown_v6.png")}" style="width:900px"></section>'
            f'<section><img id="plan" src="{rel(d / "floorplan_v6.png")}" style="width:900px"></section></body></html>')
    p = d / "review_v6.html"
    p.write_text(html, encoding="utf-8")
    _ = lay
    return p


def grab(loc) -> np.ndarray:
    b = loc.screenshot()
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sheet(res: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in res]
    n = len(keys)
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.6 * rows), dpi=70, squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(res[keys[i]])
            ax.set_title(keys[i], fontsize=13)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--src", default="outputs/v6_demo")
    ap.add_argument("--rooms", default="")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    src = ROOT / a.src
    rooms = a.rooms.split(",") if a.rooms else [pl.slug(p, s) for p, s in pl.DEMO_PROMPTS]
    srv = serve()
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=a.headless)
        pg = br.new_page(viewport={"width": 1200, "height": 950})
        for slug in rooms:
            d = src / slug
            if not (d / "layout.json").exists():
                print("missing", d)
                continue
            o = OUT / slug
            o.mkdir(parents=True, exist_ok=True)
            p = page(d)
            pg.goto(f"http://127.0.0.1:{PORT}{rel(p)}")
            pg.wait_for_load_state("networkidle")
            res = {}
            for n in SHOTS:
                if pg.locator(f"#shot_{n}").count():
                    res[f"shot_{n}"] = grab(pg.locator(f"#shot_{n}"))
            pg.locator("#inside").scroll_into_view_if_needed()
            pg.wait_for_function("() => { const f = document.getElementById('inside'); return f.contentWindow && "
                                 "f.contentWindow.viewerReady === true; }", timeout=120000)
            pg.wait_for_timeout(500)
            v0 = pg.evaluate("() => document.getElementById('inside').contentWindow.getView()")
            for k in (0, 90, 180, 270):
                pg.evaluate(f"() => document.getElementById('inside').contentWindow.setView({v0['yaw'] + k}, 0, 80)")
                pg.wait_for_timeout(300)
                res[f"inside_{k}"] = grab(pg.locator("#inside"))
            pg.locator("#doll").scroll_into_view_if_needed()
            pg.wait_for_function("() => { const m = document.getElementById('doll'); return m && m.loaded; }", timeout=180000)
            pg.wait_for_timeout(1500)
            res["doll_a"] = grab(pg.locator("#doll"))
            pg.evaluate("() => { const m = document.getElementById('doll'); m.cameraOrbit = '210deg 35deg auto'; "
                        "m.jumpCameraToGoal(); }")
            pg.wait_for_timeout(1500)
            res["doll_b"] = grab(pg.locator("#doll"))
            res["topdown"] = grab(pg.locator("#topdown"))
            res["plan"] = grab(pg.locator("#plan"))
            for k, im in res.items():
                vc.save_rgb(o / f"r{a.round}_{k}.png", im)
            sheet(res, vc.FIG / f"v6_review_r{a.round}_{slug}.png", f"{slug} - review round {a.round} (headed Chromium)")
            print("review", slug, len(res), flush=True)
        br.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
