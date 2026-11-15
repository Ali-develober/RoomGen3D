"""PLAN_V5 self-review: headed-browser screenshots of what the owner sees, v4 and v5, for the six demo rooms.

For each room and version a review page is written next to the room's files, with the same viewer markup as the app
(app.inside_html / app.inside_v5_html / app.dollhouse_html, model-viewer 3.5 from its CDN for the GLB viewers), served
over a local HTTP server and opened in headed Chromium (Playwright). Screenshots (element level):
    hero      the hero design shot at its display size and a 100 % centre crop
    gallery   all shots as the gallery shows them
    inside    the Inside view as it opens (v4: depth mesh 5 cm from the centre; v5: panorama viewer), and after the user
              moves / turns: v4 = 0.6 m from the centre (reachable in v4: radius up to 1 m), v5 = turned +120°
    doll      the dollhouse as it opens
    plan      the floor plan with view cones
-> outputs/v5_review/<room>/<version>_<what>.png, and results/figures/v5_before_after_<room>.png (v4 vs v5: hero, Inside,
dollhouse) + v5_before_after_overview.png. Round number (self-review loop) via --round.
Usage: python scripts/review_v5.py [--round 1] [--headless]
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v5_common as vc  # noqa: E402
from src import pipeline as pl  # noqa: E402

PORT = 8765
MV_JS = "https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"
OUT = ROOT / "outputs" / "v5_review"


def serve():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    h.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def rel(p: Path) -> str:
    return "/" + str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def mv(src, attrs, bg, tone="neutral", el="mv"):
    return (f'<model-viewer id="{el}" src="{src}" camera-controls disable-pan interaction-prompt="none" loading="eager" reveal="auto" {attrs} '
            f'interpolation-decay="1000" shadow-intensity="0" exposure="1" tone-mapping="{tone}" '
            f'style="width:960px;height:720px;display:block;background:{bg}"></model-viewer>')


def page(version, d, meta, hero_yaw_A=None, cam_h=1.4):
    shots = meta["shots"]
    imgs = "".join(f'<figure><img src="{rel(d / "shots" / s["file"])}" style="width:300px"><figcaption>{s["caption"]}'
                   f'</figcaption></figure>' for s in shots)
    hero = rel(d / "shots" / shots[0]["file"])
    if version == "v5":
        inside = f'<iframe id="inside" src="{rel(d / "inside_v5.html")}" style="width:960px;height:720px;border:0"></iframe>'
        doll_glb = d / "dollhouse_v5.glb"
        plan = d / "floorplan_v5.png"
    else:
        inside = mv(rel(d / "room_v4.glb"), f'camera-target="0m 0m 0m" camera-orbit="{-hero_yaw_A:.1f}deg 90deg 0.05m" '
                    'min-camera-orbit="auto 10deg 0.05m" max-camera-orbit="auto 170deg 1m" field-of-view="64deg" '
                    'min-field-of-view="30deg" max-field-of-view="90deg"', "#888", el="inside")
        doll_glb = d / "dollhouse_v4.glb"
        plan = d / "floorplan_v4.png"
    y = -float(cam_h) + 0.9
    doll = mv(rel(doll_glb), f'camera-target="0m {y:.2f}m 0m" camera-orbit="30deg 45deg auto" '
              'min-camera-orbit="auto 0deg auto" max-camera-orbit="auto 80deg auto" field-of-view="45deg"', "#e9e8e4", el="doll")
    html = (f'<!doctype html><html><head><meta charset="utf-8"><script type="module" src="{MV_JS}"></script>'
            '<style>body{margin:8px;font-family:sans-serif;background:#fff} figure{display:inline-block;margin:4px;'
            'vertical-align:top;width:300px;font-size:11px} section{margin:10px 0}</style></head><body>'
            f'<section id="gallery">{imgs}</section><section><img id="hero" src="{hero}" style="width:960px"></section>'
            f'<section>{inside}</section><section>{doll}</section><section><img id="plan" src="{rel(plan)}" '
            f'style="width:760px"></section></body></html>')
    p = d / f"review_{version}.html"
    p.write_text(html, encoding="utf-8")
    return p


def grab(loc) -> np.ndarray:
    b = loc.screenshot()
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def wait_mv(pg, el):
    pg.locator(f'#{el}').scroll_into_view_if_needed()
    pg.wait_for_function(f"() => {{ const m = document.getElementById('{el}'); return m && m.loaded; }}", timeout=180000)
    pg.wait_for_timeout(1200)


def hero_yaw_A_v4(slug, yaw_B):
    from src import room_box as rb
    d = ROOT / "outputs/room360" / slug
    dist = np.load(d / "distance.npy")
    cm = cv2.imread(str(d / "v3/class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    box = rb.fit_box(dist, cm, vc.ade_id2label())
    v = box["Q"] @ np.array([np.sin(np.radians(yaw_B)), 0.0, np.cos(np.radians(yaw_B))])
    return float(np.degrees(np.arctan2(v[0], v[2]))), box["camera_height"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--versions", nargs="+", default=["v4", "v5"])
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    srv = serve()
    shots = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=a.headless)
        pg = br.new_page(viewport={"width": 1000, "height": 900})
        for prompt, seed in pl.DEMO_PROMPTS:
            slug = pl.slug(prompt, seed)
            o = OUT / slug
            o.mkdir(parents=True, exist_ok=True)
            for ver in a.versions:
                d = ROOT / ("outputs/v5_demo" if ver == "v5" else "outputs/room360") / slug / ("" if ver == "v5" else "v4")
                if not (d / "shots.json").exists():
                    continue
                meta = json.loads((d / "shots.json").read_text())
                if ver == "v4":
                    for s in meta["shots"]:
                        s.setdefault("caption", s.get("caption", s["letter"]))
                    yA, ch = hero_yaw_A_v4(slug, meta["shots"][0]["yaw_B_deg"])
                    p = page(ver, d, meta, yA, ch)
                else:
                    p = page(ver, d, meta, cam_h=meta.get("camera_height_m", 1.4))
                pg.goto(f"http://127.0.0.1:{PORT}{rel(p)}")
                pg.wait_for_load_state("networkidle")
                res = {"gallery": grab(pg.locator("#gallery")), "hero": grab(pg.locator("#hero")),
                       "plan": grab(pg.locator("#plan"))}
                hero = vc.rgb(d / "shots" / meta["shots"][0]["file"])
                h, w = hero.shape[:2]
                res["hero_100"] = hero[h // 2 - 300: h // 2 + 300, w // 2 - 400: w // 2 + 400]
                if ver == "v5":
                    fr = pg.frame_locator("#inside")
                    pg.wait_for_function("() => { const f = document.getElementById('inside'); return f.contentWindow && "
                                         "f.contentWindow.viewerReady === true; }", timeout=120000)
                    pg.wait_for_timeout(500)
                    res["inside"] = grab(pg.locator("#inside"))
                    pg.evaluate("() => { const w = document.getElementById('inside').contentWindow; const v = w.getView(); "
                                "w.setView(v.yaw + 120, 0, v.hfov); }")
                    pg.wait_for_timeout(400)
                    res["inside_moved"] = grab(pg.locator("#inside"))
                    del fr
                else:
                    wait_mv(pg, "inside")
                    res["inside"] = grab(pg.locator("#inside"))
                    pg.evaluate("() => { const m = document.getElementById('inside'); const o = m.getCameraOrbit(); "
                                "m.cameraOrbit = `${o.theta}rad ${o.phi}rad 0.6m`; m.jumpCameraToGoal(); }")
                    pg.wait_for_timeout(1200)
                    res["inside_moved"] = grab(pg.locator("#inside"))
                wait_mv(pg, "doll")
                res["doll"] = grab(pg.locator("#doll"))
                for k, im in res.items():
                    vc.save_rgb(o / f"r{a.round}_{ver}_{k}.png", im)
                shots[(slug, ver)] = res
        br.close()
    srv.shutdown()
    over = []
    for prompt, seed in pl.DEMO_PROMPTS:
        slug = pl.slug(prompt, seed)
        if (slug, "v4") not in shots or (slug, "v5") not in shots:
            continue
        a4, a5 = shots[(slug, "v4")], shots[(slug, "v5")]
        fig, axes = plt.subplots(2, 4, figsize=(24, 9.4), dpi=75)
        for i, (lab, r) in enumerate((("v4", a4), ("v5", a5))):
            for j, (k, t) in enumerate((("hero", "hero design shot"), ("hero_100", "hero at 100 %"),
                                        ("inside_moved" if lab == "v4" else "inside", "Inside view" + (" (0.6 m from centre)" if lab == "v4" else " (as it opens)")),
                                        ("doll", "dollhouse"))):
                axes[i, j].imshow(r[k])
                axes[i, j].set_title(f"{lab}: {t}", fontsize=13)
                axes[i, j].axis("off")
        fig.suptitle(f"{prompt} (seed {seed}): v4 (top) vs v5 (bottom), headed Chromium", fontsize=15)
        fig.tight_layout()
        fig.savefig(vc.FIG / f"v5_before_after_{slug}.png", facecolor="white")
        plt.close(fig)
        over.append((prompt, a4, a5))
    if over:
        fig, axes = plt.subplots(len(over), 6, figsize=(30, 3.7 * len(over)), dpi=55, squeeze=False)
        for i, (t, a4, a5) in enumerate(over):
            for j, (r, k, lab) in enumerate(((a4, "hero", "v4 hero"), (a5, "hero", "v5 hero"), (a4, "inside_moved", "v4 inside (0.6 m)"),
                                             (a5, "inside", "v5 inside"), (a4, "doll", "v4 dollhouse"), (a5, "doll", "v5 dollhouse"))):
                axes[i, j].imshow(r[k])
                axes[i, j].set_title(lab + (f" - {t}" if j == 0 else ""), fontsize=11)
                axes[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(vc.FIG / "v5_before_after_overview.png", facecolor="white")
        plt.close(fig)
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
