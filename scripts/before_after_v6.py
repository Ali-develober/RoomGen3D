"""PLAN_V6 final: one before / after figure per demo room, v5 (outputs/v5_demo) vs v6 (outputs/v6_demo):
hero design shot, Inside view (as it opens), dollhouse, floor plan. v5 Inside / dollhouse are screenshotted live in
headed Chromium with the app's viewer markup (local model-viewer); v6 views come from the last review round.
-> results/figures/v6_before_after_<room>.png and v6_before_after_overview.png
Usage: python scripts/before_after_v6.py --round 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v5_common as vc  # noqa: E402
from review_v6 import MV_JS, PORT, grab, rel, serve  # noqa: E402
from src import pipeline as pl  # noqa: E402


def v5_page(d: Path) -> Path:
    meta = json.loads((d / "summary_v5.json").read_text()) if (d / "summary_v5.json").exists() else {}
    cam_h = float(meta.get("camera_height_m", meta.get("box", {}).get("camera_height", 1.1)) or 1.1)
    y = -cam_h + 0.9
    html = (f'<!doctype html><html><head><meta charset="utf-8"><script type="module" src="{MV_JS}"></script></head>'
            f'<body style="margin:8px"><iframe id="inside" src="{rel(d / "inside_v5.html")}" style="width:960px;height:720px;'
            f'border:0"></iframe><model-viewer id="doll" src="{rel(d / "dollhouse_v5.glb")}" camera-controls loading="eager" '
            f'camera-target="0m {y:.2f}m 0m" camera-orbit="30deg 45deg auto" field-of-view="45deg" shadow-intensity="0" '
            'exposure="1" tone-mapping="neutral" style="width:960px;height:720px;display:block;background:#e9e8e4">'
            '</model-viewer></body></html>')
    p = d / "before_after_v5.html"
    p.write_text(html, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    srv = serve()
    v5 = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=False)
        pg = br.new_page(viewport={"width": 1000, "height": 1600})
        for prompt, seed in pl.DEMO_PROMPTS:
            slug = pl.slug(prompt, seed)
            d = ROOT / "outputs" / "v5_demo" / slug
            if not d.exists():
                continue
            pg.goto(f"http://127.0.0.1:{PORT}{rel(v5_page(d))}")
            pg.wait_for_function("() => { const f = document.getElementById('inside'); return f.contentWindow && "
                                 "f.contentWindow.viewerReady === true; }", timeout=120000)
            pg.wait_for_timeout(500)
            r = {"inside": grab(pg.locator("#inside"))}
            pg.locator("#doll").scroll_into_view_if_needed()
            pg.wait_for_function("() => { const m = document.getElementById('doll'); return m && m.loaded; }", timeout=180000)
            pg.wait_for_timeout(1500)
            r["doll"] = grab(pg.locator("#doll"))
            r["hero"] = vc.rgb(d / "shots" / "H.jpg")
            r["plan"] = vc.rgb(d / "floorplan_v5.png")
            v5[slug] = r
        br.close()
    srv.shutdown()
    over = []
    for prompt, seed in pl.DEMO_PROMPTS:
        slug = pl.slug(prompt, seed)
        rv = ROOT / "outputs" / "v6_review" / slug
        if slug not in v5 or not (rv / f"r{a.round}_shot_H.png").exists():
            continue
        v6 = {"hero": vc.rgb(rv / f"r{a.round}_shot_H.png"), "inside": vc.rgb(rv / f"r{a.round}_inside_0.png"),
              "doll": vc.rgb(rv / f"r{a.round}_doll_a.png"), "plan": vc.rgb(rv / f"r{a.round}_plan.png")}
        fig, axes = plt.subplots(2, 4, figsize=(26, 10.5), dpi=70)
        for i, (lab, r) in enumerate((("v5", v5[slug]), ("v6", v6))):
            for j, (k, t) in enumerate((("hero", "hero design shot"), ("inside", "Inside view"), ("doll", "dollhouse"),
                                        ("plan", "floor plan"))):
                axes[i, j].imshow(r[k])
                axes[i, j].set_title(f"{lab}: {t}", fontsize=14)
                axes[i, j].axis("off")
        fig.suptitle(f"{prompt} (seed {seed}): v5 image-first (top) vs v6 scene-first (bottom)", fontsize=16)
        fig.tight_layout()
        fig.savefig(vc.FIG / f"v6_before_after_{slug}.png", facecolor="white")
        plt.close(fig)
        over.append((prompt, v5[slug], v6))
        print("figure", slug, flush=True)
    if over:
        fig, axes = plt.subplots(len(over), 6, figsize=(33, 4.2 * len(over)), dpi=55, squeeze=False)
        for i, (t, r5, r6) in enumerate(over):
            for j, (r, k, lab) in enumerate(((r5, "hero", "v5 hero"), (r6, "hero", "v6 hero"), (r5, "inside", "v5 Inside"),
                                             (r6, "inside", "v6 Inside"), (r5, "doll", "v5 dollhouse"), (r6, "doll", "v6 dollhouse"))):
                axes[i, j].imshow(r[k])
                axes[i, j].set_title(lab + (f" - {t}" if j == 0 else ""), fontsize=12)
                axes[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(vc.FIG / "v6_before_after_overview.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
