"""PLAN_V5 milestone 4: the Inside view without tearing.

--part mesh     how far can the v4 depth mesh ("Experimental: walkable mesh") be walked before it shows holes?
                Dev rooms (milestone-3 crops: 4096 panorama + the selected candidate): v4 inside mesh
                (src/room_v4.build_meshes), offline ray casting at the hero yaw / FOV from the panorama centre moved by
                t in TRANSLATIONS (m) in 8 horizontal directions. Hole share = rendered pixels whose surface point was
                NOT seen from the panorama centre (|P| > observed distance along dir(P) * 1.04 + 10 cm, the shell tolerance): their colour
                is invented (inpainted shell) or stretched (triangles across depth edges). Limit (fixed beforehand):
                the largest t whose 90th percentile hole share over rooms and directions is <= HOLE_MAX.
                -> results/tables/v5_viewer_mesh.csv, v5_viewer_selection.json, figure v5_viewer_mesh.png
--part browser  headed Chromium (Playwright): the v5 Inside page (src/pano_viewer.py) of each demo room, opened as
                the app opens it (hero yaw, hero FOV), screenshot at 1024 x 768; compared with the hero raw crop
                (normalised cross-correlation of grey images; wrong orientation would score low) and two more yaws.
                -> results/tables/v5_viewer_browser.csv, results/figures/v5_inside_<room>.png, v5_inside_overview.png
Usage: python scripts/eval_v5_viewer.py --part mesh|browser [--headless]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v5_rooms as rr  # noqa: E402
import v5_common as vc  # noqa: E402
from src import equirect as eq  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import room_v4 as r4  # noqa: E402

ROOT = vc.ROOT
SHOTS = ROOT / "outputs" / "v5_shots"
TRANSLATIONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
HOLE_MAX = 0.02


def hole_share(mesh, box, dist, yaw, fov, origin_A, w=320, h=240):
    from src import design_shots as ds
    from src import room_mesh as rm
    r, _ = ds.view_rays(w, h, fov)
    dA = (r @ ds.view_to_B(yaw).T @ box["Q"].T).reshape(-1, 3)
    dA /= np.linalg.norm(dA, axis=1, keepdims=True)
    rc = rm.Raycaster(mesh)
    o = np.asarray(origin_A, float)
    ans = rc.scene.cast_rays(rc.o3d.core.Tensor(np.concatenate([np.broadcast_to(o, dA.shape), dA], 1).astype(np.float32)))
    t = ans["t_hit"].numpy()
    hit = np.isfinite(t)
    P = o + dA[hit] * t[hit, None]                       # frame A
    Pc = P @ box["R"]                                    # pano camera frame
    rr_ = np.linalg.norm(Pc, axis=1)
    u, v = eq.dirs_to_uv(Pc / rr_[:, None], dist.shape[1], dist.shape[0])
    obs = dist[np.clip(np.round(v).astype(int), 0, dist.shape[0] - 1), np.round(u).astype(int) % dist.shape[1]]
    unseen = rr_ > obs * 1.04 + 0.10                   # same tolerance as the shell (room_box.HIDDEN_REL / ABS)
    return float((unseen.sum() + (~hit).sum()) / len(t))


def part_mesh():
    id2label = vc.ade_id2label()
    df = vc.rooms(["dev"])
    rows = []
    for r in tqdm(list(df.itertuples()), desc="rooms"):
        mp = SHOTS / r.id / "meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        n = meta["name"]
        g = np.load(ROOT / "outputs/v5_rooms" / r.id / f"{n}_geo.npz")
        dist = g["dist"]
        ana = rr.load_ana(r.id, n)
        hires = vc.rgb(SHOTS / r.id / "pano_4096.jpg")
        tex = cv2.resize(hires, (2048, 1024), interpolation=cv2.INTER_AREA)
        res = p3v.build(dist, tex, ana, id2label, open_sides=True)
        mesh = r4.build_meshes(dist, hires, res, ana, id2label)["inside"]
        hero = meta["shots"][0]
        box = res["box"]
        for t in TRANSLATIONS:
            for k in range(8 if t > 0 else 1):
                a = 2 * np.pi * k / 8
                oB = np.array([np.sin(a) * t, 0.0, np.cos(a) * t])
                rows.append({"id": r.id, "t": t, "dir": k,
                             "hole_share": hole_share(mesh, box, dist, hero["yaw"], hero["fov"], oB @ box["Q"].T)})
    tab = pd.DataFrame(rows)
    tab.to_csv(vc.TAB / "v5_viewer_mesh.csv", index=False, float_format="%.5f")
    q = tab.groupby("t")["hole_share"].quantile(0.9)
    ok = q[q <= HOLE_MAX]
    limit = float(ok.index.max()) if len(ok) else 0.0
    vc.write_json(vc.TAB / "v5_viewer_selection.json", {"max_translation_m": limit, "hole_max": HOLE_MAX,
                                                        "p90_by_t": q.to_dict(), "median_by_t": tab.groupby("t")["hole_share"].median().to_dict(),
                                                        "rule": "largest t with 90th percentile hole share <= 2 % (dev rooms, 8 directions, hero view)"})
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    for qq, lab in ((0.5, "median"), (0.9, "90th percentile")):
        s = tab.groupby("t")["hole_share"].quantile(qq) * 100
        ax.plot(s.index, s.values, marker="o", label=lab)
    ax.axhline(HOLE_MAX * 100, color="gray", ls="--", label="threshold")
    ax.axvline(limit, color="tab:red", ls=":", label=f"limit {limit:.2f} m")
    ax.set_xlabel("camera translation from the panorama centre (m)")
    ax.set_ylabel("pixels not seen from the centre (%)")
    ax.set_title("Depth-mesh inside view: holes vs walking distance (dev rooms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vc.FIG / "v5_viewer_mesh.png")
    plt.close(fig)
    print(q, "limit", limit)


def ncc(a, b):
    a = cv2.cvtColor(cv2.resize(a, (300, 225), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(cv2.resize(b, (300, 225), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def shoot_viewer(page, html: Path, views: list[tuple]) -> list[np.ndarray]:
    page.goto("file:///" + str(html.resolve()).replace("\\", "/"))
    page.wait_for_function("() => window.viewerReady === true", timeout=120000)
    page.wait_for_timeout(400)
    out = []
    for yaw, pitch, fov in views:
        if yaw is not None:
            page.evaluate(f"() => window.setView({yaw}, {pitch}, {fov})")
            page.wait_for_timeout(250)
        b = page.screenshot()
        out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return out


def part_browser(headless: bool, room_dir: str = "outputs/v5_demo"):
    from playwright.sync_api import sync_playwright
    from src import pipeline as pl
    rows, over = [], []
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=headless)
        page = br.new_page(viewport={"width": 1024, "height": 768})
        for prompt, seed in pl.DEMO_PROMPTS:
            d = ROOT / room_dir / pl.slug(prompt, seed)
            html, sj = d / "inside_v5.html", d / "shots.json"
            if not html.exists():
                continue
            meta = json.loads(sj.read_text())["shots"]
            hero = meta[0]
            ref = vc.rgb(d / "shots" / hero["raw"])
            imgs = shoot_viewer(page, html, [(None, 0, None), (hero["yaw_B_deg"] + 120, 0, 90), (hero["yaw_B_deg"] - 120, 0, 90),
                                             (hero["yaw_B_deg"] + 180, 0, hero["hfov_deg"])])
            opening = imgs[0]
            rows.append({"room": pl.slug(prompt, seed), "ncc_hero": ncc(opening, ref), "ncc_opposite": ncc(imgs[3], ref)})
            fig, axes = plt.subplots(1, 4, figsize=(20, 3.9), dpi=80)
            for ax, im, t in zip(axes, [ref, opening, imgs[1], imgs[2]],
                                 ["hero design shot (raw crop)", "Inside viewer as it opens", "turned +120°", "turned -120°"]):
                ax.imshow(im)
                ax.set_title(t, fontsize=12)
                ax.axis("off")
            fig.suptitle(f"{prompt}: v5 Inside view (panorama viewer, headed Chromium)", fontsize=13)
            fig.tight_layout()
            fig.savefig(vc.FIG / f"v5_inside_{pl.slug(prompt, seed)}.png", facecolor="white")
            plt.close(fig)
            over.append((prompt, opening, imgs[1]))
            cv2.imwrite(str(d / "screen_inside.png"), cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
        br.close()
    tab = pd.DataFrame(rows)
    tab.to_csv(vc.TAB / "v5_viewer_browser.csv", index=False, float_format="%.4f")
    if over:
        fig, axes = plt.subplots(len(over), 2, figsize=(11, 4.1 * len(over)), dpi=70, squeeze=False)
        for i, (t, a, b) in enumerate(over):
            axes[i, 0].imshow(a)
            axes[i, 0].set_title(f"{t} - opening view", fontsize=11)
            axes[i, 1].imshow(b)
            axes[i, 1].set_title("turned +120°", fontsize=11)
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(vc.FIG / "v5_inside_overview.png", facecolor="white")
        plt.close(fig)
    print(tab.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="mesh")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--room-dir", default="outputs/v5_demo")
    a = ap.parse_args()
    if a.part == "mesh":
        part_mesh()
    else:
        part_browser(a.headless, a.room_dir)


if __name__ == "__main__":
    main()
