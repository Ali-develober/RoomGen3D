"""PLAN_V6 milestone 1 metrics: layout rule violations, asset coverage per class, scene build / render time.

    rules     1400 layouts (5 room types x 14 styles x 20 seeds): violations per rule, missing required
              pieces, pieces per room
    rooms     the PLAN_V5 room sets (dev, held-out, demo): scene build time, equirectangular 2048 x 1024
              and perspective 768 x 576 ray-cast time, models used, fit distortion
    figure    demo rooms: floor plan | shaded proxy panorama | depth condition | instance ids
Outputs: results/tables/v6_scene_rules.csv, v6_scene_rooms.csv, v6_scene_coverage.csv,
         results/figures/v6_scene_demo.png;  scene JSON per room in outputs/v6_scene/<id>/layout.json
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import assets_v6 as A  # noqa: E402
from src import layout_v6 as LV  # noqa: E402
from src import plan_v6 as PV  # noqa: E402
from src import raycast_v6 as R  # noqa: E402
from src import scene_v6 as S  # noqa: E402
import v5_common as C  # noqa: E402

OUT = ROOT / "outputs" / "v6_scene"


def rules() -> pd.DataFrame:
    rows = []
    for room in LV.ROOMS:
        for style in LV.STYLES:
            for seed in range(1, 21):
                lay = LV.generate(f"a {style} style {room}", seed)
                v = LV.violations(lay)
                rows.append({"room": room, "style": style, "seed": seed, **v, "missing": LV.missing(lay),
                             "pieces": len(lay["items"]), "area": lay["W"] * lay["D"], "H": lay["H"]})
    return pd.DataFrame(rows)


def rooms() -> tuple[pd.DataFrame, list]:
    rows, demo = [], []
    for r in C.rooms().itertuples():
        lay = LV.generate(r.prompt, r.seed)
        sc = S.build(lay)
        rd = S.Renderer(sc)
        t0 = time.perf_counter()
        e = rd.equirect(sc["layout"]["camera"], 2048, 1024)
        t_eq = time.perf_counter() - t0
        t0 = time.perf_counter()
        rd.perspective(sc["layout"]["camera"], 0, 0, 75, 768, 576)
        t_p = time.perf_counter() - t0
        dist = [max(s) / min(s) for s in sc["fit"].values()]
        d = OUT / r.id
        d.mkdir(parents=True, exist_ok=True)
        LV.save(sc["layout"], d / "layout.json")
        rows.append({"id": r.id, "set": r.set, "room": r.room, "pieces": len(lay["items"]),
                     "violations": LV.violations(sc["layout"])["total"], "missing": LV.missing(lay),
                     "build_s": round(sc["build_s"], 2), "equirect_s": round(t_eq, 2), "persp_s": round(t_p, 3),
                     "max_distortion": round(max(dist), 3), "models": ";".join(sorted(set(sc["models"].values())))})
        if r.set == "demo":
            plan = PV.draw(sc["layout"], models=sc["models"])
            demo.append((r.prompt, plan, R.shade(e), S.depth_control(e["t"]), S.inst_colors(e["inst"])))
        print(r.id, rows[-1]["violations"], rows[-1]["equirect_s"], flush=True)
    return pd.DataFrame(rows), demo


def coverage(rule_df: pd.DataFrame) -> pd.DataFrame:
    need = Counter()
    for room in LV.ROOMS:
        for style in LV.STYLES[:4]:
            for seed in range(1, 6):
                need.update(i["cls"] for i in LV.generate(f"a {style} style {room}", seed)["items"])
    rows = []
    for cls in sorted(need):
        opts = A.CATALOG.get(cls, [])
        rows.append({"class": cls, "occurrences": need[cls], "models": len(opts),
                     "sources": ";".join(sorted({o[0] for o in opts})), "covered": len(opts) > 0,
                     "model_ids": ";".join(o[1] for o in opts)})
    return pd.DataFrame(rows)


def figure(demo: list) -> None:
    tiles = []
    for prompt, plan, sh, dc, ic in demo:
        h = 360
        pl = Image.fromarray(plan)
        pl = pl.resize((int(pl.width * h / pl.height), h))
        rest = [Image.fromarray(x).resize((720, h)) for x in (sh, dc, ic)]
        row = Image.new("RGB", (pl.width + 3 * 720 + 30, h + 24), "white")
        ImageDraw.Draw(row).text((4, 4), prompt, fill=(0, 0, 0))
        x = 0
        for im in [pl] + rest:
            row.paste(im, (x, 24))
            x += im.width + 10
        tiles.append(row)
    W = max(t.width for t in tiles)
    out = Image.new("RGB", (W, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        out.paste(t, (0, y))
        y += t.height
    out.save(C.FIG / "v6_scene_demo.png")


def main():
    t0 = time.perf_counter()
    rd = rules()
    rd.to_csv(C.TAB / "v6_scene_rules.csv", index=False)
    print("rules", round(time.perf_counter() - t0, 1), "s;", rd[["total", "missing"]].sum().to_dict())
    print(rd.groupby("room")[["overlap", "outside", "wall_align", "facing", "door", "window", "access", "missing",
                              "pieces", "area", "H"]].mean().round(3))
    cov = coverage(rd)
    cov.to_csv(C.TAB / "v6_scene_coverage.csv", index=False)
    print(cov[["class", "occurrences", "models", "covered"]].to_string(index=False))
    df, demo = rooms()
    df.to_csv(C.TAB / "v6_scene_rooms.csv", index=False)
    print(df.groupby("set")[["pieces", "violations", "missing", "build_s", "equirect_s", "persp_s", "max_distortion"]]
          .mean().round(3))
    figure(demo)


if __name__ == "__main__":
    main()
