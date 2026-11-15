"""PLAN_V5 final: one table with every before / after metric of the round (results/tables/v5_summary.csv), read from the
milestone tables. 'data' says which rooms; 'v4' is the v4 setting on the same prompts and seeds, 'v5' the adopted one.
Usage: python scripts/v5_summary.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/tables"
sys.path.insert(0, str(ROOT))


def j(name):
    try:
        return json.loads((TAB / name).read_text())
    except (OSError, ValueError):
        return {}


def main():
    rows = []

    def add(m, metric, data, v4, v5, higher_better):
        better = None if higher_better is None else (v5 > v4 if higher_better else v5 < v4)
        rows.append({"milestone": m, "metric": metric, "data": data, "v4": v4, "v5": v5, "v5_better": better})

    p = pd.read_csv(TAB / "v5_pano_summary.csv").set_index(["config", "set"])
    for split, lab in (("test", "held-out"), ("demo", "demo")):
        for c, name, hb in (("kid_x1e3", "panorama views KID x1e3 vs MMIS", False), ("clip", "panorama views CLIP", True),
                            ("brisque", "panorama views BRISQUE", False), ("sharpness", "panorama views Laplacian sharpness", True),
                            ("boxiness", "boxiness (one closed room)", True), ("seam_ratio_median", "seam ratio (1 = invisible)", None),
                            ("seconds", "seconds per panorama", False), ("peak_vram_gb", "peak VRAM GB", False),
                            ("peak_ram_gb", "peak process RAM GB", False)):
            add("1 panorama source", name + " (v4 SD 2.1 + ESRGAN vs SDXL + 360Redmond)", lab,
                p.loc[("sd21", split), c], p.loc[("xl_redmond", split), c], hb)
    cs = pd.read_csv(TAB / "v5_camera_summary.csv").set_index(["set", "version"])
    for split, lab in (("test", "held-out"), ("demo", "demo"), ("dev", "dev")):
        for c, name, hb in (("hero_valid", "rooms whose hero passes the distance test", True),
                            ("shots_valid", "shots (hero + 4 walls) passing the test", True),
                            ("area_m2", "fitted room area m2", None), ("d_near_m", "camera to nearest furniture m", True),
                            ("hero_median_depth", "hero median distance m", True)):
            add("2 camera / room size", name, lab, cs.loc[(split, "v4"), c], cs.loc[(split, "v5"), c], hb)
    s = pd.read_csv(TAB / "v5_shots_summary.csv").set_index(["set", "config"])
    sel = j("v5_shots_decision.json").get("selected", "cn_lora_s0.55")
    base = sel.replace("cn_lora", "cn_base")
    for split, lab in (("test", "held-out"), ("demo", "demo")):
        for c, name, hb in (("brisque", "shot BRISQUE", False), ("kid_x1e3", "shot KID x1e3 vs MMIS", False),
                            ("clip", "shot CLIP", True), ("sharpness", "shot Laplacian sharpness", True),
                            ("edge_f1", "structure kept: edge F1 vs crop", True), ("seconds", "seconds per shot", False)):
            add("3 shot detail", name + f" (v4 ESRGAN vs {sel})", lab, s.loc[(split, "esrgan"), c], s.loc[(split, sel), c], hb)
            if (split, base) in s.index:
                add("3 shot detail", name + f" (LoRA effect: {base} vs {sel})", lab, s.loc[(split, base), c], s.loc[(split, sel), c], hb)
    vsel = j("v5_viewer_selection.json")
    if vsel:
        add("4 inside view", "walkable distance with < 2 % holes (m): v4 mesh allowed 1 m; v5 viewer does not move", "dev",
            1.0, vsel["max_translation_m"], None)
        add("4 inside view", "pixels never seen from the centre at 0.5 m (v4 depth mesh) vs 0 (v5 panorama viewer)", "dev",
            vsel["p90_by_t"].get("0.5", vsel["p90_by_t"].get(0.5)), 0.0, False)
    vb = TAB / "v5_viewer_browser.csv"
    if vb.exists():
        b = pd.read_csv(vb)
        add("4 inside view", "browser: opening view vs hero crop NCC (opposite direction in v4 column)", "demo, headed Chromium",
            b["ncc_opposite"].mean(), b["ncc_hero"].mean(), True)
    dl = TAB / "v5_doll_summary.csv"
    if dl.exists():
        d = pd.read_csv(dl).set_index(["set", "method"])
        dsel = j("v5_doll_decision.json").get("selected")
        for split, lab in (("test", "held-out"), ("demo", "demo")):
            if (split, dsel) not in d.index:
                continue
            for c, name in (("furniture_share", "wall + floor textures: residual furniture share"), ("brisque", "wall + floor textures: BRISQUE"),
                            ("noise", "wall + floor textures: flat-area noise"), ("floor_furniture_share", "floor texture: residual furniture share")):
                add("5 dollhouse", name + f" (v5: {dsel})", lab, d.loc[(split, "v4"), c], d.loc[(split, dsel), c], False)
    qk = TAB / "v5_quick_summary.csv"
    if qk.exists():
        q = pd.read_csv(qk).set_index(["set", "config"])
        qsel = j("v5_quick_decision.json").get("selected")
        for split, lab in (("test", "held-out"), ("demo", "demo")):
            for c, name, hb in (("sharpness", "Quick image sharpness (1024 view)", True), ("brisque", "Quick image BRISQUE", False),
                                ("clip", "Quick image CLIP", True), ("kid_x1e3", "Quick image KID x1e3", False),
                                ("seconds", "Quick image seconds", False), ("peak_vram_gb", "Quick peak VRAM GB", False)):
                add("6 quick hires", name + f" (v5: {qsel})", lab, q.loc[(split, "base512"), c], q.loc[(split, qsel), c], hb)
    at = TAB / "v5_app_test.csv"
    if at.exists():
        a = pd.read_csv(at).set_index("mode")
        for c, name in (("seconds", "app request seconds"), ("peak_vram_gb", "app peak VRAM GB"), ("peak_ram_gb", "app peak RAM GB")):
            if "v4" in a.index and "v5" in a.index:
                add("final app", name + " (360° v4 k=3 vs v5 k=2)", "dining room seed 44", a.loc["v4", c], a.loc["v5", c], False)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "v5_summary.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
