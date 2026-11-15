"""PLAN_V6 final: one table of every metric (results/tables/v6_summary.csv), v5 vs v6 where both exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT))


def main():
    rows = []

    def add(ms, metric, data, v5, v6, better="", note=""):
        rows.append({"milestone": ms, "metric": metric, "data": data, "v5": v5, "v6": v6, "better": better, "note": note})

    rules = pd.read_csv(TAB / "v6_scene_rules.csv")
    add("1 scene", "layout rule violations (overlap, outside, wall alignment, chairs facing, door swing, tall pieces "
        "at openings, access)", "1400 layouts", "", int(rules.total.sum()), "v6")
    add("1 scene", "required pieces missing", "1400 layouts", "", int(rules.missing.sum()), "v6")
    cov = pd.read_csv(TAB / "v6_scene_coverage.csv")
    add("1 scene", "asset coverage (classes with >= 1 CC0 model)", "23 classes", "", f"{int(cov.covered.sum())}/{len(cov)}")
    sr = pd.read_csv(TAB / "v6_scene_rooms.csv")
    add("1 scene", "equirect 2048x1024 ray cast s / perspective 768x576 s", "26 rooms", "",
        f"{sr.equirect_s.mean():.2f} / {sr.persp_s.mean():.3f}")
    p5 = pd.read_csv(TAB / "v5_pano_summary.csv")
    p5r = p5[p5.config == "xl_redmond"].set_index("set")
    p6 = pd.read_csv(TAB / "v6_pano_summary.csv").set_index("set")
    v5d = pd.read_csv(TAB / "v6_pano_v5demo_metrics.csv")
    for s in ("dev", "test", "demo"):
        add("2 panorama", "KID x1e3 vs MMIS hold-out (8 level views)", s, round(p5r.loc[s, "kid_x1e3"], 1),
            round(p6.loc[s, "kid_x1e3"], 1), "lower")
        add("2 panorama", "CLIP", s, round(p5r.loc[s, "clip"], 2), round(p6.loc[s, "clip"], 2), "higher")
        add("2 panorama", "depth delta1 vs proxy / AbsRel", s, "", f"{p6.loc[s, 'depth_d1']:.3f} / {p6.loc[s, 'depth_absrel']:.3f}")
        add("2 panorama", "furniture present where the proxy puts it", s, "", round(p6.loc[s, "inst_present"], 3))
        add("2 panorama", "window share generated / proxy", s, "",
            f"{p6.loc[s, 'window_share_gen']:.3f} / {p6.loc[s, 'window_share_proxy']:.3f}")
    add("2 panorama", "straight-line share (proxy-free, same code)", "demo", round(v5d.straight_share.mean(), 3),
        round(p6.loc["demo", "straight_share"], 3), "higher")
    add("2 panorama", "BRISQUE of views (same code)", "demo", round(v5d.brisque.mean(), 1), round(p6.loc["demo", "brisque"], 1), "lower")
    add("2 panorama", "seconds / panorama, VRAM GB, process RAM GB", "dev", "54.8, 5.9, 10.0",
        f"{p6.loc['dev', 'seconds']:.1f}, {p6.loc['dev', 'peak_vram_gb']:.1f}, {p6.loc['dev', 'peak_ram_gb']:.1f}")
    sh = pd.read_csv(TAB / "v6_shots_summary.csv")
    for c in sh.config:
        r = sh[sh.config == c].iloc[0]
        add("3 shots", f"{c}: dLab / LPIPS / depth AbsRel / edge recall / BRISQUE / CLIP / KID", "dev 60 shots", "",
            f"{r.dlab:.2f} / {r.lpips:.3f} / {r.depth_absrel:.3f} / {r.edge_recall:.2f} / {r.brisque:.1f} / "
            f"{r['clip']:.2f} / {r.kid_x1e3:.1f}", note="selected" if c == json.loads((TAB / 'v6_shots_decision.json').read_text())["selected"] else "")
    add("3 shots", "shots valid (main group 15-50 %, near < 15 %, floor < 35 %)", "6 demo rooms", "v5: 0.23 of all shots",
        "36/36", "v6")
    d6 = json.loads((TAB / "v6_doll_summary.json").read_text())
    v5doll = pd.read_csv(TAB / "v5_doll_summary.csv")
    v5demo = v5doll[(v5doll.set == "demo") & (v5doll.method == "lama_bilateral")].iloc[0]
    add("4 dollhouse", "furniture pieces in the dollhouse = layout", "6 demo rooms", "bedroom: 2 blocks",
        f"{d6['pieces_match_rooms']}/{d6['rooms']} rooms", "v6")
    add("4 dollhouse", "residual furniture share of wall / floor textures", "demo", round(v5demo.furniture_share, 4),
        round(d6["furniture_share_v6"], 4), "lower", "v6 keeps wall art / curtains of the panorama, counted here")
    add("4 dollhouse", "flat-area noise of wall / floor textures", "demo", round(v5demo.noise, 3), round(d6["noise_v6"], 3),
        "lower", "v6 textures are not bilateral-smoothed (real stone / panelling texture)")
    add("4 dollhouse", "BRISQUE of wall / floor textures", "demo", round(v5demo.brisque, 1), round(d6["brisque_v6"], 1), "lower")
    add("4 plan", "plan / shot agreement (pieces in a shot's cone that the shot shows)", "6 demo rooms, 152 pairs", "",
        round(d6["plan_shot_agreement"], 3))
    wp = json.loads((ROOT / "outputs" / "window_presence_before.json").read_text()) if (ROOT / "outputs" / "window_presence_before.json").exists() else {}
    add("review", "window presence before repair (coastal / bathroom / contemporary)", "demo",
        "", " / ".join(str(wp.get(k)) for k in ("a_coastal_style_bedroom_s22", "an_art_deco_style_bathroom_s55",
                                                  "a_contemporary_style_living_room_s11")), note="after repair 1.0 / 1.0 / 1.0")
    app = pd.read_csv(TAB / "v6_app_test.csv")
    for r in app.itertuples():
        add("5 app", f"{r.mode} request s, VRAM GB, RAM GB ({r.prompt}, fresh process)", "app test",
            "212 s, 6.2, 10.6 (v5 PLAN_V5)" if r.mode == "v5" else "",
            f"{r.seconds:.0f} s, {r.peak_vram_gb:.1f}, {r.peak_ram_gb:.1f}; hub {r.hub_lines}, truncation "
            f"{r.truncation_warnings}, MISSING {r.missing_weight_reports}, pipeline loads {r.pipeline_loads}")
    pd.DataFrame(rows).to_csv(TAB / "v6_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
