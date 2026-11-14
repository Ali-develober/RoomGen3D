"""PLAN_V3 final: one summary figure per demo room and one before / after table of all milestones.

Figure per room (results/figures/v3_summary_<room>.png) and an overview (v3_summary_overview.png):
    panorama | v2 dollhouse | v3 dollhouse | v2 floor plan | v3 floor plan
The dollhouse images are the real-browser screenshots of scripts/viewer_check.py (azimuth 30 deg).
Table results/tables/v3_summary.csv: the held-out before / after numbers, read from the milestone tables.
Usage: python scripts/v3_summary.py   (after the milestone scripts)
"""

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
from src import pipeline as pl  # noqa: E402

T, F = ROOT / "results/tables", ROOT / "results/figures"


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def row(milestone, metric, before, after, better, note=""):
    return {"milestone": milestone, "metric": metric, "v2 (before)": before, "v3 (after)": after,
            "better": better, "note": note}


def table() -> pd.DataFrame:
    r = []
    s = pd.read_csv(T / "inst_summary.csv").set_index(["split", "method"])
    for m, c, b in (("instance F1 (any class, IoU>=0.5)", "f1_any", "higher"), ("instance precision", "precision_any", "higher"),
                    ("instance recall", "recall_any", "higher"), ("mIoU wall/floor/ceiling/furniture", "miou", "higher"),
                    ("furniture IoU", "iou_furniture", "higher")):
        r.append(row("1 instances", m, s.loc[("test", "v2"), c], s.loc[("test", "v3"), c], b))
    s = pd.read_csv(T / "box_summary.csv").set_index(["split", "method"])
    for m, c, b in (("sides detected (of 4)", "sides_detected", "higher"), ("corner error (m)", "corner_err_m", "lower"),
                    ("area |error|", "area_abs_rel_err", "lower"), ("area |error|, camera-height scaled", "area_abs_rel_err_scaled", "lower"),
                    ("wall planarity RMS (cm)", "planarity_rms_cm", "lower"), ("surface outside the true room (scaled)", "outside_share_scaled", "lower"),
                    ("flat pixels, dollhouse renders", "flat_dollhouse", "lower")):
        r.append(row("2 room box", m, s.loc[("test", "v2"), c], s.loc[("test", "v3"), c], b))
    s = pd.read_csv(T / "furn_summary.csv").set_index(["split", "method"])
    for m, c, b in (("furniture F1 (BEV IoU>=0.25)", "f1_25", "higher"), ("furniture precision", "precision25", "higher"),
                    ("furniture recall", "recall25", "higher"), ("centre error (m, median)", "centre_err_m", "lower"),
                    ("long-side size error (median)", "size_err_long", "lower"), ("orientation error (deg, median)", "orient_err_deg", "lower")):
        r.append(row("3 furniture", m, s.loc[("test", "v2"), c], s.loc[("test", "v3"), c], b))
    s = pd.read_csv(T / "plan_quality.csv").set_index("method")
    for m, c, b in (("walls off the page axes (deg)", "rotation_deg", "lower"), ("inferred (hatched) sides", "inferred_sides", "lower"),
                    ("overlapping furniture pairs per plan", "furniture_overlaps", "lower"),
                    ("furniture area outside the walls", "outside_share", "lower")):
        r.append(row("4 floor plan", m, s.loc["v2", c], s.loc["v3", c], b))
    s = pd.read_csv(T / "plan_quality.csv").set_index("method")
    r.append(row("4 floor plan", "door/window recall (v3 only)", np.nan, s.loc["v3", "openings_recall"], "higher",
                 "not working on the bench"))
    s = pd.read_csv(T / "oneroom_selection.csv")
    t = s[(s["split"] == "test")]
    a = t[(t["template"] == "A") & (t["k"] == 1)].iloc[0]
    b = t[(t["template"] == "B") & (t["k"] == 4)].iloc[0]
    for m, c in (("boxiness (current template, 1 candidate -> one-room template, best of 4)", "score"),
                 ("walls measured (sides / 4)", "sides"), ("CLIP", "clip")):
        r.append(row("5 one room", m, a[c], b[c], "higher", "SD 2.1 panoramas, 10 held-out prompts x 4 seeds"))
    s = pd.read_csv(T / "viewer_gaps.csv").set_index("version")
    r.append(row("6 viewers", "inside view showing no surface (browser)", s.loc["v2", "mean"], s.loc["v3", "mean"], "lower",
                 "6 demo rooms x 4 directions, headed Chromium"))
    if (T / "lq_summary.csv").exists():
        s = pd.read_csv(T / "lq_summary.csv")
        sel = json.loads((T / "lq_decision.json").read_text())["selected"]
        t = s[s["split"] == "test"].set_index("config")
        for m, c, b in (("layout-first depth AbsRel vs layout", "absrel_raw", "lower"), ("layout-first furniture IoU", "iou_furniture", "higher"),
                        ("layout-first CLIP", "clip", "higher"), ("layout-first KID x1e3 vs MMIS", "kid_x1e3", "lower")):
            r.append(row("7 layout-first", m, t.loc["m6", c], t.loc[sel, c], b, f"m6 boxes -> {sel}"))
    df = pd.DataFrame(r)
    imp = np.where(df["better"] == "higher", df["v3 (after)"] > df["v2 (before)"], df["v3 (after)"] < df["v2 (before)"])
    df["improved"] = pd.Series(imp, dtype=object).where(df["v2 (before)"].notna(), "n/a")
    return df


def figures():
    rows = []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        vc = ROOT / "outputs/viewer_check"
        s0 = pl.slug(prompt, 0)
        ims = [rgb(d / "pano.png"), rgb(vc / f"{s0}_v2_dollhouse_30.png"), rgb(vc / f"{s0}_v3_dollhouse_30.png"),
               rgb(d / "floorplan.png"), rgb(d / "v3" / "floorplan_v3.png")]
        titles = ["Panorama", "v2 dollhouse", "v3 dollhouse", "v2 floor plan", "v3 floor plan"]
        fig, axes = plt.subplots(1, 5, figsize=(34, 6.2), dpi=70, gridspec_kw={"width_ratios": [2, 1.33, 1.33, 1.2, 1.2]})
        for a, im, t in zip(axes, ims, titles):
            a.imshow(im)
            a.set_title(t, fontsize=16)
            a.axis("off")
        fig.suptitle(prompt, fontsize=18)
        fig.tight_layout()
        name = pl.slug(prompt, seed)
        fig.savefig(F / f"v3_summary_{name}.png", facecolor="white")
        plt.close(fig)
        rows.append((prompt, ims))
    fig, axes = plt.subplots(len(rows), 5, figsize=(34, 6.0 * len(rows)), dpi=50,
                             gridspec_kw={"width_ratios": [2, 1.33, 1.33, 1.2, 1.2]})
    for i, (prompt, ims) in enumerate(rows):
        for j, im in enumerate(ims):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(["Panorama", "v2 dollhouse", "v3 dollhouse", "v2 floor plan", "v3 floor plan"][j], fontsize=18)
        axes[i, 0].text(0, -0.04, prompt, transform=axes[i, 0].transAxes, fontsize=15)
    fig.tight_layout()
    fig.savefig(F / "v3_summary_overview.png", facecolor="white")
    plt.close(fig)


def main():
    df = table()
    df.to_csv(T / "v3_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df.round(3).to_string(index=False))
    figures()


if __name__ == "__main__":
    main()
