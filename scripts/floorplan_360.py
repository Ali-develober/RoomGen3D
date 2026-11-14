"""Milestone 5: floor plan of the full room (360 degree point cloud) for the six demo rooms.

For every room in outputs/room360/ (scripts/run_room360.py):
  plan        closed room rectangle (src/floorplan.closed_room_360: detected walls, missing sides
              inferred from the scene extent), wall lengths, area, doors, windows, furniture
  rules       two ways to close the room (src/floorplan.closed_room_360): "walls" (first version:
              longest wall line per side) and "floor" (default: the floor + furniture extent per
              side, snapped to a wall line within 0.5 m); both are evaluated
  check       plan area vs. the floor in the MESH (observed surface only, so not the room box):
                observed floor   = area of the observed-surface triangles (layer 0) that lie on the
                                   floor (all three vertices within 8 cm of the floor height and a
                                   normal within 15 deg of vertical): what the panorama shows of
                                   the floor, i.e. without the parts under furniture
                floor hull       = area of the convex hull of those floor triangles (plan view)
              ratios hull / plan and observed / plan. The "floor" rule uses the floor extent itself,
              so for it this check is not independent; therefore also:
                wall agreement   = share of the wall-labelled points (0.3-2.0 m above the floor) that
                                   lie within 30 cm of the plan's outline (independent of both rules)
  figure      plan next to a dollhouse render from above, both with forward = up
Outputs: results/tables/floorplan_360.csv, results/figures/floorplan_360_<room>.png (one per room),
         results/figures/floorplan_360_overview.png; outputs/room360/<room>/floorplan.png / .svg
Usage: python scripts/floorplan_360.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as an  # noqa: E402
from src import floorplan as fp  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_360 as p3  # noqa: E402
from src import room_mesh as rm  # noqa: E402


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def mesh_floor(mesh: dict, tol: float = 0.08, max_tilt: float = 15.0) -> tuple[float, float]:
    """(observed floor area, convex-hull area of the observed floor) in m^2, from layer-0 triangles."""
    f = mesh["faces"][mesh["layer"] == 0]
    v, h = mesh["vertices"].astype(np.float64), mesh["heights"]
    on = (np.abs(h[f]) < tol).all(1)
    a, b, c = v[f[on, 0]], v[f[on, 1]], v[f[on, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    flat = np.abs(n[:, 1]) / np.maximum(2 * area, 1e-12) >= np.cos(np.radians(max_tilt))
    obs = float(area[flat].sum())
    xz = v[np.unique(f[on][flat])][:, [0, 2]]
    hull = float(ConvexHull(xz).volume) if len(xz) >= 3 else np.nan
    return obs, hull


def wall_agreement(plan: dict, pts: np.ndarray, class_map: np.ndarray, id2label: dict, tol: float = 0.3) -> float:
    """Share of wall points (0.3-2.0 m high) within `tol` of the room outline (distance to the 4 edges)."""
    m = np.isin(class_map, fp.class_ids(id2label, fp.WALL_CLASSES))
    xy, h = fp.to_plan(pts[m], plan["frame"])
    xy = xy[(h > 0.3) & (h < 2.0)]
    cs = plan["room"]["corners"]
    d = np.full(len(xy), np.inf)
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        t = np.clip(((xy - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (a + t[:, None] * (b - a)), axis=1))
    return float(np.mean(d < tol)) if len(d) else np.nan


def dollhouse_topview(dh: dict, size: int = 640) -> np.ndarray:
    """Dollhouse seen from behind the panorama centre and high above (forward = up in the image)."""
    v = dh["vertices"]
    lo, hi = np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)
    c = (lo + hi) / 2
    r = 1.5 * np.linalg.norm((hi - lo)[[0, 2]])
    el = np.radians(72)
    eye = c + np.array([0.0, -r * np.sin(el), -r * np.cos(el)])
    _, _, col = rm.Raycaster(dh).cast(eye, rm.look_at_dirs(eye, c, 45, size), color=True)
    return col.reshape(size, size, 3)


def main():
    from transformers import AutoConfig
    id2label = AutoConfig.from_pretrained(an.SEG_MODEL).id2label
    rows, overview = [], []
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        res = p3.build(np.load(d / "distance.npy"), rgb(d / "texture.png"),
                       cv2.imread(str(d / "class_map.png"), cv2.IMREAD_UNCHANGED).astype(np.int32), id2label)
        plan, mesh = res["plan"], res["mesh"]
        dist = np.load(d / "distance.npy")
        cm = cv2.imread(str(d / "class_map.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        pts = rm.aligned_points(dist, mesh["R"])
        plan_w = fp.build_floorplan(dist, None, cm, id2label, points=pts, fov_deg=360.0, room_rule="walls")
        plan_wf = fp.build_floorplan(dist, None, cm, id2label, points=pts, fov_deg=360.0, room_rule="walls+floor")
        rwf = plan_wf["room"]
        plan_img = fp.draw_floorplan(plan, d / "floorplan.png", d / "floorplan.svg", title=prompt)
        top = dollhouse_topview(res["dollhouse"])
        obs, hull = mesh_floor(mesh)
        room = plan["room"]
        s = fp.plan_summary(plan)
        walls = [w["length"] for w in plan["walls"]]
        rw = plan_w["room"]
        rows.append({"prompt": prompt, "seed": seed, "walls_rule_area_m2": rw["area"],
                     "walls_rule_hull_over_plan": hull / rw["area"], "walls_rule_observed_over_plan": obs / rw["area"],
                     "walls_rule_wall_agreement": wall_agreement(plan_w, pts, cm, id2label),
                     "walls_rule_inferred_sides": 4 - rw["n_observed_sides"],
                     "wf_rule_area_m2": rwf["area"], "wf_rule_hull_over_plan": hull / rwf["area"],
                     "wf_rule_observed_over_plan": obs / rwf["area"],
                     "wf_rule_wall_agreement": wall_agreement(plan_wf, pts, cm, id2label),
                     "wf_rule_inferred_sides": 4 - rwf["n_observed_sides"],
                     "area_m2": room["area"], "width_m": room["width"],
                     "depth_m": room["depth"], "wall_lengths_m": " / ".join(f"{x:.2f}" for x in walls),
                     "observed_sides": room["n_observed_sides"], "inferred_sides": 4 - room["n_observed_sides"],
                     "detected_wall_lines": len(plan["detected_walls"]), "manhattan_support": plan["manhattan"]["support"],
                     "doors": s["doors"], "windows": s["windows"], "furniture": s["furniture"],
                     "furniture_labels": ", ".join(sorted({f["label"] for f in plan["furniture"]})),
                     "camera_height_m": s["camera_height_m"], "ceiling_height_m": s["ceiling_height_m"],
                     "mesh_floor_observed_m2": obs, "mesh_floor_hull_m2": hull,
                     "hull_over_plan": hull / room["area"], "observed_over_plan": obs / room["area"],
                     "wall_agreement": wall_agreement(plan, pts, cm, id2label)})
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=110, gridspec_kw={"width_ratios": [1, 1]})
        axes[0].imshow(plan_img)
        axes[1].imshow(top)
        axes[1].set_title("Dollhouse (ceiling removed), seen from above and behind the camera", fontsize=10)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{prompt}: 360° floor plan and full-room mesh", fontsize=12)
        fig.tight_layout()
        fig.savefig(ROOT / f"results/figures/floorplan_360_{pl.slug(prompt, seed)}.png", facecolor="white")
        plt.close(fig)
        overview.append((prompt, rgb(d / "pano.png"), plan_img, top))
    fig, axes = plt.subplots(len(overview), 3, figsize=(18, 4.3 * len(overview)), dpi=80,
                             gridspec_kw={"width_ratios": [2, 1, 1]})
    for i, (prompt, pano, pimg, top) in enumerate(overview):
        for j, im in enumerate((pano, pimg, top)):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")
        axes[i, 0].set_title(prompt, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/floorplan_360_overview.png", facecolor="white")
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/tables/floorplan_360.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["furniture_labels", "wall_lengths_m"]).round(2).T.to_string())


if __name__ == "__main__":
    main()
