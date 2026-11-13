# PROGRESS — RoomGen3D v2 (PLAN_V2.md)

Read this first after a restart. Each entry: what was built, results, what failed, what is next.

## Downloads (budget 12 GB for the whole plan)

| Date | What | Size |
|---|---|---|
| — | nothing downloaded yet | 0 GB |

**Total so far: 0 GB.**

---

## 2026-11-14 — Milestone 1: floor plan from depth — DONE

**Built**
- `src/floorplan.py`: gravity alignment from a RANSAC floor plane (fallbacks: ceiling, upright camera) → orthographic top-down projection (2 cm grid) → walls (occupancy + morphological closing/opening, sequential RANSAC lines with range-relative tolerance, free-space check, duplicate-line removal, Manhattan snapping, collinear merging, corner snapping) → furniture instances (connected components, robust point cleaning, minimum-area rectangle, class-prior completion of hidden backs, merging of split instances) → doors/windows projected onto walls → architect-style drawing (walls, door swings, windows, labelled furniture, "≈" dimension lines, scale bar, camera + FOV, floor area) to PNG + SVG. Plan comparison (wall matching by viewing direction, furniture matching by image instance).
- `src/pipeline.py`: `analyse` now also returns the ADE20K class map; new stage [7] `build_floorplan`; `depth_gray`; `save_outputs` writes `floorplan.png/.svg`.
- `app/app.py`: main outputs are now 2D image + floor plan + 3D viewer; depth (grayscale), edges and segmentation moved into a collapsed "Analysis" accordion; floor-plan SVG download; plan summary in the info box. Tested headless (one full request + `build()`).
- `scripts/eval_floorplan.py` (resumable: segmentation and depth cached in `outputs/floorplan_eval/`), `scripts/floorplan_demo.py`; `scripts/run_pipeline.py` lists the new stage.

**Results** (654 NYU test images; reference = plan from GT Kinect depth, same segmentation and intrinsics)

| Plan from | Wall recall / precision | Wall length rel. err (median) | Length ratio pred/GT | Furniture IoU | Centroid err (median) |
|---|---|---|---|---|---|
| Predicted metric depth (as in the app) | 0.54 / 0.57 | 23% | 1.16 | 0.25 | 0.47 m |
| Median-scaled prediction (diagnostic) | 0.53 / 0.60 | 12% | 0.99 | 0.51 | 0.17 m |

Gravity from predicted depth within 1.8° (median) of GT; floor plane found in ~85% of views; 0.18 s per plan. Tables `results/tables/floorplan_nyu*.csv`, `floorplan_demo.csv`; figures `results/figures/floorplan_nyu_examples.png`, `floorplan_demo.png`. README section 8 "Floor Plan".

**Failed / limitations**
- No single view ever closes the room (0/654 NYU, 0/6 demo), so "floor area" is only the visible floor area there. Closed rooms are expected from milestone 5 (360°).
- Global scale is the main error (walls +16% median); wall agreement is only ~55%; chairs around a table are often merged into the table segment by SegFormer.
- Generated rooms come out tall (ceilings 3.7–5.2 m): scale overestimate or SD bias, cannot be told apart.

**Next:** Milestone 2 — 360° panorama generation (`src/generate_pano.py`): download the panorama LoRA (`ProGamerGov/360-Diffusion-LoRA-sd-v1-5`), circular padding, LoRA weight sweep, seam / CLIP / FID metrics on 30 prompts, super-resolution comparison.
