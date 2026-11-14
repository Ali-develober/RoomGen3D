# PROGRESS — RoomGen3D v2 (PLAN_V2.md)

Read this first after a restart. Each entry: what was built, results, what failed, what is next.

## Downloads (budget 12 GB for the whole plan)

| Date | What | Size |
|---|---|---|
| 2026-11-14 | `ProGamerGov/360-Diffusion-LoRA-sd-v1-5` (360Diffusion_v1.safetensors) → checkpoints/pano_lora | 0.15 GB |
| 2026-11-14 | Real-ESRGAN x2plus (from `archerfmy0831/sd-t2i-360panoimage`, single file) → checkpoints/realesrgan | 0.07 GB |
| 2026-11-14 | DAP code + weights (model.pth) from `Insta360-Research/DAP-weights` → checkpoints/dap | 1.46 GB |
| 2026-11-14 | SD 2.1-base panorama model (`archerfmy0831/sd-t2i-360panoimage`, sd-base: unet, text encoder, vae, tokenizer; no safety checker, no SR parts) → checkpoints/pano_sd_base | 5.16 GB |

| 2026-11-14 | ControlNet depth SD 1.5 (`lllyasviel/control_v11f1p_sd15_depth`, config + fp16 safetensors) → checkpoints/controlnet_depth | 0.72 GB |

**Total: 7.56 GB of 12 GB.** (Local conversion, not a download: fp16 safetensors copy of the SD 2.1 panorama model in checkpoints/pano_sd_base/sd-base-fp16, 2.6 GB.) pip (venv only): einops 0.8.2, termcolor 3.3.0 (needed by the DAP code).

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

---

## 2026-11-14 — Milestone 2: 360° panorama generation — DONE

**Built**
- `src/equirect.py`: equirectangular geometry (pixel rays, perspective crops e2p, rotations), seam ratio, Laplacian variance.
- `src/generate_pano.py`: SD v1.5 + pano LoRA + MMIS LoRA as unfused adapters; alternative SD 2.1-base panorama model loader; circular x-padding patched into every padded Conv2d of UNet + VAE (optional start step); Real-ESRGAN x2 (own RRDBNet, tiled, wrap-padded); bicubic + unsharp baseline (wrap-aware).
- `scripts/eval_pano.py` (resumable; panoramas in outputs/pano_eval/<config>/).

**Results** (60 panoramas per config = 5 rooms × 6 styles × 2 seeds; crops vs. 500 MMIS hold-out)
- Circular padding: seam ratio median 3.60 → 0.83 (1 = invisible), no realism cost.
- MMIS LoRA weight 0 / 0.5 / 0.8 / 1.0 with the pano LoRA: KID 63.7 / 64.1 / 64.6 / 66.3 (std ~3), CLIP 20.4 → 19.8: **no benefit (negative result)**.
- Alternative SD 2.1 panorama model: KID 38.2, FID 93, CLIP 22.4, 7.3 s — **selected by the fixed rule** (mean rank of KID and CLIP). It was added to the candidates after the six LoRA configs ran but before it ran. Consequence: the 360° mode does not use our MMIS LoRA.
- Super-resolution (30 panos): Real-ESRGAN x2 Laplacian variance 315 vs. 97 for bicubic+unsharp, 0.76 s; less faithful (PSNR downscaled 33.8 vs. 51.6 dB). Default: Real-ESRGAN.
- Tables `results/tables/pano_configs.csv`, `pano_per_image.csv`, `pano_sr*.csv`; figures `pano_configs_grid.png`, `pano_configs_metrics.png`, `pano_sr_compare.png`. README section 8 "360° Panorama Generation".

**Failed / limitations**: pano-LoRA rooms repeat objects and have bulging walls; our LoRA does not transfer to panoramas; Real-ESRGAN smooths fine texture.

**Next:** Milestone 3 (done below).

---

## 2026-11-14 — Milestone 3: panoramic depth — DONE

**Built**
- `src/pano_depth.py`: method A (14 overlapping 100° views → Depth Anything V2 Metric Indoor → z to radial distance → resampling to equirect with feather weights → robust log-space least-squares per-view scale, optional shift step with scales fixed → feathered blend); method B DAP loader (fp32; fp16 gives NaN) + `dap_depth_wrap` (DAP on the panorama and on it rolled by half, cross-faded cos²/sin²); panorama segmentation via the 14 views; wrap-aware gated denoising and depth refinement.
- `scripts/eval_pano_depth.py` (30 panoramas of the selected generator).

**Results** (medians): method A alignment 33% → 10% view disagreement, floor inliers 43% → 62%, but rooms 6.1 m tall (0% plausible), distances 2× DAP's. DAP: floor inliers 95%, tilt 0.4°, room height 2.88 m (100% plausible), wall inliers 28% vs 17%, 0.4 s, 2.3 GB. DAP seam ratio 9.75 → 0.62 with the wrap fix (0.9 s). Gated denoise + refinement: no geometry gain (floor RMS 2.7% → 3.1%). **Selected for 3D: DAP + wrap fix, no refinement.** Tables `results/tables/pano_depth*.csv`, figure `pano_depth_examples.png`, README section 8 "Panoramic Depth".

**Failed / limitations**: no real panorama ground truth (official sets need signed licences; HF mirrors are third-party redistributions, not used) → only plausibility checks; walls only ~28% planar (curved generated walls, SegFormer labels cupboards as wall); method A's absolute scale is wrong.

**Next:** Milestone 4 — full-room 3D (`src/room_mesh.py`): spherical back-projection of the DAP distance map, textured mesh with discontinuity cuts, seam stitching and poles, RANSAC room box, dollhouse variant, two-mode viewer, empty-share metrics vs. the single-view mesh.

---

## 2026-11-14 — Milestone 4: full-room 3D — DONE

**Built**
- `src/room_mesh.py`: gravity rotation from the floor plane; spherical back-projection on a wrap-around grid (duplicate column with u = 1) + pole caps; layer 0 with sharpened depth edges and cuts, layer 1 disocclusion band, layer 2 room box from the closed 360° plan, all 2D filters wrap-padded; dollhouse (ceiling −25 cm removed); Open3D ray caster with texture lookup.
- `src/pipeline_360.py` (stages), `scripts/run_room360.py` (6 demo prompts, resumable, per-room timing.json), `scripts/eval_room3d.py`.
- `src/floorplan.py`: accepts a point cloud (spherical back-projection), wrap-aware azimuth intervals, 360° camera marker, `closed_room_360` (sides without a wall line inferred from the 98th percentile of the scene, drawn hatched), `room_edges` (closed sides become the walls).
- `src/generate_pano.py`: fp16 safetensors cache of the SD 2.1 panorama model (load 70 s → 17 s).
- `app/app.py`: mode switch Quick / 360° Room; inside viewer (orbit radius 5 cm–1 m around the centre), dollhouse viewer, Model3D fallback; only the active mode's diffusion models on the GPU. Headless test OK (quick → 360 → quick), 360° peak 5.5 GB.

**Results** (6 demo prompts; empty share of ray-cast views; final numbers after the milestone-5 room rule): inside at the centre 0.4% (single view 89.7%), 1.0 m off-centre 0.7% (88.6%); dollhouse 27.6% (57.2%). 16.8 s per room. Tables `room3d_empty*.csv`, `room360_timing.csv`; figures `room3d_views.png`, `room3d_dollhouse.png`.

**Failed / limitations**: repeated furniture in some panoramas; thin objects torn; rectangular box (dining room box 72 m²); `<model-viewer>` inside/dollhouse modes not yet checked by hand in a browser.

**Next:** Milestone 5 (below).

---

## 2026-11-14 — Milestone 5: floor plan of the full room — DONE

**Built**
- `src/floorplan.py`: `closed_room_360` with three rules ("walls", "floor", "walls+floor"); default "walls+floor" (walls rule, but no side may cut through the observed floor: pushed out to the floor's 95th percentile); inferred sides hatched.
- `scripts/floorplan_360.py`: plans (PNG/SVG) for the 6 demo rooms, plan-vs-mesh floor check (convex hull of the mesh's observed floor triangles ÷ plan area), independent wall agreement (share of wall points within 30 cm of the outline), one figure per room (plan next to the dollhouse from above) + overview.
- Room meshes rebuilt and milestone 4 evaluation rerun with the final rule (numbers updated above and in the README).

**Results**: 6/6 plans closed (single view: 0/654 NYU, 0/6 demo). Rule comparison on the same 6 rooms (selection, not a test): wall agreement 0.53 (walls) / 0.46 (floor) / **0.55 (walls+floor)**; hull ÷ plan up to 2.38 with the walls rule (impossible), ≤ 1.08 with walls+floor. Bedroom 23.7 m² (wall agreement 0.99) and bathroom 14.7 m² (0.84) are good; open-plan dining (72 m²) and living room (45 m²) are boxed too large. Table `results/tables/floorplan_360.csv`; figures `floorplan_360_overview.png`, `floorplan_360_<room>.png`.

**Failed / limitations**: only 0–2 sides per room backed by a detected wall line (wardrobes / shelving not labelled as wall); rectangles only; the rule was chosen on the same 6 rooms it is reported on.

**Next:** Milestone 6 (below).

---

## 2026-11-14 — Milestone 6: layout first (research experiment) — DONE

**Built**
- `src/layout.py`: seeded parametric layouts for bedroom / living room / kitchen / dining room / bathroom (size ranges, door + clearance, windows, furniture boxes by placement rules, no overlaps), camera at the free spot nearest the centre (1.4 m), exact equirectangular distance + ADE20K label rendering, ControlNet condition image, layout → floor-plan dict (drawn with floorplan.py), JSON.
- `src/generate_pano.py`: `load_controlnet_pipeline` (SD 1.5 + ControlNet depth fp16 + pano LoRA), `generate_pano_controlled` (circular padding in UNet, ControlNet and VAE; scale 0 = baseline).
- `scripts/eval_layout.py`: 12 rooms (6 demo prompts × 2 layouts); adherence (DAP depth vs. layout, SegFormer vs. layout labels), ground-truth check of the 360° plan pipeline, exact-geometry mesh vs. milestone 4, CLIP.
- `app/app.py`: third mode "Layout-first" (layout from the room type in the prompt, seed = layout); generalized model parking (only the active mode's large models on the GPU). Headless test of all three modes with switching: peaks Quick 2.9 / Layout-first 4.6 / 360° 5.5 GB.

**Results**: conditioned panoramas follow the layout: depth AbsRel 0.062 as predicted (0.227 without ControlNet), δ<1.25 0.99, mIoU 0.49 (0.28), furniture IoU 0.45 (0.07). 360° plan pipeline vs. true layout: area error median 8%, mean |err| 12%, sides 6%; DAP scale correct (ratio 0.98, camera 1.47 m vs. 1.40 m). Exact-geometry mesh: empty 0.02% inside, 5.9% dollhouse (milestone 4: 0.4% / 27.6%). Cost: CLIP 21.2 → 18.6, box-shaped furniture, plain styles. Kept as a separate mode; 360° stays the default.

**Failed / limitations**: image quality and style; bathrooms (area error 26–42%); SD 1.5 base (no SD 2.1 depth ControlNet tried, to stay within the plan).

---

## 2026-11-14 — Final — DONE

- App modes that work: Quick (single view), 360° Room, Layout-first (`app/app.py`).
- README: new pipeline diagram (three modes), sections for milestones 1–6 with all tables and figures, objectives 6–8 added with status, limitations updated.
- Everything committed and pushed. Downloads 7.56 GB of the 12 GB budget.
- Not done (on purpose / open): the final course notebook (waits for the owner's review, per the plan); the `<model-viewer>` inside / dollhouse modes were tested headlessly but not checked by hand in a browser; the app is not deployed. The older untracked `*_smoke` result files and the modified `ARTI308_Demo.ipynb` in the working tree were left untouched (not mine to commit).

---

# PLAN_V3 work log

## 2026-11-14 — Disk cleanup (before PLAN_V3) — DONE

Free space on C: **22.3 GB before → 33.5 GB after** (+11.2 GB). Sizes measured with a Python `os.scandir` walk (junctions not followed).

**Before**

| Location | Size | Notes |
|---|---|---|
| `checkpoints/` | 10.27 GB | pano_sd_base 7.75 (fp32 `sd-base` 5.16 + fp16 `sd-base-fp16` 2.58), dap 1.46, controlnet_depth 0.72, pano_lora 0.15, lora_mmis 0.11 (6 steps), realesrgan 0.07, lora_smoke 0.02 |
| `.venv/` | 5.96 GB | kept |
| `outputs/` | 3.28 GB | pano_eval 1.06, eval_2d 0.82, floorplan_eval 0.41, layout 0.30, app 0.20, pipeline 0.17, pano_depth 0.16, room360 0.14, depth_eval/gated/isometric/smoke 0.04 |
| `data/` | 2.87 GB | nyu/train 1.33, nyu/test 1.24, mmis 0.30 (images 0.10, meta json 0.19) |
| `results/` | 0.10 GB | kept |
| `.git/` | 0.09 GB | kept |
| HF cache `~/.cache/huggingface/hub` | 7.27 GB | SD 1.5 2.13, CLIP-L/14 1.71, DPT-hybrid MiDaS 0.98, DA-V2 metric indoor 0.39, Depth Anything base 0.39, SegFormer-b2 0.22, **not this project:** arabertv02 0.54, timm convnext_base 0.35, vit_base 0.35, convnext_tiny 0.11, vit_small 0.09 (DAP and SD-2.1-pano entries are empty refs; files live in `checkpoints/`) |
| pip cache `%LOCALAPPDATA%/pip/cache` | 3.85 GB | |
| torch hub `~/.cache/torch/hub` | 0.10 GB | Inception weights for FID, kept |
| Temp `%TEMP%/claude/<project>` | 0.20 GB | 2 old session scratchpads + current |

**Deleted**
- pip cache (`pip cache purge`): 3.85 GB.
- Duplicate model: `checkpoints/pano_sd_base/sd-base` (fp32 .bin, 5.16 GB). The loader uses the fp16 safetensors copy `sd-base-fp16` first; verified it still loads. (If the fp16 copy is ever deleted, the fallback needs a re-download, 5.2 GB.)
- Regenerable eval dumps whose tables/figures are saved: `outputs/pano_eval`, `eval_2d`, `floorplan_eval`, `app`, `pano_depth`, `depth_eval(_smoke)`, `depth_gated(_smoke)`, `isometric`, `isometric_grid.png`, `mesh_compare_smoke.*`, `test_sd.png`, smoke logs.
- Intermediate LoRA checkpoints step 500–2500 and `checkpoints/lora_smoke` (final `lora_mmis/step_03000` kept).
- `__pycache__` in app/, scripts/, src/, checkpoints/dap/; two old session scratchpads.
- No incomplete downloads were found in the HF cache; no raw archives were left (`data/nyu/raw` holds only `splits.mat`, kept).

**Kept on purpose**: `outputs/room360`, `outputs/pipeline` (referenced in the README; v2 rooms for the v3 comparison), `outputs/layout` (layout-first bench), all of `results/` — including the untracked `results/*_smoke*` tables/figures, because results/tables and results/figures are on the never-delete list (they total < 5 MB).

**For the owner to decide** (outside this project's scope, not deleted)
- HF cache models from other projects (dated before this project started on 2026-11-10): `FatimahEmadEldin/EXP-001_arabertv02_wojood_flat_baseline` 0.54 GB, `timm/convnext_base.fb_in22k_ft_in1k` 0.35 GB, `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` 0.35 GB, `timm/convnext_tiny.fb_in22k_ft_in1k` 0.11 GB, `timm/vit_small_patch16_224.augreg_in21k_ft_in1k` 0.09 GB — 1.44 GB total. Remove with `huggingface-cli delete-cache` if no longer needed.
- `Intel/dpt-hybrid-midas` (0.98 GB) is only used by the old depth-model comparison (`scripts/eval_depth.py`); kept because code still loads it.
- The ~22 GB drop since project start is fully explained by the items above (project 22.5 GB + HF 5.8 GB + pip 3.9 GB); nothing large outside these locations was scanned.

### PLAN_V3 downloads (budget 8 GB)

| Date | What | Size |
|---|---|---|
| 2026-11-14 | `facebook/mask2former-swin-large-ade-panoptic` (config + model.safetensors only) → checkpoints/mask2former_ade_panoptic | 0.87 GB |
| 2026-11-14 | HorizonNet code (git clone, github.com/sunset1995/HorizonNet) + `sunset1995/HorizonNet` resnet50_rnn__mp3d.pth → checkpoints/horizonnet | 0.33 GB |
| 2026-11-14 | `thibaud/controlnet-sd21-depth-diffusers` (config + .bin) → checkpoints/controlnet_sd21_depth | 0.73 GB |
| 2026-11-14 | pip (venv): shapely 2.2.0, playwright 1.65.2 + its Chromium (%LOCALAPPDATA%\ms-playwright) | ~0.3 GB |

**Total PLAN_V3 downloads: ~2.2 GB of 8 GB.**

## 2026-11-14 — Test bench for PLAN_V3 — DONE

- `scripts/make_bench.py`: 40 layout-first rooms (5 types × layout seeds 1–8), dev = seeds 1–4, held-out = seeds 5–8; per room GT distance / classes / furniture instances (`layout.render_full`, new), ControlNet panorama, Real-ESRGAN texture, DAP distance, v2 SegFormer labels in `outputs/bench/<id>/`. 13 s per room. Index `results/tables/bench_v3_rooms.csv`.

## 2026-11-14 — V3 Milestone 1: instance segmentation on panoramas — DONE

**Built**: `src/instances.py` (Mask2Former panoptic on 13 views, projection to the panorama, cross-view union-find merging within furniture families, v2 baseline, hybrid class map), `scripts/eval_instances.py` (dev sweep, held-out report, figure).

**Results (held-out, 109 GT boxes)**: instance F1 (any class) 0.17 → **0.35**, precision 0.14 → 0.44, recall 0.21 → 0.28; same-family F1 0.05 → 0.12; seam objects found 1/12 → 3/12; mIoU 0.585 → 0.589, furniture IoU 0.46 → 0.52 (hybrid class map). Tables `inst_*.csv`, figure `inst_examples.png`, README "Instance segmentation on panoramas".

**Failed / limitations**: the pure panoptic class map labels the up-view ceiling as floor (dev ceiling IoU 0.16) → hybrid map (SegFormer surfaces + panoptic furniture). Recall stays low (0.28) because the generator often does not draw what the layout says; doors/windows ≈ 0 IoU for both. The sweep was flat (all segment scores > 0.7). Swin final layernorm reported "missing" by transformers 5 (unused by Mask2Former's backbone; outputs are confident and correct).

**Next**: Milestone 2 — room box regularization (`src/room_box.py`, `scripts/eval_room_box.py` already drafted and smoke-tested on 2 rooms).

## 2026-11-14 — V3 Milestone 2: room box regularization — DONE

**Built**: `src/room_box.py` (normals, dominant yaw from a mod-90° normal histogram, per-side density-peak wall fit with detected/inferred sides, exact textured shell with openings flattened onto walls, hidden texels inpainted in plane texture space: Telea / Navier–Stokes / own exemplar patch fill, observed furniture surfaces, texture-atlas mesh assembly); `scripts/eval_room_box.py` (v2 vs v3 vs layout GT); `scripts/eval_horizonnet.py` (pretrained HorizonNet, with a numpy PCA shim instead of installing scikit-learn). pip (venv): shapely 2.2.0 (HorizonNet import).

**Results (held-out)**: sides detected 2.5 → 3.85 of 4; corner error 0.33 → 0.23 m; area |err| 12.1% → 10.6%; wall planarity 7.6 → 3.7 cm RMS; surface outside the true room (camera-height scaled) 12.0% → 6.1%; build 3.6 → 0.5 s. Inpainting: NS selected on dev (SSIM 0.720 vs Telea 0.718, patch 0.670); patch fill leaves 0.32 flat texels vs 0.50. HorizonNet comparable with DAP scale (area 9.8% vs 10.6%), worse in shape with true scale (10.4% vs 5.4%).

**Failed / limitations**: v3 dollhouse renders are flatter (11.0% vs 7.6% flat pixels) because hidden regions are filled smoothly (pre-declared SSIM rule picked NS; patch fill would be less flat). Global scale error from DAP remains (camera 1.44–1.48 m vs 1.40 m). Bug found and fixed: histogram smoothing with fewer bins than the kernel.

**Next**: Milestone 3 — solid furniture from the v3 instances (footprint cleaning, extrusion to the floor, class size priors, wall-axis snapping, overlap resolution; footprint IoU / centre / size / orientation errors vs. layout boxes).

## 2026-11-14 — V3 Milestone 3: solid furniture — DONE

**Built**: `src/furniture.py` (point cleaning + percentile trimming, wall-axis snapping, class size priors with clamp/reject, back-to-wall snap, overlap merge/shrink/clip, leftover components, textured extruded solids with donor-face texture for hidden faces and seam-safe UVs); `scripts/eval_furniture.py` (64-config dev sweep, v2 vs v3 vs layout boxes, figure).

**Results (held-out)**: F1@0.25 0.42 → 0.48, precision 0.39 → 0.67, recall 0.47 → 0.38; centre error 0.24 → 0.20 m; size error long side 30% → 25%; orientation 5.3° → 1.4°; height error 5 cm. Priors: 35 clamped, 98 rejected of 194 candidates (after the milestone-4 box rule change).

**Failed / limitations**: recall below v2; without the leftover components (first try) held-out F1 was 0.36 < v2 — fixed by adding unclaimed furniture pixels as candidates (chosen on dev). Boxes only; hidden faces copy the visible side's texture. Bugs fixed: dict removal with numpy arrays in merging, donor-face fallback when nothing is visible.

**Next**: Milestone 4 — floor plan drawing (axis-aligned, symbols, labels without overlap, dimensions, PNG/SVG/DXF).

## 2026-11-14 — V3 Milestone 4: floor plan drawing — DONE

**Built**: `src/plan_v3.py` (openings cast onto box walls; primitive-list drawing: solid/hatched walls, door leaves + swings incl. double doors, windows, 20+ furniture symbols, collision-checked labels with leader lines, room and clearance dimensions, scale bar, area, note; PNG/SVG via matplotlib, DXF R12 own writer); `src/pipeline_v3.py` (analyse → build → save; candidate generation for milestone 5); `scripts/plan_v3_eval.py` (six demo rooms v2 vs v3 figure, held-out plan quality).

**Results (held-out)**: walls off-axis 3.8° → 0°; hatched sides 1.5 → 0.15; overlapping furniture pairs 2.9 → 0.3; furniture outside walls 17.9% → 0.1%; label failures 0. Figure `plan_v2_v3.png`.

**Failed / limitations**: door/window detection on the bench: precision 0.05, recall 0.02 (images rarely show them). Box fit change triggered here (kitchen 2.15 m deep, no furniture): side peak rule "max" → "farthest" (chosen on dev: area err 11.3% → 10.4%, under-size rooms 5% → 0%; held-out worse 10.6% → 12.6%); milestones 2–4 re-evaluated, README/tables updated. Open-plan demo rooms are boxed too large (dining 74.7 m², living 56 m²), Mediterranean living room 8.6 × 2.65 m.

**Next**: Milestone 5 — one-room prompt template + boxiness-based candidate selection (`src/boxiness.py`, `scripts/eval_one_room.py` written), app slider.

## 2026-11-14 — V3 Milestone 5: panoramas that are one room — DONE

**Built**: one-room prompt template + negative (`generate_pano.ONE_ROOM_*`, new `negative` argument), `src/boxiness.py` (wall fit, enclosed share, detected sides), `pipeline_v3.generate_candidates` (best of k), `scripts/eval_one_room.py` (80 SD 2.1 panoramas: 5 rooms × 2 styles × 4 seeds × 2 templates). App: 360° mode runs v3 by default, v2 selectable, "Candidates" slider 1–4, DXF download, top view (milestone 6) — tested headlessly in all modes (v3 k=1 13.7 s, k=2 29.6 s, peak 5.9 GB).

**Results**: template selected on dev (boxiness 0.856 vs 0.844, CLIP equal); held-out 0.830 vs 0.798, but the paired difference is within noise (+0.032 ± 0.033); rooms more compact (median 18 vs 26 m²). Best-of-k (held-out): boxiness 0.83 → 0.89 → 0.92 for k = 1/2/4, four measured walls from k = 2, CLIP 22.1 → 23.4, +9.6 s per extra candidate.

**Failed / limitations**: template effect not significant; the score favours one look (same seed kept in 5 of 6 held-out rows), reducing variety. Intermediate panoramas (outputs/one_room) and app test outputs deleted after the tables/figures were saved.

**Next**: Milestone 6 — viewers (single-sided dollhouse, background colour, top view) + headed browser screenshots (`scripts/viewer_check.py` written; Playwright 1.65 + Chromium installed in the venv / %LOCALAPPDATA%\ms-playwright).

## 2026-11-14 — V3 Milestone 6: viewers — DONE

**Built**: single-sided GLB export (`export_glb(double_sided=False)`), `src/viewers.py` (room mean colour, orthographic top view), v3 inside mesh = shell + observed surfaces in front of it (`pipeline_v3.in_front_of_shell`), dollhouse = shell without ceiling + solids; app: inside background = room colour, neutral dollhouse background, top view next to the plan; `scripts/viewer_check.py` (Playwright headed Chromium + `<model-viewer>` 3.5, local HTTP server, magenta-background gap measurement).

**Results**: headed browser worked. Inside gaps (6 rooms × 4 directions): v2 0.34% mean / 2.0% worst → v3 0.00% / 0.00%. Dollhouse screenshots: v3 closed room, near walls culled automatically. Figures `viewer_inside.png`, `viewer_dollhouse.png`, `plan_vs_topdown.png`.

**Failed / limitations**: first v3 inside mesh used the solids and looked crude from inside → replaced by observed surfaces (solids stay in the dollhouse). Box solids with repeated textures; smooth inpainted floor blotches. Mouse dragging not tested, only scripted camera positions. Playwright's Chromium lives in %LOCALAPPDATA%\ms-playwright (~0.3 GB, outside the project).

**Next**: Milestone 7 — layout-first image quality (shaped proxies, ControlNet strength / step-range sweep, doors/windows, SD 2.1 compatibility, optional img2img pass).

## 2026-11-14 — V3 Milestone 7: layout-first image quality — DONE

**Built**: `layout.proxy_parts` + `render_full(shapes=True)` (bed/sofa/armchair/chair/table proxies, door and window recesses), `generate_pano_controlled(end=, prompt=, negative=)`, `load_controlnet_pipeline_sd21` (SD 2.1 panorama model + SD 2.1 depth ControlNet), `scripts/eval_layout_quality.py` (9 dev configurations, rule-based selection, held-out report, img2img pass). App Layout-first mode switched to the selected configuration (20.8 s, 4.8 GB).

**Results (held-out)**: selected = SD 2.1 + depth ControlNet, shaped proxies, scale 0.75, first 60% of steps. KID 120 → 63 ×10³, FID 192 → 139, CLIP 18.5 → 21.6; depth AbsRel 0.060 → 0.059; furniture IoU 0.43 → 0.47; door / window IoU 0.02 / 0.00 → 0.31 / 0.24. Shapes alone (SD 1.5): AbsRel 0.050, KID 107.

**Failed / limitations**: mIoU of room surfaces 0.47 → 0.42; img2img pass not adopted (dev CLIP 20.6 → 20.5). Intermediate panoramas (outputs/layout_quality, see size above) deleted after tables/figures were saved.

**Next**: Final — summary figure per room, summary table, README / objectives, push.
