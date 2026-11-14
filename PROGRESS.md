# PROGRESS — RoomGen3D v2 (PLAN_V2.md)

Read this first after a restart. Each entry: what was built, results, what failed, what is next.

## Downloads (budget 12 GB for the whole plan)

| Date | What | Size |
|---|---|---|
| 2026-11-14 | `ProGamerGov/360-Diffusion-LoRA-sd-v1-5` (360Diffusion_v1.safetensors) → checkpoints/pano_lora | 0.15 GB |
| 2026-11-14 | Real-ESRGAN x2plus (from `archerfmy0831/sd-t2i-360panoimage`, single file) → checkpoints/realesrgan | 0.07 GB |
| 2026-11-14 | DAP code + weights (model.pth) from `Insta360-Research/DAP-weights` → checkpoints/dap | 1.46 GB |
| 2026-11-14 | SD 2.1-base panorama model (`archerfmy0831/sd-t2i-360panoimage`, sd-base: unet, text encoder, vae, tokenizer; no safety checker, no SR parts) → checkpoints/pano_sd_base | 5.16 GB |

**Total so far: 6.84 GB of 12 GB.** pip (venv only): einops 0.8.2, termcolor 3.3.0 (needed by the DAP code).

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
