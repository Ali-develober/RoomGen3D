# RoomGen3D: Text-Driven 2D & 3D Interior Design Generation with Image-Processing Enhancement

**Course:** ARTI 403 – Image Processing | Imam Abdulrahman Bin Faisal University
**Team:** _TBD_ | **Team Leader:** _TBD_

---

## 1. Idea

The user types a description of the room design they want (e.g., *"modern minimalist bedroom with wooden floor and warm lighting"*). The system:

1. Generates a photorealistic **2D design** of the room.
2. Enhances and analyzes the image using **classical image-processing techniques**.
3. Converts it into a **3D model** (point cloud and mesh) that the user can rotate and explore.

**Input:** text prompt → **Output:** 2D image + interactive 3D room model.

---

## 2. Problem

- Visualizing an interior design idea usually requires a designer, 3D software skills, or expensive tools.
- Existing text-to-image tools produce only flat 2D images with no spatial or depth understanding.
- Single-image 3D reconstruction is sensitive to image quality (noise, poor lighting, blur), which degrades depth estimation.

**Goal:** an accessible pipeline that turns a text description into a 2D design and a 3D room, with image processing used to improve quality at every stage.

---

## 3. Pipeline

```
Text Prompt
   │
   ▼
[1] 2D Generation ───── Stable Diffusion v1.5 + custom-trained LoRA (MMIS)
   │
   ▼
[2] Preprocessing ───── Noise-gated denoising (CLAHE / sharpening evaluated and dropped, see section 8)
   │
   ▼
[3] Image Analysis ──── Canny edges · Room segmentation (walls / floor / furniture)
   │
   ▼
[4] Depth Estimation ── Depth Anything V2 Metric Indoor (Depth Anything and MiDaS compared on NYU)
   │
   ▼
[5] Depth Post-processing ── Bilateral / median filtering · edge-aware refinement · hole filling
   │
   ▼
[6] 3D Reconstruction ─ Depth → Point Cloud (Open3D) → Textured mesh with filled occlusions + room planes
   │
   ▼
[7] Floor Plan ──────── Gravity alignment (floor plane) → top-down projection → walls (morphology + RANSAC
   │                    lines + Manhattan snap) · furniture footprints · doors/windows · dimensions (PNG/SVG)
   ▼
[8] Interface ───────── Gradio: 2D image + floor plan + interactive 3D viewer + downloads (.png/.svg/.ply/.glb)
```

This is the **Quick mode** (one perspective image: a 3D relief of what the camera saw). Version 2 (PLAN_V2.md) adds two modes that build the **whole room**:

```
360° Room mode                                         Layout-first mode
──────────────                                         ─────────────────
Text prompt                                            Text prompt (room type + style) + seed
   │                                                      │
[1] 360° panorama: SD 2.1 panorama model,              [1] Parametric layout: room size, furniture boxes by
    circular padding (seamless wrap), 1024×512              placement rules → dimensioned floor plan (exact)
   │                                                      │
[2] Real-ESRGAN ×2 (wrap-padded tiles) → 2048×1024     [2] Layout rendered from the room centre →
   │                                                        equirectangular depth + semantic labels
[3] Depth: DAP + roll-and-blend wrap fix                  │
    (metric distance map)                              [3] SD 1.5 + ControlNet depth + 360° LoRA,
   │                                                        circular padding → panorama that follows the layout
[4] Segmentation: SegFormer on 14 perspective views,      │
    labels projected back                              [4] Real-ESRGAN ×2
   │                                                      │
[5] Full-room mesh: gravity alignment, wrap-around     [5] Mesh on the EXACT layout geometry (same mesh code),
    grid + pole caps, depth cuts, disocclusion band,        dollhouse; floor plan = the layout
    room box; dollhouse variant
   │
[6] 360° floor plan: closed room, walls, doors,
    windows, furniture, area
   │
[7] App: panorama + plan + inside / dollhouse viewers
```

---

## 4. Image-Processing Techniques (Course Concepts)

| Stage | Technique | Purpose |
|---|---|---|
| Training data cleaning | Laplacian variance | Detect and remove blurry images |
| Training data cleaning | Histogram analysis | Remove over/under-exposed images |
| Preprocessing | Resize, normalization | Standardize inputs to model resolution |
| Preprocessing | Gaussian / bilateral filtering | Remove noise while preserving edges |
| Preprocessing | CLAHE (contrast-limited histogram equalization) | Fix uneven indoor lighting |
| Preprocessing | Unsharp masking | Sharpen fine details before depth estimation |
| Analysis | Canny edge detection | Detect wall and furniture boundaries |
| Analysis | Segmentation (thresholding / morphology + pretrained segmenter) | Separate walls, floor, and furniture |
| Depth refinement | Median / bilateral filtering on depth map | Remove depth noise and spikes |
| Depth refinement | Edge-aware refinement using Canny map | Keep sharp object boundaries in 3D |
| Depth refinement | Morphological operations | Fill holes in the depth map |
| 3D | Back-projection with pinhole camera model | Convert pixels + depth into 3D points |
| Floor plan | Orthographic projection to an occupancy grid, morphological closing / opening | Top-down wall and floor maps |
| Floor plan | RANSAC line fitting, Manhattan snapping, collinear merging | Straight walls from noisy wall points |
| Floor plan | Connected components + minimum-area rectangles | Furniture instances and footprints |

---

## 5. Datasets

| Dataset | Use | Notes |
|---|---|---|
| **MMIS** (Kassab et al., 2024) | Train the LoRA for 2D generation | ~160K interior images (256×256), 5 room types, ~35 styles. https://github.com/AhmedMahmoudMostafa/MMIS (accessed Oct. 15, 2026) |
| **NYU Depth V2** | Evaluate depth estimation and preprocessing effect | Indoor RGB images with ground-truth depth. https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html |

We use a **subset of MMIS** (e.g., 5–10K images after cleaning) to fit local GPU resources (RTX 5060, 8 GB VRAM).

---

## 6. Models

| Component | Model | Role |
|---|---|---|
| Text-to-image | Stable Diffusion v1.5 | Base generator |
| 360° panorama | SD 2.1-base panorama fine-tune (SD-T2I-360PanoImage); compared with SD v1.5 + 360 LoRA | Equirectangular room |
| Super-resolution | Real-ESRGAN ×2 | Panorama to 2048×1024 |
| Panoramic depth | DAP (Depth Any Panoramas) + our wrap fix; compared with our multi-view Depth Anything alignment | Metric 360° distance map |
| Domain adaptation | LoRA (trained by us on MMIS) | Specialize generation for interiors |
| Depth estimation | Depth Anything (best on NYU), MiDaS (baseline), Depth Anything V2 Metric Indoor (3D stage) | Monocular depth from a single image |
| Segmentation | SegFormer-B2 (ADE20K) | Room-element masks |
| 3D | Open3D + own textured grid mesh | Point cloud cleaning, RANSAC planes, Poisson (comparison); textured mesh |

---

## 7. Training Plan (LoRA)

1. Clean the MMIS subset with Laplacian blur detection and exposure filtering.
2. Preprocess: upsample 256×256 → 512×512 (bicubic), normalize, apply CLAHE.
3. Build captions from MMIS labels (room type + style).
4. Train LoRA on SD v1.5 (Diffusers + PEFT) locally on RTX 5060 (8 GB VRAM).
   - Rank: 4–16 | LR: 1e-4 | Steps: ~2–5K | fp16 | gradient checkpointing | 8-bit Adam | batch 1 + gradient accumulation.
5. Save checkpoints; select the best by validation FID / CLIP Score.

**Done:** rank 8, alpha 8, LR 1e-4 cosine, 3,000 steps × grad-accum 4, 512 px, fp16, on 7,028 cleaned MMIS images. CLAHE was not applied to training images; the preprocessing was bicubic upsampling plus a light unsharp mask (`src/preprocessing.py`). Peak VRAM was 2.9 GB, and training took about 70 min on the RTX 5060. **Selected checkpoint: step 3000** (see section 8, Results (2D)).

---

## 8. Evaluation

### 2D Generation
| Metric | Measures |
|---|---|
| FID | Realism vs. real MMIS images |
| CLIP Score | How well the image matches the text prompt |

Comparison: **Vanilla SD v1.5 vs. SD v1.5 + our LoRA**.

#### Results (2D)
Protocol: 500 held-out MMIS captions (never used in training, 9–10 per style×room cell), one image per caption per model, same seed per caption across models, DDIM 30 steps, CFG 7.5, 512×512. FID and KID use Inception-v3 features against the 500 real held-out images, with generated images downscaled to 256×256 to match MMIS. KID is mean ± std over 100 random subsets of 100. CLIP Score uses ViT-L/14 and is reported as mean ± std per image. Code: `src/evaluate.py`, `scripts/eval_2d.py`.

| Model | FID ↓ | KID ×10³ ↓ | CLIP Score ↑ |
|---|---|---|---|
| SD v1.5 (base) | 64.5 | 19.1 ± 2.4 | **27.9 ± 1.9** |
| + LoRA step 1000 | 55.0 | 11.4 ± 2.2 | 26.8 ± 2.1 |
| + LoRA step 2000 | 54.9 | 10.5 ± 2.2 | 27.0 ± 2.1 |
| + LoRA step 3000 | **54.6** | **10.2 ± 2.1** | 27.0 ± 2.2 |
| *Real held-out MMIS images* | – | – | *22.8 ± 3.1* |

**Chosen checkpoint: `checkpoints/lora_mmis/step_03000`** (lowest FID and KID; rule: mean FID/KID rank, CLIP as tiebreaker).

- The LoRA makes generations much closer to real MMIS interiors: FID drops by 15% (64.5 → 54.6) and KID roughly halves (19.1 → 10.2). That gap is about four times the KID std.
- The three checkpoints are nearly tied. Their KID differences (11.4 / 10.5 / 10.2) are smaller than one std. Most of the gain comes in the first 1,000 steps, and later steps add a little.
- CLIP Score falls by about 1 point (27.9 → 27.0, about half a per-image std). The real MMIS images themselves score only 22.8 on the same captions, so moving toward the MMIS look costs some text alignment as CLIP measures it. This is a trade-off rather than a failure, but objective 1 is only met for FID/KID.
- Figures: `results/figures/eval_2d_metrics.png` (metrics) and `results/figures/eval_2d_qualitative.png` (6 captions × real + 4 models). Tables: `results/tables/eval_2d.csv` and `eval_2d_clip_per_image.csv`.

### Depth / 3D
| Metric | Measures |
|---|---|
| AbsRel | Mean relative depth error |
| RMSE | Absolute depth error |
| δ < 1.25 | % of pixels with accurate depth |
| Edge AbsRel / Edge RMSE | The same errors, only on pixels near a ground-truth depth edge (object boundaries) |
| Flying pixels | % of back-projected 3D points hanging between a foreground and a background surface |

Comparison: **MiDaS vs. Depth Anything** on NYU Depth V2.

#### Results (Depth)
Protocol: all 654 NYU Depth V2 test images at 480×640. Both models predict relative inverse depth, so each prediction is aligned to the ground truth with a least-squares scale and shift in disparity space, then scored inside the Eigen crop on pixels with valid depth up to 10 m. Metrics are computed per image and averaged. Nothing was tuned on the test set: every filter parameter and metric threshold was fixed before the full run. Code: `src/depth.py`, `src/evaluate.py`, `scripts/eval_depth.py`.

**Models (clean images, raw input)**

| Model | AbsRel ↓ | RMSE (m) ↓ | δ<1.25 ↑ | Edge AbsRel ↓ | Edge RMSE ↓ | Flying px % ↓ |
|---|---|---|---|---|---|---|
| MiDaS (DPT-Hybrid) | 0.0975 | 0.418 | 0.904 | 0.142 | 0.570 | 0.58 |
| Depth Anything (ViT-B) | **0.0546** | **0.273** | **0.965** | **0.102** | **0.436** | 0.67 |
| Depth Anything V2 Metric Indoor (ViT-B), aligned the same way | 0.0660 | 0.315 | 0.951 | 0.115 | 0.494 | 0.80 |

Depth Anything has 44% lower AbsRel than MiDaS and is the model used for all NYU experiments below. The 3D stage uses the metric variant, for the reason given under "3D Reconstruction".

### Key Experiment: Effect of Image Processing (Ablation)
Depth estimation on the clean NYU Depth V2 test images under each condition. Denoising and CLAHE are each applied on their own; "full" is denoise → CLAHE → unsharp mask.

| Condition | Depth Anything AbsRel | RMSE | δ<1.25 | MiDaS AbsRel | RMSE | δ<1.25 |
|---|---|---|---|---|---|---|
| Raw images | **0.0546** | 0.273 | 0.965 | **0.0975** | **0.418** | **0.904** |
| + Denoising | 0.0547 | **0.271** | 0.965 | 0.1027 | 0.437 | 0.896 |
| + CLAHE | 0.0553 | 0.274 | 0.964 | 0.0975 | 0.423 | 0.903 |
| + Full preprocessing | 0.0563 | 0.280 | 0.962 | 0.1000 | 0.428 | 0.898 |
| + Full preprocessing + depth refinement | 0.0566 | 0.280 | 0.962 | 0.1001 | 0.428 | 0.898 |

**On clean images preprocessing does not reduce depth error.** Every step is neutral or slightly worse than the raw image for both models (full preprocessing: +3% AbsRel for both).

#### Robustness
Each image is degraded synthetically at a moderate and a severe level, then the same ablation is run. AbsRel (lower is better); bold = best in the row.

| Model | Degradation | Raw | + Denoise | + CLAHE | + Full (plain) | + Full (noise-aware) | + Full (plain) + refine |
|---|---|---|---|---|---|---|---|
| Depth Anything | Clean | **0.0546** | 0.0547 | 0.0553 | 0.0563 | 0.0563 | 0.0566 |
| | Gaussian noise σ=15 | **0.0580** | 0.0580 | 0.0602 | 0.0602 | 0.0607 | 0.0605 |
| | Gaussian noise σ=25 | 0.0616 | **0.0600** | 0.0670 | 0.0637 | 0.0647 | 0.0640 |
| | Gaussian noise σ=50 | 0.0786 | **0.0675** | 0.0977 | 0.0835 | 0.0832 | 0.0837 |
| | Motion blur 15 px | **0.0636** | 0.0637 | 0.0665 | 0.0678 | 0.0678 | 0.0682 |
| | Motion blur 25 px | **0.0754** | 0.0763 | 0.0810 | 0.0829 | 0.0829 | 0.0832 |
| | Low light (γ 2.2 + noise 10) | **0.0618** | 0.0621 | 0.0639 | 0.0653 | 0.0661 | 0.0656 |
| | Low light (γ 3.0 + noise 20) | 0.0789 | **0.0775** | 0.0855 | 0.0843 | 0.0894 | 0.0845 |
| MiDaS | Clean | **0.0975** | 0.1027 | 0.0975 | 0.1000 | 0.1000 | 0.1001 |
| | Gaussian noise σ=15 | **0.1008** | 0.1074 | 0.1056 | 0.1045 | 0.1125 | 0.1045 |
| | Gaussian noise σ=25 | **0.1066** | 0.1095 | 0.1139 | 0.1116 | 0.1185 | 0.1115 |
| | Gaussian noise σ=50 | 0.1205 | **0.1169** | 0.1323 | 0.1289 | 0.1332 | 0.1288 |
| | Motion blur 15 px | 0.1252 | 0.1285 | **0.1207** | 0.1218 | 0.1218 | 0.1220 |
| | Motion blur 25 px | 0.1433 | 0.1473 | **0.1428** | 0.1433 | 0.1433 | 0.1434 |
| | Low light (γ 2.2 + noise 10) | **0.1096** | 0.1171 | 0.1118 | 0.1116 | 0.1183 | 0.1116 |
| | Low light (γ 3.0 + noise 20) | **0.1314** | 0.1381 | 0.1358 | 0.1411 | 0.1498 | 0.1410 |

- **The full pipeline does not improve robustness.** It is worse than the raw degraded image in 12 of the 14 degraded cases and equal in one (MiDaS, 25 px blur). The only gain is MiDaS under 15 px motion blur (0.1252 → 0.1218, 12% of the damage recovered).
- **Denoising alone helps when the noise is strong.** For Depth Anything it recovers 24% of the damage at σ=25 and 46% at σ=50 (0.0786 → 0.0675), and 6% in severe low light. For MiDaS it helps only at σ=50 (16%).
- **CLAHE alone hurts on noisy input** (Depth Anything σ=50: 0.0786 → 0.0977) because it amplifies the noise. It gives MiDaS a small gain under motion blur (16% at 15 px).
- **Both models are already robust.** Moderate degradations cost Depth Anything only 6–17% AbsRel, so there is little damage for classical filters to win back; these networks were trained on large, varied data.
- Recovery % = (error on degraded raw − error after the step) / (error on degraded raw − error on clean raw). All values are in `results/tables/depth_robustness.csv`.

**Default full pipeline: plain three steps (`noise_aware=False`).** A noise-aware variant (denoise again after CLAHE and skip sharpening when the input is noisy) was also evaluated. The two are identical on clean and blurred images. On noisy and dark images the plain version is better for MiDaS in all five cases (AbsRel lower by 0.004–0.009, five to ten standard errors in a per-image paired comparison) and for Depth Anything in severe low light (0.0843 vs. 0.0894). For Depth Anything under Gaussian noise the two are within 0.001 of each other. The noise-aware variant is never clearly better, so the simpler pipeline is the default. Neither variant beats the raw image.

#### Noise-gated preprocessing
The results above show that denoising only pays off when the image is really noisy, and that CLAHE and sharpening never help depth. `preprocess_gated` (`src/preprocessing.py`) therefore estimates the noise σ of the image and denoises only if it is above a threshold; otherwise the image is passed through unchanged. No CLAHE, no sharpening.

**Selection on the NYU train split (795 images, never the test split).** For both models and all 8 degradations, the raw, Gaussian-denoised and adaptive-bilateral-denoised images were scored; every threshold was then scored from those per-image numbers. Rule, fixed before the run: lowest mean relative AbsRel change against the raw image, averaged equally over the 2 models × 8 degradations; ties go to the higher threshold. 30 configurations (2 denoisers × 15 thresholds).

| Rank | Denoiser | Threshold (noise σ) | Mean AbsRel change vs. raw | Depth Anything | MiDaS | Worst cell | Images denoised |
|---|---|---|---|---|---|---|---|
| **1** | **Gaussian (σ=1)** | **15** | **−1.63%** | −2.91% | −0.34% | +0.40% | 33% |
| 2 | Gaussian | 12 | −1.62% | −3.35% | +0.12% | +3.04% | 50% |
| 3 | Gaussian | 10 | −1.61% | −3.35% | +0.12% | +3.05% | 50% |
| 4 | Gaussian | 20 | −1.49% | −2.59% | −0.39% | 0.00% | 25% |
| 14 | Adaptive bilateral (its best) | 30 | −0.93% | −1.60% | −0.26% | 0.00% | 13% |
| 19 | Gaussian | 0 (always denoise) | −0.45% | −3.52% | +2.63% | +6.42% | 100% |
| 30 | Adaptive bilateral | 0 (always denoise) | +1.14% | −2.23% | +4.51% | +10.13% | 100% |

**Selected: Gaussian denoising when the estimated noise σ is above 15.** The top three are within 0.02 points of each other, so the exact threshold between 10 and 15 matters little. The Gaussian filter beats the adaptive bilateral filter at every threshold even though the bilateral filter has the better PSNR at σ=15 and σ=25: for depth, removing the noise completely matters more than keeping edges sharp. Tables: `results/tables/depth_gated_selection.csv`, `depth_gated_selection_cells.csv`, `depth_gated_selected.csv`.

**Evaluation on the 654 test images (run once with the selected config).** AbsRel, raw → gated; recovery = share of the damage won back.

| Degradation | Images denoised | Depth Anything | Recovery | MiDaS | Recovery |
|---|---|---|---|---|---|
| Clean | 0% | 0.0546 → 0.0546 | – | 0.0975 → 0.0975 | – |
| Gaussian noise σ=15 | 0% | 0.0580 → 0.0580 | 0% | 0.1008 → 0.1008 | 0% |
| Gaussian noise σ=25 | 100% | 0.0616 → **0.0582** | 49% | 0.1066 → **0.1054** | 14% |
| Gaussian noise σ=50 | 100% | 0.0786 → **0.0643** | 59% | 0.1205 → **0.1167** | 16% |
| Motion blur 15 px | 0% | 0.0636 → 0.0636 | 0% | 0.1252 → 0.1252 | 0% |
| Motion blur 25 px | 0% | 0.0754 → 0.0754 | 0% | 0.1433 → 0.1433 | 0% |
| Low light (γ 2.2 + noise 10) | 0% | 0.0618 → 0.0618 | 0% | 0.1096 → 0.1096 | 0% |
| Low light (γ 3.0 + noise 20) | 59% | 0.0789 → **0.0774** | 6% | 0.1314 → 0.1324 | −3% |

- **The gated pipeline never hurts where it does nothing:** clean, lightly noisy, blurred and moderately dark images pass through unchanged (5 of 8 conditions), so the losses of the full pipeline on those cases are gone.
- **It helps under strong Gaussian noise.** Depth Anything recovers 49% of the AbsRel damage at σ=25 and 59% at σ=50 (RMSE 0.350 → 0.303 m, δ<1.25 0.933 → 0.954); per image, AbsRel improves on 71% and 83% of the images. MiDaS gains less (14% and 16%).
- **Severe low light is mixed:** a small gain for Depth Anything (6%), a small loss for MiDaS (−3%, about two standard errors).
- **It does not address blur or moderate low light,** and it does not improve clean images. Averaged over the 8 conditions the AbsRel change is −3.2% for Depth Anything and −0.4% for MiDaS, close to the train-split estimates (−2.9% and −0.3%), so the selection did not overfit.
- **Depth refinement** on top of the gated input behaves as before: flying pixels drop by roughly a third (Depth Anything, σ=25: 0.61% → 0.39%), AbsRel changes by less than 0.001.
- Tables: `results/tables/depth_gated_test.csv`, `depth_gated_per_image.csv`. Figures: `results/figures/depth_gated_selection.png`, `depth_gated_test.png`. Code: `scripts/eval_gated.py`.

#### Denoising quality
PSNR and SSIM against the clean image, mean over the 654 images. "Adaptive bilateral" sets the colour sigma to three times the estimated noise level and is what the pipeline uses.

| Noise σ | Method | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| 15 | Noisy input | 25.1 | 0.596 |
| | Gaussian (σ=1) | 28.9 | **0.897** |
| | Bilateral, fixed | 32.2 | 0.881 |
| | Bilateral, adaptive | **32.4** | 0.887 |
| 25 | Noisy input | 20.9 | 0.405 |
| | Gaussian (σ=1) | 27.2 | **0.821** |
| | Bilateral, fixed | 25.4 | 0.622 |
| | Bilateral, adaptive | **28.9** | 0.791 |
| 50 | Noisy input | 15.3 | 0.211 |
| | Gaussian (σ=1) | **23.2** | **0.634** |
| | Bilateral, fixed | 16.5 | 0.247 |
| | Bilateral, adaptive | 21.3 | 0.443 |

The adaptive bilateral filter gives the best PSNR at σ=15 and σ=25 (+7.3 and +8.0 dB). The plain Gaussian filter has the best SSIM at every level and also the best PSNR at σ=50, where the bilateral filter leaves visible residual noise. A fixed bilateral filter stops working once the noise exceeds its colour sigma. Table: `results/tables/depth_denoise_quality.csv`.

#### Depth refinement and boundary metrics
Refinement = median + bilateral filtering of the depth map, edge-aware smoothing guided by the Canny map, and morphological hole filling. Depth Anything, before → after refinement:

| Input | Degradation | Flying px % | Edge AbsRel | AbsRel |
|---|---|---|---|---|
| Raw | Clean | 0.67 → 0.43 | 0.1020 → 0.1038 | 0.0546 → 0.0550 |
| Full (plain) | Clean | 0.68 → 0.44 | 0.1026 → 0.1037 | 0.0563 → 0.0566 |
| Full (plain) | Gaussian noise σ=25 | 0.58 → 0.37 | 0.1123 → 0.1134 | 0.0637 → 0.0640 |
| Full (plain) | Low light (γ 2.2 + noise 10) | 0.54 → 0.35 | 0.1145 → 0.1154 | 0.0653 → 0.0656 |

- **Refinement removes roughly a third of the flying pixels** for both models in every condition (MiDaS clean, raw: 0.58 → 0.38).
- **It does not improve accuracy.** Edge AbsRel and global AbsRel change by about 0.001 or less, slightly worse for Depth Anything and mixed for MiDaS. Its value is a cleaner point cloud, not a lower depth error.
- **Caveat, flying pixels:** the measure counts points whose neighbours on both sides are more than 5% of the depth away. A blurred prediction spreads a depth step over many pixels and falls under that threshold, so the ratio also drops when the depth map is simply smoother (Depth Anything under 25 px motion blur: 0.19% with a higher error). It must be read together with Edge AbsRel.
- **Caveat, Kinect edges:** the NYU ground truth is in-painted and imprecise near object boundaries, so the edge mask and the edge errors are approximate.
- Table: `results/tables/depth_refinement_boundary.csv`. Per-image values: `results/tables/depth_per_image.csv`.

Figures: `results/figures/depth_ablation.png`, `depth_robustness_depth_anything.png`, `depth_robustness_midas.png` (metrics); `depth_preprocessing_steps.png`, `depth_refinement_steps.png`, `depth_qualitative.png`, `analysis_edges_segmentation.png` (before/after examples).

### 3D Reconstruction
Code: `src/reconstruct_3d.py`, `src/pipeline.py`, `scripts/run_pipeline.py`, `scripts/compare_depth_3d.py`.

**Camera model.** A generated image has no real camera, so one is assumed: a pinhole camera with a **60° horizontal field of view** and the principal point at the image centre (focal length 443 px at 512×512). A pixel (u, v) with depth Z is back-projected to X = (u − cx)·Z/f, Y = (v − cy)·Z/f. A wrong FOV stretches or squeezes the room along the viewing direction.

**Steps.** Metric depth → depth refinement (on inverse depth) → back-projection → (a) coloured point cloud with flying-pixel and statistical outlier removal (Open3D, 20 neighbours, 2 std), exported as `.ply`; (b) textured mesh, exported as `.glb` and `.obj`.

**Mesh: textured grid with filled occlusions** (`src/textured_mesh.py`). The first version used Poisson surface reconstruction of the point cloud. It gave blurry colours (one colour per vertex), rounded edges and holes, so the mesh is now built directly on the pixel grid in three layers that share one texture atlas:

1. **Observed surface.** One vertex every 2 pixels, two triangles per quad, the generated image as UV texture at full resolution. Blurred depth edges are first turned into steps by a morphological toggle mapping (each pixel in a transition zone snaps to the local erosion or dilation, whichever is closer), and the foreground is grown by 2 px so that object-coloured boundary pixels stay on the object. Triangles are cut where the depth step between neighbouring vertices is above 8%, or above 4.5% with a Canny edge next to it.
2. **Disocclusion band.** Behind every cut, the background is continued 24 px under the foreground object: the strip is built by geodesic dilation, its depth is grown in from the background side, and its colour comes from `cv2.inpaint` (Telea) with the mask dilated by 11 px so that no foreground-coloured pixel is used as a source.
3. **Room planes.** RANSAC planes are fitted to the floor, ceiling and wall masks (up to three walls). Each pixel ray is intersected with the accepted planes and the nearest crossing is the room surface, which extends every plane to the edge of the view frustum and clips the planes against each other. Where that surface is hidden behind furniture it is added to the mesh, coloured by inpainting from the visible part of the same plane and fading to that plane's median colour far from it. A plane is skipped if too few mask points lie on it, if it is not roughly horizontal (floor, ceiling) or vertical (wall), or if more than 15% of the image shows geometry behind it. In the six demo rooms 17 of 25 fitted planes were accepted.

The `.glb` uses an unlit material, so a viewer shows the image colours unchanged. Poisson reconstruction (`reconstruct_3d.poisson_mesh`) is kept as the comparison method.

**Old vs. new mesh** on the six demo rooms (`scripts/compare_mesh.py`), each rendered from the original camera and with the camera turned ±25° around the scene. "Empty" is the share of the rendered picture where there is no surface:

| Method | Time (s) | Triangles | Empty, camera view | Empty, −25° | Empty, +25° |
|---|---|---|---|---|---|
| Poisson (old) | 9.3 | 200,000 | 3.1% | 24.0% | 18.7% |
| Textured grid, cuts only | 0.03 | 128,000 | 0.2% | 25.6% | 21.0% |
| Textured grid + inpainting + planes (new) | **0.7** | 193,000 | **0.0%** | **14.6%** | **13.0%** |

- **From the camera view the new mesh reproduces the image**, sharp and without holes; the Poisson mesh is blurred and 3% empty.
- **Turned by 25°, the empty share falls from 24% / 19% to 15% / 13%.** It is lower in all six rooms at −25° and in five of six at +25° (the bathroom, a flat wall, is unchanged). Most of what stays empty is the margin outside the original field of view, which a single image cannot fill.
- **It is about 12 times faster** (0.7 s vs. 9.3 s) and the `.glb` is 3.3–5.7 MB.
- **Where the new mesh still fails:**
  - Filled areas have no real texture. Large hidden regions get a flat plane colour (clearly visible as plain patches in the rustic kitchen).
  - Thin objects (chair legs, the chandelier, plant leaves) are torn into fragments when the view turns, because the depth map does not resolve them.
  - Surfaces seen at a grazing angle are stretched (the plants at the left edge of the mediterranean room).
  - The room planes assume a convex room with the camera inside; 8 of 25 planes were rejected, and those rooms are filled only by the 24 px band.
  - Beyond about ±30° the missing parts dominate, which is why the demo limits the orbit.
- Figures: `results/figures/mesh_old_vs_new.png` (all six rooms) and `results/figures/mesh_layers.png` (what each step adds). Table: `results/tables/mesh_old_vs_new.csv`.

**Depth model for the 3D stage: Depth Anything V2 Metric Indoor.** The relative Depth Anything model is the more accurate one on NYU after alignment, but alignment needs ground truth. On a generated image the unknown scale and shift have to be guessed (here: nearest point 1.5 m, farthest 6 m), and a wrong shift is not a uniform scaling: it bends the geometry. The metric model predicts metres directly. Both were compared on the six demo rooms (`results/figures/depth_relative_vs_metric_3d.png`, side and top views):

- **Visual:** with the relative model the far parts of the room are stretched into long streaks (the windows of the living room, the back wall of the bathroom); with the metric model the same rooms are compact and the bathroom wall is a straight line in the top view.
- **Room size:** the guessed range makes every room 1.4–6 m deep, a bathroom as well as a large living room. The metric model gives each room its own size (bathroom 2.0–3.1 m, living room 3.0–13.4 m).
- **Plane checks from the segmentation masks are not decisive.** The floor is slightly flatter with the metric model (RMS distance from the best plane 1.16% vs. 1.37% of the scene depth, better or equal in all 5 rooms with a visible floor). The floor–wall angle is closer to 90° for the relative model in 3 of 5 rooms and for the metric model in 2; the means (error 3.5° vs. 14.9°) are dominated by one room. Table: `results/tables/depth_3d_model_choice.csv`.

The choice rests on the visual comparison and on not having to guess a scale and shift. The price, measured on the 654 NYU test images (`scripts/eval_depth_metric.py`):

| Depth Anything V2 Metric Indoor (ViT-B) | AbsRel ↓ | RMSE (m) ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Aligned (scale + shift, same protocol as the models table) | 0.0660 | 0.315 | 0.951 |
| As predicted, in metres, no alignment | 0.2195 | 0.660 | 0.664 |

After alignment it is a little less accurate than the relative model (0.0660 vs. 0.0546) and better than MiDaS. Without alignment its absolute scale is off by about 22% on NYU (it was trained on synthetic indoor scenes, not on NYU), so sizes read from the 3D model are approximate.

**End-to-end run** on six fixed prompts and seeds (not selected after looking at the results), RTX 5060 Laptop 8 GB, seconds per prompt:

| Prompt (seed) | 2D image | Analysis + depth | Point cloud | Mesh | Export | Total |
|---|---|---|---|---|---|---|
| a contemporary style living room (11) | 3.9 | 0.10 | 0.6 | 0.7 | 0.7 | 6.0 |
| a coastal style bedroom (22) | 3.7 | 0.10 | 0.2 | 0.7 | 0.7 | 5.4 |
| a rustic style kitchen (33) | 3.6 | 0.11 | 0.2 | 0.9 | 0.9 | 5.6 |
| a mid-century modern style dining room (44) | 3.6 | 0.10 | 0.1 | 0.7 | 0.7 | 5.3 |
| an art deco style bathroom (55) | 3.6 | 0.10 | 0.1 | 0.3 | 0.5 | 4.6 |
| a mediterranean style living room (66) | 3.5 | 0.10 | 0.1 | 1.1 | 0.8 | 5.6 |
| **Mean** | 3.7 | 0.10 | 0.2 | 0.7 | 0.7 | **5.4** |

- **Objective 4 is met:** 5.4 s per prompt on average (max 6.0 s) with the models loaded; loading the three models once adds 13.1 s. With the earlier Poisson mesh the average was 14.4 s. "Analysis + depth" covers gated preprocessing, Canny, segmentation, depth and refinement. Image generation is now the slowest stage.
- **Peak VRAM:** 2.7 GB (Stable Diffusion); the segmenter and the depth model need about 0.3 GB each. With all three loaded together, as in the demo, the peak is 2.9 GB.
- **What the 3D model is:** a single-view reconstruction. It matches the image from the original viewpoint and holds up for moderately turned views; what the image never showed is either filled with plane colour or missing.
- Table: `results/tables/pipeline_timing.csv`. Figures: `results/figures/pipeline_3d_results.png` (image, edges, segmentation, depth, point cloud, mesh for each prompt) and `results/figures/pointcloud_refinement.png` (point cloud without and with refinement + cleaning). The 3D files are written to `outputs/pipeline/` (not in git).

### Demo (local)
`app/app.py` is a Gradio app: prompt, seed, steps and CFG in; the **2D image, the floor plan and the 3D viewer** out, with download buttons for the image, the floor plan (`.svg`), the mesh (`.glb`) and the point cloud (`.ply`). The depth map (grayscale, white = near), the Canny edges and the segmentation are in a collapsed "Analysis" section. The 3D model is shown in two tabs:

- **Original view (limited orbit)**, the default: a `<model-viewer>` element that opens exactly at the camera of the generated image (60° field of view) and limits the orbit to ±30° left/right, so the first impression matches the image and the parts a single view cannot know stay out of sight. Its script is loaded from a CDN, so this tab needs an internet connection.
- **Free orbit**: `gr.Model3D` without limits (works offline). `gr.Model3D` has no setting for orbit limits, which is why the first tab uses a different viewer.

The three models stay loaded together (2.9 GB VRAM); a request takes about 5 s after the first one. Launch from the project root and open http://127.0.0.1:7860:

```
.venv\Scripts\python.exe app\app.py
```

It is not deployed to Hugging Face Spaces yet.

### Floor Plan (milestone 1 of PLAN_V2)
Code: `src/floorplan.py`, `scripts/eval_floorplan.py`, `scripts/floorplan_demo.py`.

**Method.** The metric point cloud and the SegFormer class map of one view are turned into a measured, architect-style plan:

1. **Gravity.** A RANSAC plane is fitted to the floor pixels (fallback: the ceiling, then an upright camera). Its normal is "up"; the camera height is its distance. Points are projected orthographically onto the floor plane, x to the right and y forward in metres, camera at the origin.
2. **Walls.** Wall points higher than 0.3 m are rasterised on a 2 cm grid, cleaned by morphological closing (5×5) and opening (3×3), and lines are found by sequential RANSAC on the occupied cells (inlier band 6 cm or 2% of the range, since depth noise grows with distance). Lines are split at gaps over 35 cm. A **free-space check** drops lines that have more than 15% of the visible floor/wall/furniture points behind them (a real boundary is the last surface along each ray), and of two nearly parallel lines that cover the same viewing directions only the better supported one is kept. If at least 60% of the wall length agrees with one direction (mod 90°), segments within 12° are snapped to it (Manhattan). Collinear segments are merged and perpendicular walls are extended to their corners.
3. **Furniture.** Each furniture class mask is split into instances (opening + connected components, at least 0.12% of the image, eroded by 3 px). The projected points are cleaned (range IQR filter, largest blob on a 5 cm grid), and the footprint is the **minimum-area rectangle**. One view sees only the front of an object, so a footprint thinner than a class prior (e.g. bed 1.0 m, sofa 0.8 m, chair 0.45 m) is extended away from the camera and drawn dashed. Same-label instances whose footprints overlap are merged (one object split by an occluder).
4. **Doors and windows** are projected onto the wall line most of their points lie on; their extent along it becomes the opening.
5. **Drawing.** Walls as thick black bands, door leaf + swing arc, window triple lines, labelled furniture outlines, dimension lines with "≈", a 1 m scale bar, the camera with its field of view, and the floor area if the room polygon is closed (otherwise the visible floor area). PNG and SVG.

All dimensions are marked approximate: the metric depth model is about 22% off in absolute scale on NYU (section 8, 3D Reconstruction).

**Validation on the 654 NYU test images.** Ground-truth floor plans do not exist for NYU, so the plan built from the **ground-truth Kinect depth** is the reference. The same segmentation, the real NYU camera intrinsics and the same pixels (Eigen crop, valid GT) are used for every plan, so the comparison measures what the depth prediction does to the plan. Walls are matched by the overlap of their viewing directions (independent of scale), furniture by shared image instances. "Median-scaled" multiplies the prediction by median(GT / prediction) per image; it needs ground truth and is a diagnostic only, separating the global-scale error from the shape error. Parameters were fixed on the six demo rooms before this run.

| Plan from | Wall recall | Wall precision | Wall length error (mean / rel. mean / rel. median) | Wall length ratio pred/GT (median) | Wall distance error | Furniture IoU | Furniture IoU (observed part) | Centroid error (mean / median) |
|---|---|---|---|---|---|---|---|---|
| Predicted metric depth, as is (the app) | 0.54 | 0.57 | 0.60 m / 32% / 23% | 1.16 | 27% | 0.25 | 0.20 | 0.53 m / 0.47 m |
| Predicted depth, median-scaled (diagnostic) | 0.53 | 0.60 | 0.42 m / 22% / 12% | 0.99 | 15% | 0.51 | 0.43 | 0.22 m / 0.17 m |

- **Scale dominates the error.** With the depth as predicted, walls come out 16% too long (median) and objects are placed about 0.5 m off; furniture footprints overlap their reference with IoU 0.25. Fixing only the global scale halves the relative length error (median 23% → 12%), cuts the centroid error by almost two thirds (median 0.47 → 0.17 m) and doubles the IoU (0.25 → 0.51). So the shape of the plan is mostly right and the absolute size is the weak point, which is what the "≈" marks are for.
- **Gravity is robust:** the up direction from predicted depth is within 1.8° (median) of the one from ground-truth depth; a floor plane is found in 84–86% of the images.
- **Walls agree only about half the time** (recall 0.54, precision 0.57). Many NYU views show short, cluttered wall pieces, and the reference plan is itself built from noisy Kinect depth, so small segments appear in one plan and not in the other.
- **No NYU view gives a closed room** (0 of 654): a single 60° view never shows walls on all four sides. This is why milestones 2–5 move to 360° panoramas.
- Plan building takes 0.17–0.19 s per image on the CPU. Ceiling height above the fitted floor, median: 2.36 m from GT depth, 2.91 m from predicted depth (the same scale bias).
- Tables: `results/tables/floorplan_nyu.csv` (summary), `floorplan_nyu_per_image.csv`, `floorplan_nyu_stats.csv`. Figure: `results/figures/floorplan_nyu_examples.png` (image | plan from GT depth | from predicted depth | median-scaled).

**Demo rooms** (`results/figures/floorplan_demo.png`, `results/tables/floorplan_demo.csv`): 2–3 walls per room after cleaning, 1–14 furniture items, 0.1–0.7 s to build and 0.1–0.2 s to draw. The generated rooms come out large (ceilings 3.7–5.2 m above the fitted floor), so either the scale is overestimated on generated images too or SD draws tall rooms; without ground truth the two cannot be told apart. In the bathroom no floor or ceiling is visible, so the plan falls back to an upright camera.

### 360° Panorama Generation (milestone 2 of PLAN_V2)
Code: `src/generate_pano.py`, `src/equirect.py`, `scripts/eval_pano.py`.

A single 60° view can only give a relief of what the camera saw. The 360° mode generates one **equirectangular panorama** (1024×512, 360°×180°) instead, which later stages turn into depth, a full-room mesh and a closed floor plan.

- **Generators.** (a) SD v1.5 + the panorama LoRA `ProGamerGov/360-Diffusion-LoRA-sd-v1-5` (trigger word `qxj`) + our MMIS LoRA at weight 0 / 0.5 / 0.8 / 1.0 (unfused PEFT adapters, so the weight can change per call). (b) The alternative generator of the plan, the SD 2.1-base panorama fine-tune of SD-T2I-360PanoImage (`archerfmy0831/sd-t2i-360panoimage`, only its `sd-base` text-to-panorama part, run with our code). DDIM 30 steps, CFG 7.5, fixed seeds.
- **Seamless wrap (image processing).** The left and right borders of an equirectangular image are the same meridian. Every padded convolution of the UNet and the VAE decoder is patched to pad **circularly along x** (zeros along y), so the borders are neighbours during the whole denoising and decoding. Optionally the padding is switched on only after half of the steps.
- **Protocol.** 5 rooms × 6 styles × 2 seeds = 60 panoramas per configuration. Seam ratio = mean |grey difference| across the wrap divided by the mean between neighbouring columns (1 = no visible seam). CLIP (ViT-L/14) and FID/KID are computed on four 90° perspective crops per panorama (yaw 0/90/180/270), 240 crops against the 500 MMIS hold-out images, the same protocol as the 2D evaluation. Selection rule, fixed before the run: lowest mean rank of KID and CLIP among the circular configurations.

| Configuration | Seam ratio (median) ↓ | CLIP ↑ | FID ↓ | KID ×10³ ↓ | s / panorama |
|---|---|---|---|---|---|
| Pano LoRA, no circular padding | 3.60 | 20.6 ± 2.5 | 121.9 | 66.0 ± 3.2 | 10.2 |
| Pano LoRA, circular from 50% of the steps | 0.77 | 20.5 ± 2.5 | 121.0 | 64.5 ± 3.0 | 10.1 |
| Pano LoRA, circular | 0.83 | 20.4 ± 2.8 | 124.3 | 63.7 ± 2.9 | 10.1 |
| + MMIS LoRA 0.5 | 0.79 | 20.2 ± 2.8 | 122.3 | 64.1 ± 3.1 | 10.1 |
| + MMIS LoRA 0.8 | 0.76 | 19.9 ± 2.5 | 123.1 | 64.6 ± 3.3 | 10.1 |
| + MMIS LoRA 1.0 | 0.73 | 19.8 ± 2.4 | 123.8 | 66.3 ± 3.2 | 10.1 |
| **SD 2.1 panorama model, circular (selected)** | 0.81 | **22.4 ± 2.0** | **93.3** | **38.2 ± 2.4** | **7.3** |

- **Circular padding removes the seam:** median seam ratio 3.60 → 0.83, i.e. the wrap looks like any other column boundary. Switching it on halfway is equally seamless; neither variant costs realism (KID within one std).
- **Our MMIS LoRA does not help in panorama mode** (negative result). KID stays within about one std (63.7 → 64.1 / 64.6 / 66.3) and CLIP falls slightly as its weight rises. Visually it changes materials (wood floors, rugs) but not the room layout. The LoRA was trained on perspective photos, so it brings little to equirectangular images.
- **The alternative generator is clearly better:** KID 38.2 vs. 63.7 (ten standard deviations), FID 93 vs. 124, CLIP +2.0, and 28% faster. The pano-LoRA rooms tend to repeat the same objects around the horizon and have bulging, arch-like walls; the SD 2.1 panorama model draws coherent rooms with straight walls. **It is the 360° generator from here on, so the 360° mode does not use our MMIS LoRA** (the 2D Quick mode still does). Peak VRAM: 3.6 GB (SD 1.5 + two adapters).
- Tables: `results/tables/pano_configs.csv`, `pano_per_image.csv`. Figures: `results/figures/pano_configs_grid.png` (panoramas rolled so the seam is in the middle), `pano_configs_metrics.png`.

**Super-resolution to 2048×1024** (30 panoramas of the selected configuration):

| Method | Sharpness (Laplacian variance) ↑ | PSNR after downscaling back (dB) | Seam ratio | s / panorama |
|---|---|---|---|---|
| Bicubic ×2 + unsharp mask (wrap-aware) | 97 | 51.6 | 0.98 | 0.05 |
| **Real-ESRGAN ×2** (tiled, wrap-padded) | **315** | 33.8 | 1.04 | 0.76 |

Real-ESRGAN (own minimal RRDBNet implementation, x2plus weights, 256-px tiles; the panorama is padded by 16 columns copied from the opposite border so the network sees continuous content across the seam) is 3.3 times sharper and stays seamless, so it is the default. It is less faithful to the input: downscaled back, it differs from it by 33.8 dB PSNR against 51.6 dB for bicubic, because it smooths fine texture into clean, painterly regions (`results/figures/pano_sr_compare.png`). Tables: `pano_sr.csv`, `pano_sr_per_image.csv`.

### Panoramic Depth (milestone 3 of PLAN_V2)
Code: `src/pano_depth.py`, `scripts/eval_pano_depth.py`.

Two ways to get a metric **distance map** for an equirectangular panorama were compared:

- **Method A (ours, image processing).** The panorama is cut into 14 overlapping 100° perspective views (6 cube faces + 8 views at ±35° pitch between them). Depth Anything V2 Metric Indoor predicts z-depth per view; z is converted to the distance along each ray (r = z·|ray|). Every view is resampled onto the equirectangular grid with a feather weight (cos⁴ of the angle to the view axis × a border ramp). Per-view **scales** are found by a global least-squares fit over all overlapping pixels in log space (log sᵢ − log sⱼ = log Dⱼ − log Dᵢ, mean log s = 0, robust Cauchy re-weighting), optionally followed by per-view **shifts** with the scales fixed (a joint scale + shift fit is ill-posed: all views can shrink and take the same shift, which makes them agree on a constant map). The aligned views are blended with the feather weights.
- **Method B: DAP** ("Depth Any Panoramas", Insta360, ViT-L/DINOv3), a metric panoramic depth network run on the whole 1024×512 image. It must run in fp32 (fp16 overflows to NaN); 2.3 GB VRAM.
- **Wrap fix for DAP (ours).** DAP does not treat the left and right borders as neighbours and its prediction jumps at ±180° (seam ratio 9.8). It is run a second time on the panorama rolled by half a width, the result is rolled back, and the two are cross-faded with weights cos²(lon/2) and sin²(lon/2), so each prediction is used far from its own border.
- **Post-processing:** the noise-gated denoising and the depth refinement of the Quick mode, applied with horizontal wrap padding (the filters see the opposite border).

**Evaluation** on the 30 panoramas of the selected generator (first seed of every prompt). Segmentation for the checks: SegFormer on the 14 perspective views, labels projected back (segmenting the equirectangular image directly labels most of the distorted ceiling as wall). Medians over the 30 panoramas:

| Method | View disagreement before → after | Floor: inliers / RMS (% of camera height) / tilt | Walls: points on a plane / RMS (% of distance) / angle error | Room height (share in 2.2–3.5 m) | Depth seam ratio | s / VRAM |
|---|---|---|---|---|---|---|
| A, views blended without alignment | 33% | 43% / 9.5% / 7.9° | 17% / 8.4% / 9.5° | 5.11 m (3%) | 0.67 | – |
| A, scale alignment | 33% → 10.3% | 62% / 6.6% / 4.3° | 17% / 8.5% / 6.6° | 6.13 m (0%) | 0.65 | 1.7 s / 0.3 GB |
| A, scale + shift | 33% → 10.7% | 61% / 6.5% / 4.8° | 21% / 7.9% / 4.2° | 6.05 m (0%) | 0.64 | 1.8 s / 0.3 GB |
| B, DAP | – | 95% / 2.6% / 0.4° | 28% / 8.3% / 4.5° | 2.88 m (100%) | 9.75 | 0.4 s / 2.3 GB |
| **B, DAP + wrap fix (selected)** | – | **95% / 2.7% / 0.5°** | **28% / 8.3% / 3.7°** | **2.88 m (100%)** | **0.62** | 0.9 s / 2.3 GB |
| B, DAP + wrap fix + gated denoise + refinement | – | 94% / 3.1% / 0.4° | 28% / 8.3% / 4.8° | 2.91 m (100%) | 0.74 | |

- **Our alignment works on its own terms:** the overlapping views disagree by 33% before and 10% after the scale fit, and the floor becomes much flatter (inliers 43% → 62%, tilt 7.9° → 4.3°). The shift step adds nothing (shifts come out near zero).
- **But DAP gives far better room geometry:** a flat, level floor (95% of the floor points on one plane, 0.4° tilt), and every room height between 2.2 and 3.5 m (median 2.88 m, camera 1.19 m above the floor). Method A puts the rooms twice as far away (median distance 3.9 m vs. 1.8 m) and makes them 6 m tall: the per-view metric model is given perspective crops of a generated room without scale cues (a crop of the ceiling is predicted 0.7 m away), and the alignment can only make the views agree with each other, not fix the common scale. The same overestimate showed up in the single-view floor plans (milestone 1: ceilings 3.7–5.2 m).
- **The wrap fix removes DAP's seam** (seam ratio 9.75 → 0.62) for 0.4 s more and changes none of the geometry numbers. Wrap-aware refinement alone also reduces the seam (to 1.31) but less.
- **Gated denoising + refinement do not improve the geometry** (floor RMS 2.7% → 3.1%), the same finding as on NYU (it cleans boundaries but does not lower the error), so the selected depth for the 3D stage is DAP + wrap fix without refinement.
- **Walls are only partly planar for every method** (28% of the wall points on one of up to 6 vertical planes within 2% of the distance): generated rooms contain curved and bulging walls, and SegFormer labels some cupboards and curtains as wall.
- **No ground truth.** No real RGB-D panorama set could be used within the plan's rules: the official Stanford2D3D, Matterport3D and Structured3D releases require a signed licence agreement, and the copies found on the Hugging Face Hub are third-party redistributions of that data, which were not used. The numbers above are geometric plausibility checks, not accuracy.
- Tables: `results/tables/pano_depth.csv` (summary), `pano_depth_per_image.csv`, `pano_depth_timing.csv`. Figure: `results/figures/pano_depth_examples.png`.

### Full-Room 3D (milestone 4 of PLAN_V2)
Code: `src/room_mesh.py`, `src/pipeline_360.py`, `scripts/run_room360.py`, `scripts/eval_room3d.py`.

**360° Room pipeline:** prompt → SD 2.1 panorama model with circular padding (1024×512) → Real-ESRGAN ×2 texture (2048×1024) → DAP + wrap fix (distance map) → SegFormer on 14 views (class map) → mesh + dollhouse + floor plan.

**Mesh** (`room_mesh.build_room_mesh`), reusing the Quick-mode mesh logic on a sphere:
- **Gravity alignment:** RANSAC floor plane → the mesh is rotated so that "up" is exact; the panorama centre is the origin.
- **Spherical back-projection** on a grid with one vertex every 2 pixels. The grid **wraps around**: the last column is joined to the first through a duplicate column that carries texture u = 1, so there is no seam in the mesh. **Pole caps** close the top and bottom with a fan of triangles (one pole vertex per column, so the texture is not smeared).
- **Observed surface (layer 0):** depth edges are sharpened and the foreground grown, then triangles are cut at depth discontinuities (8%, or 4.5% next to a Canny edge). Every 2D filter runs on a wrap-padded copy.
- **Disocclusion band (layer 1):** the background is continued 24 px behind every cut, coloured by inpainting.
- **Room box (layer 2):** floor plane, ceiling height and the four walls of the closed 360° floor plan (milestone 5). Where the box surface is hidden behind furniture it is added and painted from the visible part of the same plane, so moving away from the centre shows wall and floor behind objects instead of holes.
- **Dollhouse variant:** every triangle within 25 cm of the ceiling height is removed, so the room can be looked into from above.
- Exported as `.glb` (unlit, double-sided): 0.34–0.41 M triangles, 8–10 MB.

**Viewer (app, 360° Room mode):** "3D viewer" opens **inside** the room: the orbit target is the panorama centre, dragging turns the view around, and the zoom moves the camera 5 cm–1 m from the centre. "Dollhouse" orbits the ceiling-less room from outside and above (polar angle 0–80°). `gr.Model3D` stays as the offline fallback. The app keeps only the active mode's diffusion models on the GPU and moves the other mode's to the CPU when the mode changes (first switch 8.6 s, later about 2 s). Peak VRAM of a 360° request: 5.5 GB. Tested headlessly (both modes, switching back and forth); the two `<model-viewer>` modes have not yet been checked by hand in a browser.

**Evaluation** on the six demo prompts and seeds (the Quick mode's single-view meshes vs. the 360° meshes). Both are gravity-aligned and rendered by exact ray casting (Open3D); "empty" = share of rays that hit no surface. Inside: camera at the centre and moved 0.5 m / 1.0 m in four horizontal directions, eight 90° views around (yaw 0, 45, …, 315). "Front" = the three yaws within ±45° of the single view's own direction. Dollhouse: eight views from outside, 55° above the horizon, counting only rays that pass through the room's bounding box.

| Empty share | Single view (all directions / front) | 360° room (all directions / front) |
|---|---|---|
| Inside, at the centre | 89.7% / 72.4% | **0.4% / 0.3%** |
| Inside, 0.5 m from the centre | 89.3% / 71.5% | **0.5% / 0.4%** |
| Inside, 1.0 m from the centre | 88.6% / 69.7% | **0.7% / 0.5%** |
| Dollhouse (outside, above) | 57.2% / 64.7% | **27.6% / 27.1%** |

- **The 360° mesh is a whole room:** from the centre only 0.4% of the view is empty in any direction (the single view: 90%, and still 72% in its own direction because a 90° view is wider than the 60° image). Moving 1 m off-centre raises it only to 0.7%, because the band and the room box fill what the panorama did not see.
- **Dollhouse:** 28% of the rays through the room's bounding box miss (single view 57%). The misses are ragged room outlines (doors and windows open to outside depth, broken wall pieces at depth cuts, as in the kitchen) and the space between the bounding box and the walls.
- **Timing** (RTX 5060 Laptop, models loaded): 16.8 s per room on average (panorama 7.5 s, super-resolution 0.8 s, depth 0.9 s, segmentation 1.4 s, mesh + floor plan 5.7 s, export 0.3 s). Peak VRAM per stage: panorama 3.9 GB, DAP 2.3 GB, SegFormer 0.3 GB, Real-ESRGAN 0.2 GB. The SD 2.1 panorama model is converted once to fp16 safetensors (load time 70 s → 17 s, then cached).
- **Where it still fails:** the panorama model sometimes repeats large furniture (two beds in the coastal bedroom); thin objects are torn at depth cuts; the room box is a rectangle, so L-shaped or open-plan rooms are boxed (the mid-century dining room's box is 72 m²).
- Tables: `results/tables/room3d_empty.csv`, `room3d_empty_per_view.csv`, `room360_timing.csv`. Figures: `results/figures/room3d_views.png` (inside views, single view vs. 360°), `room3d_dollhouse.png`. The 3D files are in `outputs/room360/<room>/` (not in git).

### Floor Plan of the Full Room (milestone 5 of PLAN_V2)
Code: `src/floorplan.py` (`closed_room_360`, `room_edges`), `scripts/floorplan_360.py`.

The floor-plan code of milestone 1 runs unchanged on the spherical point cloud of the panorama (gravity from the floor plane, walls by RANSAC lines + Manhattan snapping, furniture rectangles, doors and windows). The azimuth tests were made wrap-aware for walls behind the camera. A 360° camera sees every side of the room, so the room is **closed into a rectangle** in the dominant direction, and its four sides become the walls of the plan (dimensions, area, doors and windows on them). A side with no detected wall line (e.g. a wall covered by wardrobes, which SegFormer labels as furniture) is drawn **hatched as inferred**.

**How to close the room: three rules compared.** The check that decides is the **wall agreement**: the share of wall-labelled points (0.3–2.0 m above the floor) within 30 cm of the plan's outline. It is independent of all three rules. The plan-vs-mesh floor check (convex hull of the floor triangles in the mesh ÷ plan area) must stay ≤ 1, because the room cannot be smaller than the floor that was seen. These are the same six rooms the rules were developed on, so this is a selection, not a test.

| Rule | Wall agreement (mean) | Hull ÷ plan (range) | Problem |
|---|---|---|---|
| Walls: longest wall line per side; missing side at the 98th percentile of the scene | 0.53 | 0.36–**2.38** | Picks inner lines: two plans are smaller than the floor seen (impossible) |
| Floor: floor + furniture extent per side, snapped to a wall within 0.5 m | 0.46 | 0.39–0.62 | Floor seen through doorways and open-plan areas inflates the room |
| **Walls + floor (selected):** the walls rule, but no side may cut through the observed floor (pushed out to its 95th percentile) | **0.55** | 0.36–1.08 | |

**Result on the six demo rooms** (rule "walls + floor"):

| Room | Plan (m) | Area (m²) | Inferred sides | Doors / windows | Furniture | Wall agreement | Mesh floor hull ÷ plan |
|---|---|---|---|---|---|---|---|
| Contemporary living room | 7.75 × 5.84 | 45.3 | 3 | 0 / 0 | 15 | 0.31 | 0.50 |
| Coastal bedroom | 4.56 × 5.20 | 23.7 | 2 | 1 / 1 | 8 | 0.99 | 0.38 |
| Rustic kitchen | 5.98 × 4.93 | 29.5 | 4 | 0 / 2 | 9 | 0.41 | 0.92 |
| Mid-century dining room | 9.00 × 8.00 | 72.0 | 3 | 2 / 1 | 16 | 0.40 | 0.36 |
| Art deco bathroom | 3.77 × 3.91 | 14.7 | 4 | 2 / 3 | 5 | 0.84 | 0.53 |
| Mediterranean living room | 6.22 × 4.68 | 29.1 | 4 | 1 / 0 | 14 | 0.36 | 1.08 |

- **Every 360° plan is closed** (6 of 6, against 0 of 654 single NYU views and 0 of 6 single demo views), with wall lengths, area, doors, windows and labelled furniture.
- **The bedroom and the bathroom are good plans:** 99% and 84% of the wall points lie on the outline, and the dollhouse view matches the drawing (`results/figures/floorplan_360_a_coastal_style_bedroom_s22.png`).
- **Open-plan rooms are boxed:** the dining room and the contemporary living room open into other spaces, so their rectangles (72 and 45 m²) contain more than the room. In those rooms only 31–41% of the wall points are on the outline.
- **Plan vs. mesh floor:** the convex hull of the floor that the mesh actually shows covers 36–108% of the plan area (median 51%). Below 100% is expected: furniture hides floor near the walls, and the hull of the visible floor misses the corners. The one value above 1 (1.08, Mediterranean living room) means the plan is about 8% smaller than the visible floor: a side is placed at the floor's 95th percentile, not its maximum.
- **Only 0–2 sides per room are backed by a detected wall line** (inferred: 2–4). Generated rooms are full of wardrobes, shelving and curtains, which SegFormer does not label as wall, so most sides come from the floor extent. Sizes also carry the depth scale (DAP; no ground truth here).
- Tables: `results/tables/floorplan_360.csv` (all three rules, all checks). Figures: `results/figures/floorplan_360_overview.png` (panorama | plan | dollhouse for all rooms) and one figure per room, `results/figures/floorplan_360_<room>.png` (plan next to the dollhouse). PNG and SVG plans per room: `outputs/room360/<room>/floorplan.png / .svg`.

### Layout First (milestone 6 of PLAN_V2, research experiment)
Code: `src/layout.py`, `generate_pano.generate_pano_controlled`, `scripts/eval_layout.py`.

**Idea:** like an architect, decide the plan first, then make the image obey it, so the 3D geometry is exact instead of estimated.

1. **Parametric layout** (`layout.generate_layout`, seeded): room size from realistic ranges per room type (e.g. bedroom 3.2–4.5 × 3.5–5.0 m, bathroom 2.2–3.0 × 2.4–3.5 m, ceiling 2.5–3.1 m), one door with 1 m clearance in front of it, one or two windows. Furniture are boxes placed by rules: bed against a wall with a nightstand on each side, wardrobe on another wall, desk with a chair in front; sofa facing the TV unit with the coffee table between them; a counter run with stove and sink, fridge, island if the room is wide enough; dining table in the middle with chairs around it, sideboard; bathtub, toilet and vanity against walls. Nothing may overlap (8 cm gap) or block the door. The camera is the free spot nearest the room centre, 1.4 m high. Output: JSON and a dimensioned plan.
2. **Rendering:** exact ray–box intersection from the camera gives an equirectangular distance map and an ADE20K label map (wall, floor, ceiling, door, window, furniture classes).
3. **Generation:** SD 1.5 + ControlNet depth (`lllyasviel/control_v11f1p_sd15_depth`) + the 360° LoRA, circular padding also in the ControlNet, conditioned on the inverse layout depth. (Our MMIS LoRA is left out: milestone 2 showed it does not help in panorama mode.)
4. **3D:** the panorama (Real-ESRGAN texture) is projected onto the **known geometry**: the room mesh of milestone 4 built on the layout distance and labels instead of estimated depth.
5. **Adherence:** DAP depth of the generated panorama vs. the layout depth, and SegFormer labels vs. the layout labels. Baseline: the same pipeline and seed with the ControlNet switched off.

**Evaluation:** the six demo prompts × two layouts = 12 rooms.

| Panorama | Depth AbsRel vs. layout (as predicted / median-scaled) | δ<1.25 (as predicted) | mIoU (wall / floor / ceiling / furniture) | CLIP (crops) |
|---|---|---|---|---|
| **Layout-conditioned (ControlNet)** | **0.062 / 0.033** | **0.989** | **0.49** (0.48 / 0.60 / 0.42 / 0.45) | 18.6 |
| Same seed, no ControlNet | 0.227 / 0.203 | 0.678 | 0.28 (0.28 / 0.40 / 0.37 / 0.07) | 21.2 |
| *Milestone 2, selected SD 2.1 generator (other prompts)* | – | – | – | *22.4* |

| 3D mesh (empty share) | Inside, centre | Inside, 1 m off-centre | Dollhouse |
|---|---|---|---|
| **Layout-first (exact geometry)** | **0.02%** | **0.02%** | **5.9%** |
| Milestone 4 (estimated geometry, same prompts) | 0.4% | 0.7% | 27.6% |

- **The images follow the layout closely.** The depth that DAP estimates from the generated panorama matches the layout with AbsRel 0.062, without any scale fitting (DAP's scale is right: median ratio 0.98, camera height 1.47 m vs. the true 1.40 m). Without the ControlNet the same seed gives AbsRel 0.227. Furniture lands where the boxes are (furniture IoU 0.45 vs. 0.07).
- **Ground truth for the plan pipeline.** Because the true room is known, this is the one place where the 360° floor-plan pipeline of milestones 3 + 5 (DAP + SegFormer + floor plan) can be checked: on the layout panoramas it recovers the room area with a median error of 8% (mean |error| 12%) and the side lengths within 6% on average. The bathrooms are the weak case (26% and 42% area error): small rooms where the camera is close to the walls.
- **The 3D is exact by construction** and has almost no holes: 0.02% empty from inside, 5.9% from the dollhouse views (milestone 4: 27.6%), because walls, floor and furniture are complete boxes.
- **The cost is image quality and text alignment** (negative side). CLIP falls from 21.2 to 18.6. Box-shaped conditions make box-shaped furniture (beds and tables become plain blocks), and the style words matter less: rooms look plain and similar. The SD 1.5 + 360 LoRA base is also weaker than the SD 2.1 panorama model selected in milestone 2 (no SD 2.1 depth ControlNet was tried, to stay within the plan).
- **Verdict:** adherence is good, so layout-first works as a mode for exact geometry and exact plans. It does not replace milestone 4 as the default 360° mode, which gives better-looking rooms. Both are in the app.
- Timing: layout 0.3 s, panorama 13.1 s (4.3 GB peak VRAM), mesh 2.1 s. Tables: `results/tables/layout_adherence.csv`, `layout_mesh.csv`, `layout_plan_check.csv`, `layout_per_room.csv`. Figure: `results/figures/layout_first.png` (layout | depth condition | conditioned panorama | same seed without ControlNet | plan recovered from the panorama | dollhouse on the exact geometry).

### PLAN_V3 test bench (layout-first rooms with ground truth)
Code: `scripts/make_bench.py`, `layout.render_full`. All PLAN_V3 milestones are measured on the same **40 layout-conditioned panoramas**: 5 room types × layout seeds 1–8, one style per room (8 styles). Each room has its exact layout (room box, furniture boxes, door, windows, camera), the true equirectangular distance, class and **furniture-instance** maps, the generated panorama (ControlNet depth + 360° LoRA, milestone-6 settings), the Real-ESRGAN texture, the DAP distance and the v2 SegFormer labels. **Seeds 1–4 (20 rooms) are the development set** (every choice is tuned there), **seeds 5–8 (20 rooms) the held-out set** (reported). The generated images follow the layout only approximately (milestone 6: furniture IoU 0.45), so all scores against the layout are lower bounds; v2 and v3 always run on identical panoramas. Index: `results/tables/bench_v3_rooms.csv`. Building the bench: 13 s per room.

### Instance segmentation on panoramas (milestone 1 of PLAN_V3)
Code: `src/instances.py`, `scripts/eval_instances.py`.

- **v2:** SegFormer on 14 views → one class map → connected components per furniture class. Touching objects of one class become one instance, an occluder splits one object in two, and an object crossing the panorama seam is always cut (connected components do not wrap).
- **v3:** 13 perspective views (100° FOV; 8 around the horizon at −15°, 4 looking down at −55°, 1 up) → **Mask2Former Swin-L ADE20K panoptic** (`facebook/mask2former-swin-large-ade-panoptic`, same 150 class ids as SegFormer, 0.6 GB VRAM, ~1 s per view) → every view's segment map projected back onto the panorama (each pixel from the view that sees it most centrally) → **cross-view merging**: inside the area two views share, the IoU of every pair of furniture segments is read from one contingency table; segments of the same furniture family with IoU ≥ 0.3 are joined (union-find). Score threshold and merge IoU were swept on the dev rooms (results/tables/inst_dev_sweep.csv; nearly flat, all segments score > 0.7).
- **Class map:** the pure panoptic class map was worse for room surfaces (dev ceiling IoU 0.16 vs. 0.58): Mask2Former calls the ceiling of the upward view "floor". The v3 class map is therefore a **hybrid**: walls / floor / ceiling / openings from SegFormer, furniture from the panoptic instances (`instances.hybrid_class_map`). This was decided on the dev numbers.

**Held-out rooms (20 rooms, 109 visible GT furniture boxes; IoU ≥ 0.5 matching):**

| | Pred. instances | Precision (any class) | Recall (any) | F1 (any) | F1 (same family) | Seam objects found | mIoU (wall/floor/ceiling/furniture) | Furniture IoU |
|---|---|---|---|---|---|---|---|---|
| v2 SegFormer + components | 165 | 0.14 | 0.21 | 0.17 | 0.05 | 1 / 12 | 0.585 | 0.46 |
| **v3 panoptic, merged** | 70 | **0.44** | **0.28** | **0.35** | **0.12** | **3 / 12** | **0.589** | **0.52** |
| v3 panoptic class map only | 70 | 0.44 | 0.28 | 0.35 | 0.12 | 3 / 12 | 0.452 | 0.36 |

- v3 doubles the instance F1 (dev: 0.37 vs. 0.14) mainly by **precision**: v2 produces 2.4 predictions per GT object (ceiling and floor patches labelled as furniture, objects split into fragments), v3 one clean mask per object. Per class (held-out): chairs 0.35 vs. 0.24 recall, wardrobes 0.50 vs. 0.00, toilets 0.50 vs. 0.00, beds equal (0.75).
- **What stays low and why:** recall 0.28. The generated rooms often do not contain what the layout says (a kitchen "counter run" becomes one block; armchairs and plants are drawn as something else or not at all), so "same family" scores (0.12) mostly measure the generator, not the segmenter. Doors and windows are almost never recognisable in these plain images (IoU ≈ 0 for both).
- Figure: `results/figures/inst_examples.png` (held-out: panorama | GT boxes | v2 | v3). Tables: `results/tables/inst_summary.csv`, `inst_by_class_test.csv`, `inst_per_room.csv`, `inst_dev_sweep.csv`.

### Demo modes (v2)
`app/app.py` now has three modes (radio button): **Quick (single view)**, **360° Room** and **Layout-first**. 360° and layout-first show the panorama, the floor plan (360° plan / exact layout plan), an **inside** viewer (camera at the room centre, look around, move up to 1 m) and a **dollhouse** viewer (ceiling removed, orbit from above), with `gr.Model3D` as the offline fallback. Only the active mode's large models sit on the GPU. Measured peaks: Quick 2.9 GB, Layout-first 4.6 GB, 360° 5.5 GB (6.8 GB reserved). One request: Quick 7–9 s, 360° 17.5 s, Layout-first 17.9 s. Tested headlessly in all three modes with switching back and forth; the `<model-viewer>` inside / dollhouse modes have not been checked by hand in a browser yet.

### Visual Results
- Prompt → 2D image → edges → segmentation → depth → point cloud → mesh: `results/figures/pipeline_3d_results.png`
- Floor plans of the demo rooms: `results/figures/floorplan_demo.png`; NYU plans from GT vs. predicted depth: `results/figures/floorplan_nyu_examples.png`
- 360° panoramas (configurations, seam): `results/figures/pano_configs_grid.png`; super-resolution: `pano_sr_compare.png`
- Panoramic depth, method A vs. DAP: `results/figures/pano_depth_examples.png`
- Full-room mesh vs. single view (inside views, dollhouse): `results/figures/room3d_views.png`, `room3d_dollhouse.png`
- 360° floor plans next to the dollhouse: `results/figures/floorplan_360_overview.png`, `floorplan_360_<room>.png`
- Layout-first: `results/figures/layout_first.png`
- Poisson mesh vs. textured grid mesh: `results/figures/mesh_old_vs_new.png`, `results/figures/mesh_layers.png`
- Point cloud without / with refinement and cleaning: `results/figures/pointcloud_refinement.png`
- Relative vs. metric depth in 3D: `results/figures/depth_relative_vs_metric_3d.png`
- Before/after preprocessing examples: `results/figures/depth_preprocessing_steps.png`
- Edge and segmentation overlays: `results/figures/analysis_edges_segmentation.png`

---

## 9. Tech Stack

Python · PyTorch · HuggingFace Diffusers · PEFT · Transformers · OpenCV · scikit-image · Open3D · NumPy · Matplotlib · Gradio · Hugging Face Spaces · GitHub

---

## 10. Repository Structure

```
RoomGen3D/
├── README.md
├── requirements.txt
├── notebooks/
│   └── RoomGen3D.ipynb        # Full pipeline, step by step (course submission)
├── src/
│   ├── data_cleaning.py       # Blur/exposure filtering
│   ├── preprocessing.py       # Denoise, CLAHE, sharpen, normalize
│   ├── analysis.py            # Canny, segmentation
│   ├── generate_2d.py         # SD + LoRA inference
│   ├── train_lora.py          # LoRA training
│   ├── depth.py               # Depth estimation + refinement
│   ├── reconstruct_3d.py      # Point cloud, Poisson mesh (comparison)
│   ├── textured_mesh.py       # Textured grid mesh, disocclusion filling, room planes
│   ├── floorplan.py           # Floor plan: gravity alignment, walls, furniture, doors/windows, drawing
│   ├── equirect.py            # Equirectangular geometry, perspective crops, seam ratio
│   ├── generate_pano.py       # 360° panoramas: circular padding, LoRAs / SD 2.1 panorama model, Real-ESRGAN
│   ├── pano_depth.py          # Panoramic depth: multi-view alignment (ours), DAP + wrap fix, wrap-aware filters
│   ├── room_mesh.py           # Full-room mesh from a panorama: wrap-around grid, poles, room box, dollhouse, ray casting
│   ├── pipeline_360.py        # 360° Room mode stages (used by the script and the demo)
│   ├── layout.py              # Layout-first: parametric room layouts, equirectangular depth / label rendering
│   ├── pipeline.py            # End-to-end stages (used by the script and the demo)
│   └── evaluate.py            # FID, CLIP, AbsRel, RMSE, δ
├── scripts/                   # Long jobs: training, evaluations, run_pipeline.py, figures
├── app/
│   └── app.py                 # Gradio demo (2D + 3D viewer)
├── results/                   # Figures, tables, sample outputs
└── paper/                     # IEEE paper (Word + PDF)
```

---

## 11. Work Plan

| Phase | Week | Tasks | Deliverable |
|---|---|---|---|
| 1. Proposal | ≤ Week 8 | Title, team, roles, intro, problem, datasets | **Project proposal** |
| 2. Literature | Week 8–9 | Review text-to-image, monocular depth, 3D reconstruction, image enhancement papers | Related-work notes |
| 3. Data | Week 9 | Download MMIS subset + NYU Depth V2, cleaning, preprocessing | Clean dataset |
| 4. 2D model | Week 9–10 | Train LoRA, generate samples, FID/CLIP | Trained LoRA + results |
| 5. Depth & 3D | Week 10–11 | Depth estimation, refinement, point cloud, mesh | Working 3D pipeline |
| 6. Experiments | Week 11–12 | Ablation, robustness tests, model comparisons | Tables + figures |
| 7. Demo | Week 12 | Gradio app, deploy to HF Spaces | Live demo |
| 8. Paper | Week 12–13 | IEEE-format paper, similarity check | Paper (Word + PDF) + report |
| 9. Final | Week 13 | Clean notebook, slides | **Code + paper + slides** |

---

## 12. Paper Outline (IEEE Format)

1. **Introduction:** problem, relevance to DIP, proposed solution
2. **Goals & Objectives:** measurable objectives, techniques, evaluation criteria
3. **Related Work:** text-to-image, monocular depth, 3D reconstruction, image enhancement; gaps
4. **Data Acquisition & Preprocessing:** MMIS, NYU Depth V2, cleaning, preprocessing
5. **Image Processing & Model Development:** pipeline, LoRA training, depth, 3D
6. **Results, Discussion & Visualization:** metrics, ablation, comparisons, limitations
7. **Conclusion & Future Work**

---

## 13. Objectives (Measurable)

1. Train a LoRA that improves FID and CLIP Score over vanilla SD v1.5. *Status: FID 64.5 → 54.6 and KID 19.1 → 10.2 ×10³ (met). CLIP 27.9 → 27.0 (not met; see section 8).*
2. Show that preprocessing reduces depth error (AbsRel/RMSE) on NYU Depth V2. *Status: not met. On clean images every preprocessing step is neutral or slightly worse for both models (Depth Anything AbsRel 0.0546 raw vs. 0.0563 full). The noise-gated pipeline leaves clean images unchanged, so it avoids the loss but gives no gain; see section 8.*
3. Show that preprocessing improves robustness under noise, blur, and low light. *Status: partly met, for strong noise only. The full pipeline (denoise + CLAHE + unsharp) is worse than raw in 12 of 14 degraded cases. The noise-gated pipeline, selected on the NYU train split and evaluated once on the test split, recovers 49% (σ=25) and 59% (σ=50) of the AbsRel damage for Depth Anything and 14% / 16% for MiDaS. It changes nothing under σ=15 noise, motion blur and moderate low light, and is mixed in severe low light (+6% Depth Anything, −3% MiDaS).*
4. Produce a 3D mesh from any generated design in under ~1 minute on a GPU. *Status: met. 5.4 s per prompt on average (max 6.0 s) over six prompts, 2.7 GB peak VRAM; see section 8.*
5. Deploy a public demo showing 2D and 3D outputs side by side. *Status: the Gradio demo runs locally (`app/app.py`, three modes); not deployed yet.*

Version 2 (PLAN_V2.md):

6. Replace the depth heat map by a measured floor plan. *Status: met. Plans with walls, doors, windows, furniture and dimensions in all modes; on NYU the plan from predicted depth has walls 16% too long (median) and furniture IoU 0.25 against the plan from ground-truth depth, 0.51 after fixing the global scale.*
7. Build the whole room, not a relief. *Status: met. 360° mode: 0.4% of the view empty from the room centre (single view 89.7%), 0.7% at 1 m off-centre; 16.8 s per room, 5.5 GB peak VRAM. Every 360° floor plan is closed (6/6). Our MMIS LoRA does not carry over to panoramas (negative result), so the 360° mode uses another generator.*
8. Make the geometry exact by generating the layout first. *Status: met as an experiment. The generated panoramas follow the layout (depth AbsRel 0.062, furniture IoU 0.45, vs. 0.227 and 0.07 without conditioning) and the 3D is exact, at the cost of image quality (CLIP 21.2 → 18.6). Kept as a separate mode; the 360° mode stays the default.*

---

## 14. Limitations

- In Quick mode the 3D comes from a **single view**, so hidden areas (behind furniture, other walls) are not reconstructed. The 360° Room mode builds the whole room, but from one viewpoint: surfaces hidden from the panorama centre are only filled by the background band and the rectangular room box.
- The **floor plan** of a single view is never a closed room (0 of 654 NYU views); furniture behind the visible face is completed from class priors, and sizes carry the depth model's scale error (wall lengths +16% median on NYU). The 360° plan is always closed, but as a rectangle: open-plan and L-shaped rooms are boxed, and most sides are inferred from the floor extent because walls hidden behind furniture are not detected.
- An **isometric cutaway mode** (prompting for a corner view from above on a plain white background, so that fewer regions are hidden) was tried with base SD v1.5 and with the LoRA at strength 1.0 and 0.5 on the six demo prompts and dropped: only four of six images got the corner view, five of six had no plain background to remove, and the room style was mostly lost.
- Depth from generated images has **no ground truth**, so depth accuracy is measured on NYU Depth V2 only; for panoramas only plausibility checks (and the layout-first adherence) are available, since no licensed real RGB-D panorama set was used.
- The **360° mode does not use our MMIS LoRA** (it did not help on panoramas), and the panorama model sometimes repeats large furniture. **Layout-first** gives exact geometry but box-shaped furniture and weaker styles.
- The 360° room box and plan are rectangles; open-plan and L-shaped rooms are boxed too large.
- Training on a subset of MMIS due to compute limits.
- The 3D stage uses a metric depth model, but its absolute scale is only approximate (AbsRel 0.22 without alignment on NYU), and the camera field of view (60°) is assumed, so sizes in the 3D model are rough.
- Classical preprocessing did **not** improve depth accuracy on clean NYU Depth V2 images: modern depth networks are already robust to moderate noise, blur and low light. Only noise-gated denoising helps, and only under strong noise (σ ≥ 25); nothing tested helps against blur. Depth refinement gives a cleaner point cloud (fewer flying pixels) but not a lower depth error.

## 15. Future Work

- A panorama LoRA trained on MMIS-style interiors (our 2D LoRA does not transfer to equirectangular images)
- Text-guided editing of individual furniture items
- Metric-scale depth for real measurements
- AR/VR export

---

## 16. Submission Checklist (Week 13)

- [ ] Paper (Word + PDF) in IEEE format, template source cited in footnote
- [ ] Plagiarism/similarity report
- [ ] Jupyter notebook: clean, commented, logical order
- [ ] PowerPoint slides
- [ ] Project title + team members on every file
