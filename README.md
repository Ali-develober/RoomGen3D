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
[7] Interface ───────── Gradio: 2D image + interactive 3D viewer + download (.ply / .glb)
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
`app/app.py` is a Gradio app: prompt, seed, steps and CFG in; 2D image, depth map and edge + segmentation overlay out, with download buttons for the image, the mesh (`.glb`) and the point cloud (`.ply`). The 3D model is shown in two tabs:

- **Original view (limited orbit)**, the default: a `<model-viewer>` element that opens exactly at the camera of the generated image (60° field of view) and limits the orbit to ±30° left/right, so the first impression matches the image and the parts a single view cannot know stay out of sight. Its script is loaded from a CDN, so this tab needs an internet connection.
- **Free orbit**: `gr.Model3D` without limits (works offline). `gr.Model3D` has no setting for orbit limits, which is why the first tab uses a different viewer.

The three models stay loaded together (2.9 GB VRAM); a request takes about 5 s after the first one. Launch from the project root and open http://127.0.0.1:7860:

```
.venv\Scripts\python.exe app\app.py
```

It is not deployed to Hugging Face Spaces yet.

### Visual Results
- Prompt → 2D image → edges → segmentation → depth → point cloud → mesh: `results/figures/pipeline_3d_results.png`
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
5. Deploy a public demo showing 2D and 3D outputs side by side. *Status: the Gradio demo runs locally (`app/app.py`); not deployed yet.*

---

## 14. Limitations

- 3D comes from a **single view**, so hidden areas (behind furniture, other walls) are not reconstructed. It is not a full 360° room.
- An **isometric cutaway mode** (prompting for a corner view from above on a plain white background, so that fewer regions are hidden) was tried with base SD v1.5 and with the LoRA at strength 1.0 and 0.5 on the six demo prompts and dropped: only four of six images got the corner view, five of six had no plain background to remove, and the room style was mostly lost.
- Depth from generated images has **no ground truth**, so depth accuracy is measured on NYU Depth V2 only.
- Training on a subset of MMIS due to compute limits.
- The 3D stage uses a metric depth model, but its absolute scale is only approximate (AbsRel 0.22 without alignment on NYU), and the camera field of view (60°) is assumed, so sizes in the 3D model are rough.
- Classical preprocessing did **not** improve depth accuracy on clean NYU Depth V2 images: modern depth networks are already robust to moderate noise, blur and low light. Only noise-gated denoising helps, and only under strong noise (σ ≥ 25); nothing tested helps against blur. Depth refinement gives a cleaner point cloud (fewer flying pixels) but not a lower depth error.

## 15. Future Work

- Multi-view generation for full 360° rooms
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
