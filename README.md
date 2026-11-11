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
[2] Preprocessing ───── Denoising · CLAHE · Sharpening · Resize/Normalize
   │
   ▼
[3] Image Analysis ──── Canny edges · Room segmentation (walls / floor / furniture)
   │
   ▼
[4] Depth Estimation ── Depth Anything (compared with MiDaS)
   │
   ▼
[5] Depth Post-processing ── Bilateral / median filtering · edge-aware refinement · hole filling
   │
   ▼
[6] 3D Reconstruction ─ Depth → Point Cloud → Mesh (Open3D, Poisson reconstruction)
   │
   ▼
[7] Interface ───────── Gradio: 2D image + interactive 3D viewer + download (.ply / .obj)
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
| Depth estimation | Depth Anything (main), MiDaS (baseline) | Monocular depth from a single image |
| Segmentation | Pretrained semantic segmenter (e.g., SegFormer on ADE20K) | Room-element masks |
| 3D | Open3D | Point cloud + mesh reconstruction |

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

Depth Anything has 44% lower AbsRel than MiDaS and is the model used in the pipeline.

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

### Visual Results
- Prompt → 2D image → depth map → 3D mesh (side by side)
- Before/after preprocessing examples
- Edge and segmentation overlays

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
│   ├── reconstruct_3d.py      # Point cloud + mesh
│   └── evaluate.py            # FID, CLIP, AbsRel, RMSE, δ
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
2. Show that preprocessing reduces depth error (AbsRel/RMSE) on NYU Depth V2. *Status: not met. On clean images every preprocessing step is neutral or slightly worse for both models (Depth Anything AbsRel 0.0546 raw vs. 0.0563 full); see section 8.*
3. Show that preprocessing improves robustness under noise, blur, and low light. *Status: not met for the full pipeline (worse than raw in 12 of 14 degraded cases). Partly met for denoising alone under strong noise: it recovers 46% of the AbsRel damage at σ=50 and 24% at σ=25 for Depth Anything.*
4. Produce a 3D mesh from any generated design in under ~1 minute on a GPU.
5. Deploy a public demo showing 2D and 3D outputs side by side.

---

## 14. Limitations

- 3D comes from a **single view**, so hidden areas (behind furniture, other walls) are not reconstructed. It is not a full 360° room.
- Depth from generated images has **no ground truth**, so depth accuracy is measured on NYU Depth V2 only.
- Training on a subset of MMIS due to compute limits.
- Monocular depth is **relative**, not true metric scale.
- Classical preprocessing did **not** improve depth accuracy on NYU Depth V2: modern depth networks are already robust to moderate noise, blur and low light. Depth refinement gives a cleaner point cloud (fewer flying pixels) but not a lower depth error.

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
