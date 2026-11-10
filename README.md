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

---

## 8. Evaluation

### 2D Generation
| Metric | Measures |
|---|---|
| FID | Realism vs. real MMIS images |
| CLIP Score | How well the image matches the text prompt |

Comparison: **Vanilla SD v1.5 vs. SD v1.5 + our LoRA**.

### Depth / 3D
| Metric | Measures |
|---|---|
| AbsRel | Mean relative depth error |
| RMSE | Absolute depth error |
| δ < 1.25 | % of pixels with accurate depth |

Comparison: **MiDaS vs. Depth Anything** on NYU Depth V2.

### Key Experiment: Effect of Image Processing (Ablation)
Run depth estimation on NYU Depth V2 under each condition and compare metrics:

| Condition | AbsRel | RMSE | δ<1.25 |
|---|---|---|---|
| Raw images | – | – | – |
| + Denoising | – | – | – |
| + CLAHE | – | – | – |
| + Full preprocessing | – | – | – |
| + Full preprocessing + depth refinement | – | – | – |

Also test robustness by adding **synthetic noise, blur, and low light** to images, then measuring how much preprocessing recovers.

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

1. Train a LoRA that improves FID and CLIP Score over vanilla SD v1.5.
2. Show that preprocessing reduces depth error (AbsRel/RMSE) on NYU Depth V2.
3. Show that preprocessing improves robustness under noise, blur, and low light.
4. Produce a 3D mesh from any generated design in under ~1 minute on a GPU.
5. Deploy a public demo showing 2D and 3D outputs side by side.

---

## 14. Limitations

- 3D comes from a **single view**, so hidden areas (behind furniture, other walls) are not reconstructed. It is not a full 360° room.
- Depth from generated images has **no ground truth**, so depth accuracy is measured on NYU Depth V2 only.
- Training on a subset of MMIS due to compute limits.
- Monocular depth is **relative**, not true metric scale.

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
