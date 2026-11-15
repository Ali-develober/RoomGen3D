# RoomGen3D v5 plan: outputs that look good to a person

## Owner's verdict on v4 (app screenshots, dining room, seed 44): very bad

1. The hero 2D shot is a tilted close-up of plates on the table; textures are painterly mush. Real-ESRGAN sharpened edges but added no real detail, because the source is a low-resolution SD panorama.
2. The panorama camera sits in the middle of the furniture in a small room (3.09 × 2.82 m), so every crop is a close-up.
3. The inside 3D view is torn: chairs smeared into ribbons, floor is noise, holes behind furniture. A single-viewpoint depth mesh cannot survive camera translation.

**Goal**: outputs that look good to a person, judged on the six demo rooms. Numbers are still required, but a result that measures well and looks bad is a failure.

## Ground rules

- Same as PLAN_V4: CLAUDE.md, 8 GB VRAM, one model on the GPU at a time, resumable foreground jobs, negative results recorded, commit + push + `PROGRESS.md` entry after each milestone, v4 stays callable for before/after on the same seeds. The owner is asleep: decide, and log decisions in `PROGRESS.md`.
- RAM is 16 GB (≈ 9 GB free with the desktop running): free every model (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) before loading the next, never two pipelines in RAM, `enable_model_cpu_offload` + VAE tiling for SDXL, check free RAM before each heavy step and wait/retry instead of crashing (`src/memory.py`).
- Downloads pre-approved up to 15 GB, each listed in `PROGRESS.md`. Keep 10 GB free on C:. Intermediate outputs of this round may be deleted; never delete source, `results/`, `checkpoints/lora_mmis`, cleaned data, NYU, docs, notebooks.
- Do not start the final course notebook.

## Rooms (fixed before any run)

| Set | Prompts | Seeds | Use |
|---|---|---|---|
| development | `a {scandinavian, farmhouse} style {bedroom, living room, kitchen, dining room, bathroom}` (10) | 701 (+702, 703 as candidates) | all tuning and selection |
| held-out | `a {industrial, bohemian} style {…same five rooms…}` (10) | 701 (+702, 703) | reported only |
| demo | the six demo prompts and seeds of `src/pipeline.DEMO_PROMPTS` | as in v4 | reported + screenshots + self-review |

Reference set for KID/FID: the 500-image MMIS hold-out used since PLAN_V2. Perspective views for image metrics: 8 level views per panorama (yaw 0, 45, …, 315°, 90° FOV, 512 × 384).

## Milestone 1: better panorama source (SDXL)

1. `src/generate_pano_xl.py`: SDXL base 1.0 fp16 (+ fp16-fix VAE) with a 360° equirectangular adapter, 2048 × 1024, circular padding in the UNet and the VAE decoder (VAE tiled decode on a wrap-padded latent), `enable_model_cpu_offload`.
   - Candidate A: `artificialguybr/360Redmond` LoRA (bigscience-openrail-m; trigger "360, 360 view").
   - Candidate B (second try): `ProGamerGov/sdxl-360-diffusion` (MIT; full SDXL UNet fine-tuned on 2048 × 1024 equirectangular images, trigger "equirectangular 360 view").
   - Smoke test first (one prompt): loads, fits in 8 GB VRAM / 16 GB RAM, time per panorama.
2. Compare against the current generator (SD 2.1 panorama model + one-room template, 1024 × 512 → Real-ESRGAN 2048) on the same prompts and seeds (dev + held-out + demo, seed 701 / demo seeds): KID/FID of perspective views vs the MMIS hold-out, CLIP score, Laplacian sharpness and BRISQUE of the views at their native detail, boxiness, seam continuity, time, peak VRAM, peak RAM.
3. Decision rule (fixed now): drop any config that fails reliability (OOM / crash / RAM > 15 GB) or has seam ratio > 1.5. Among the rest, mean rank on dev of KID, CLIP, BRISQUE, sharpness, boxiness; ties → faster. If SDXL cannot run reliably: fall back to the current generator at its native size + per-view re-rendering (milestone 3) as the detail source.

## Milestone 2: camera and room size

1. Prompt: add "spacious room, wide angle, photographed from a corner near the wall at eye height, furniture against the walls, open floor in the middle"; negative: "close-up, macro, cluttered, cramped, furniture in front of the camera, tabletop view". Measure fitted room area and nearest-furniture distance before/after (same seeds, dev).
2. Candidate selection: score = boxiness + w_a · min(area, 25 m²)/25 + w_d · min(d_near, 1.5 m)/1.5; reject candidates with d_near < 0.5 m (camera inside or touching furniture) unless all are rejected. d_near = 5th percentile of the horizontal distance to furniture pixels (instance / class map, below eye level). Weights chosen on dev from a small grid by the share of valid shots (item 3).
3. Shot validity: level horizon (pitch = roll = 0 in the room frame), 85–90° horizontal FOV, camera at the panorama height (re-rendering from 1.5 m needs geometry that tears — not done). A view is valid if the main piece covers 15–50 % of the frame, median depth > 1.5 m and content nearer than 0.8 m covers < 15 % of the frame. Hero = best valid view (by the v4 score); if none is valid, the least bad one, and the caption says so.
4. Metric: share of valid shots and share of rooms with a valid hero, v4 vs v5.

## Milestone 3: real detail in the 2D design shots

1. Re-render each design shot: img2img on the perspective crop with SD 1.5 + MMIS LoRA + ControlNet depth (`lllyasviel/control_v11f1p_sd15_depth`, already downloaded) conditioned on the crop's own depth (rendered from the panorama distance), at 768 × 576, strength swept 0.35 / 0.45 / 0.55 / 0.65, then Real-ESRGAN to 1536 × 1152. Same with SDXL img2img (+ depth agreement only, no SDXL ControlNet download) if milestone 1 adopts SDXL.
2. Choose on dev by mean rank of BRISQUE, sharpness, CLIP, KID vs MMIS hold-out, under a structure guard (edge F1 vs the crop ≥ 0.6 and depth AbsRel of re-estimated depth vs crop depth ≤ 0.15). Report with and without our LoRA.
3. Gallery shows the re-rendered shots; raw crops in the Analysis section.

## Milestone 4: inside view without tearing

1. Default Inside viewer = panorama sphere viewer: the 4096 × 2048 panorama on a sphere, camera fixed at the centre, rotation + FOV zoom only, opens level facing the hero direction, no tone mapping, self-contained WebGL page (no CDN, works offline).
2. The depth-mesh inside view stays as a tab "Experimental: walkable mesh" with translation limited to the radius where holes stay under a measured threshold (hole share vs translation measured on dev).
3. Headed browser screenshots of the new Inside view for the six demo rooms.

## Milestone 5: clean dollhouse

1. Furniture removed from wall and floor textures before projection: dilated instance masks in each plane's rectified texture space, inpainted (compare cv2 Telea, LaMa if it fits, SD inpainting with a plain floor / wall prompt), then edge-preserving denoising of plane textures.
2. Furniture stays as blocks: correct footprint / height / orientation, flat-shaded average colour on faces not seen by the camera instead of stretched texture.
3. Metric: no-reference quality (BRISQUE, residual furniture share, noise level) of floor / wall textures v4 vs v5, plus screenshots.

## Milestone 6: Quick mode hires pass

512 generation with the LoRA → img2img at 1024 (strength 0.35–0.45) with the LoRA → optional Real-ESRGAN. Sharpness, BRISQUE, CLIP, time, VRAM vs the current output (dev prompts for the strength, held-out prompts for the report).

## Self-review loop (before the final commit)

For each demo room: headed browser screenshots of hero shot, gallery, Inside view, Dollhouse, floor plan. Check by eye: hero shows the room and its main furniture from a distance, level horizon; no painterly mush at 100 %; Inside view has no torn / stretched surfaces; dollhouse floor and walls clean; plan agrees with the shots. Fix and repeat up to three rounds; record what still fails.

## Final

- App defaults to v5 with a v4 toggle; app test passes; time per room, peak VRAM, peak RAM.
- One before/after figure per demo room (v4 vs v5: hero, Inside, dollhouse) in `results/figures`, one summary table of every metric in `results/tables`.
- README and PROGRESS.md updated, commit and push, short summary: what improved, what did not, what is still weak.
