# RoomGen3D v6 plan: scene first, image second

## Owner's verdict on v5: still bad

Root cause: v5 generates an image and then tries to infer the room from it. v6 reverses the order, as in
Ctrl-Room (Fang et al., arXiv:2310.03602): **layout → proxy 3D scene → depth-conditioned generation**.
Geometry, sizes, furniture, floor plan and camera are then known by construction instead of estimated.

## Ground rules

- Same as PLAN_V5 (CLAUDE.md, 8 GB VRAM, one GPU job at a time, free every model before loading the next,
  check free RAM before heavy steps, resumable jobs, negative results recorded).
- Keep at least 20 GB free on C:. Download budget 10 GB, each download listed in PROGRESS.md.
- v5 stays callable behind a toggle for before/after on the same prompts and seeds.
- Tune on development rooms only; report on held-out and demo rooms (room sets of PLAN_V5).
- Commit + push + PROGRESS.md entry after each milestone and after every review round.
- The owner is asleep: decide, log decisions in PROGRESS.md. Do not start the final course notebook.

## Milestone 0: free disk space

Delete the unused / approved items (list and sizes in PROGRESS.md), keep ≥ 20 GB free during the run.

## Milestone 1: scene (layout + real furniture + proxy mesh)

- `src/layout_v6.py`: prompt → room type + style; room size from per-type ranges (ceiling 2.6-3.0 m);
  furniture per room type as oriented pieces (centre, footprint, height, yaw = facing direction) with
  rules: wall-aligned or aligned to the main piece (bed against a wall + nightstands, dining table
  centred + chairs around it facing it, sofa facing a TV unit, cabinets against walls), no overlaps,
  walking clearance, one door and 1-2 normal-sized windows. Deterministic per seed, varied across seeds.
  Unit tests (`tests/test_layout_v6.py`): overlaps, wall alignment, clearance, determinism.
- `src/assets_v6.py` + `scripts/download_assets_v6.py`: CC0 library (Kenney Furniture Kit, Poly Haven
  models API, another CC0 source if a class is missing), licences in `assets/LICENSES.md`, models
  normalised (y up, origin on the floor, front = +z), several models per class, style-aware choice,
  scaled to the footprint with ≤ ~20 % proportion distortion.
- `src/scene_v6.py`: room shell with real door / window openings and frames + placed assets;
  CPU ray casting (Open3D RaycastingScene) of depth, normals, instance ids, equirectangular and perspective.
- Metrics: rule violations per room, asset coverage per class, render time.

## Milestone 2: panorama on the scene

- SDXL + 360Redmond LoRA + depth conditioning from the proxy equirectangular depth at 2048 × 1024 with
  circular padding. Candidates: `diffusers/controlnet-depth-sdxl-1.0-small`, `-mid`, then TencentARC
  T2I-Adapter depth SDXL. Model CPU offload, VAE tiling. Sweep conditioning scale and end step.
- Panorama camera: point of maximum clearance at 1.5 m height.
- Style words expanded into interior descriptors (materials, palette, furniture style); negative prompt
  against glass walls / floor-to-ceiling windows; every prompt asserted within 77 CLIP tokens.
- Gates (dev): depth agreement (estimate on the generated panorama vs proxy depth), line straightness
  in perspective views, per-instance presence (segmentation IoU vs rendered instance ids), KID, CLIP,
  BRISQUE, sharpness, seam continuity, time, VRAM, RAM, all against v5.
- Fallback: SD 2.1 depth ControlNet layout-first path.

## Milestone 3: design shots from chosen cameras

- Cameras in the scene: hero from a corner or doorway at 1.5 m, 70-80° FOV, level, framing the main
  piece; one view per wall from the opposite side. Validity from rendered instance ids: main piece
  15-50 % of the frame, nothing nearer than 0.8 m covering > 15 %, floor < 35 %.
- Per shot: z-buffered reprojection of the panorama through the proxy geometry as init image (holes
  inpainted), SD 1.5 + MMIS LoRA + depth ControlNet on the exact rendered depth, img2img; test IP-Adapter
  with the panorama crop; sweep strength; Real-ESRGAN to 1536 × 1152.
- Choose on dev with consistency as a hard constraint (colour / LPIPS vs the reprojection on co-visible
  pixels, depth and edge agreement), then BRISQUE, KID vs MMIS hold-out, CLIP; with / without the LoRA.

## Milestone 4: dollhouse, top-down, floor plan from the scene

- Dollhouse = the proxy scene, textured from the panorama where visible; unseen regions by texture
  synthesis from the seen part of the same plane; real openings; furniture = asset meshes coloured by
  projection on visible faces, the instance's dominant colours elsewhere.
- Top-down: orthographic render ≥ 1024 px with outlines, oriented like the plan.
- Floor plan drawn from the layout: class symbols, label collision avoidance, shot cones H, H2, A-D in
  one style at true positions, overall dimensions + labelled extras only, scale bar proportional.
- Metrics: dollhouse texture quality v5 vs v6, furniture count vs layout, plan / shot agreement.

## Milestone 5: app and run-time

- 360 mode defaults to v6 (v5 toggle); ≤ 150 s per room in a fresh process; time, peak VRAM, peak RAM.
- Lazy loading, each pipeline once per request, offline mode once models are local (clear message if
  a model is missing). Tabs: Inside, Dollhouse, Top-down, Free orbit (panorama duplicate tab and
  walkable-mesh tab removed from the UI, code kept). Viewer scripts bundled locally.
- Labels without "v3 / v4", mode-specific options only in their mode, Latin digits in number fields
  in an Arabic-locale browser, Mask2Former MISSING weights fixed or explained, no token-truncation warnings.
- App test for Quick, v6 and v5 must pass.

## Review loop (mandatory, no maximum)

Rooms: the six demo rooms (incl. owner rooms dining seed 44 and coastal bedroom seed 22). For each:
headed browser screenshots of hero, every wall shot, Inside in four directions, Dollhouse from two
angles, Top-down, floor plan; every screenshot looked at full size; every issue below marked with the
screenshot that proves it. An issue is "cannot fix" only after three genuinely different approaches,
measured and written in PROGRESS.md, plus a web search for alternatives.

## Acceptance list

Status: OPEN / FIXED (round, evidence) / CANNOT FIX (evidence).

| # | Area | Issue | Status | Evidence |
|---|---|---|---|---|
| 1 | Inside | straight lines are curved (window frames, ceiling, curtain rod) | OPEN | |
| 2 | Inside | chairs deformed, legs melted, merged | OPEN | |
| 3 | Inside | table irregular and oversized | OPEN | |
| 4 | Inside | camera low (0.95-1.05 m) and touching furniture | OPEN | |
| 5 | Inside | floor tiles warped | OPEN | |
| 31 | Inside | furniture melted into blobs | OPEN | |
| 33 | Inside | horizon tilted, outside view differs between windows | OPEN | |
| 34 | Inside | almost every wall is glass; "coastal" read as a sea view instead of a style | OPEN | |
| 6 | Selection | candidate chosen on a trivial boxiness difference | OPEN | |
| 7 | Measure | room size wrong (7 × 7 m, 3.86 m ceiling; depth sees through glass) | OPEN | |
| 8, 38 | Measure | boxiness low | OPEN | |
| 9, 39 | Measure | only 2 of 5 shots pass the distance test | OPEN | |
| 10 | Plan | furniture at random angles, not wall-aligned | OPEN | |
| 11 | Plan | table drawn huge over a chair and the camera; second table under two chairs | OPEN | |
| 12 | Plan | cabinets floating mid-room | OPEN | |
| 13 | Plan | furniture clustered, rest empty | OPEN | |
| 14 | Plan | labels overlap | OPEN | |
| 15 | Plan | hero cone drawn differently, H2 missing | OPEN | |
| 16 | Plan | unexplained dimension lines; scale bar too short | OPEN | |
| 17 | 2D shots | shot does not match the panorama (carpet vs tiles, different windows) | OPEN | |
| 18 | 2D shots | broken furniture (orphan chair leg, missing parts, chair melted into table) | OPEN | |
| 19 | 2D shots | nonsense wall decor, distorted mirror | OPEN | |
| 20 | 2D shots | camera low, floor fills half the frame, furniture cropped at the sides | OPEN | |
| 24, 37 | Dollhouse | only 2 furniture pieces found in a bedroom | OPEN | |
| 25 | Dollhouse | furniture is plain boxes with broken texture | OPEN | |
| 26 | Dollhouse | floor is inpainting patches and a stretched light streak | OPEN | |
| 27 | Dollhouse | walls stretched, windows small, deformed, misplaced | OPEN | |
| 28 | Dollhouse | large grey-blue fills | OPEN | |
| 29 | Top-down | blurry unreadable blob | OPEN | |
| 30 | Top-down | title says v3 / v4 | OPEN | |
| 21 | UI | button labels say v3 / v4 | OPEN | |
| 22 | UI | Quick-only option shown in 360 mode | OPEN | |
| 23 | Run | 221-237 s per room | OPEN | |
| 35 | UI | "360 panorama" tab duplicates "Inside" | OPEN | |
| 36 | UI | walkable mesh tears even at 5 cm | OPEN | |
| 40 | UI | CFG number renders as Arabic-Indic digits ("V.o") | OPEN | |
| 41 | Run | Quick models load at startup even for 360 mode | OPEN | |
| 42 | Run | prompt exceeds 77 CLIP tokens and is truncated | OPEN | |
| 43 | Run | Mask2Former loads with MISSING weights (swin layernorm) | OPEN | |
| 44 | Run | unauthenticated Hub requests on every start; SD 1.5 loaded three times per request | OPEN | |
