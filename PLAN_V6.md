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

Status after review round 13 (rounds 1-13 in PROGRESS.md). Evidence: headed-Chromium sheets
`results/figures/v6_review_r13_<room>.png` (every shot, Inside at 0 / 90 / 180 / 270 deg, dollhouse from two angles,
top-down, plan) for all six demo rooms, which include the owner rooms (dining seed 44, coastal bedroom seed 22);
before / after `results/figures/v6_before_after_<room>.png`; numbers in `results/tables/v6_summary.csv`.

| # | Area | Issue | Status | Evidence |
|---|---|---|---|---|
| 1 | Inside | straight lines are curved | FIXED (r1) | Inside views of all rooms: window frames, ceiling edges, curtain rods straight (a straight rod is part of the proxy); straight-line share demo 0.34 (v5) -> 0.49 |
| 2 | Inside | chairs deformed, legs melted, merged | FIXED (r1) | dining 44 inside_0, H, H2: eight separate chairs with legs; proxy uses real chair models |
| 3 | Inside | table irregular and oversized | FIXED (r1) | one rectangular 1.9-2.0 m table sized to the room (layout rule), dining 44 inside_0 / plan; weak conditioning split it in two (sweep) -> scale 0.9 |
| 4 | Inside | camera low and touching furniture | FIXED (r1) | camera 1.5 m, >= 0.5 m from furniture and walls (unit test test_camera_clearance) |
| 5 | Inside | floor tiles warped | FIXED (r1) | kitchen 33 / Mediterranean 66 Inside and shots: regular terracotta tiles |
| 31 | Inside | furniture melted into blobs | FIXED (r1) | no blobs in any Inside view; instance presence demo 64 % |
| 33 | Inside | horizon tilted, outside view differs | FIXED (r6) | level viewer and level panorama; outside views all greenery; windows repaired in place (window presence 1.0) |
| 34 | Inside | glass walls; "coastal" read as sea view | FIXED (r1) | window share generated 0.035 vs proxy 0.046 (demo); coastal 22 = white oak, linen, a sea PICTURE, no sea view |
| 6 | Selection | candidate chosen on trivial boxiness | FIXED (design) | no candidate search in v6: the scene is designed first |
| 7 | Measure | room size wrong | FIXED (design) | dining 44 is 3.92 x 5.02 m, ceiling 2.71 m by construction (plan title) |
| 8, 38 | Measure | boxiness low | FIXED (design) | the room is the proxy box; panorama depth delta1 0.99 vs the proxy (demo) |
| 9, 39 | Measure | only 2 of 5 shots pass | FIXED (r1) | 36 / 36 shots valid (H, H2, A-D in six rooms) |
| 10 | Plan | random angles | FIXED (r1) | yaw in {0, 90, 180, 270} by construction (unit test) |
| 11 | Plan | table over chair and camera; second table | FIXED (r1) | plan drawn from the layout; no overlaps in 1400 layouts |
| 12 | Plan | cabinets floating mid-room | FIXED (r1) | cabinets / wardrobes / shelves have a wall (wall_align test) |
| 13 | Plan | furniture clustered | FIXED (r1) | groups spread by rules; living-room ranges narrowed (4.0-5.2 x 4.4-5.8 m) |
| 14 | Plan | labels overlap | FIXED (r3) | label collision avoidance + camera markers >= 0.6 m apart; all r13 plans |
| 15 | Plan | hero cone different, H2 missing | FIXED (r1) | H, H2, A-D same cone style at true positions |
| 16 | Plan | unexplained dimension lines; scale bar short | FIXED (r2) | overall dimensions only; 2 m scale bar from 3.5 m width (~half the room) |
| 17 | 2D shots | shot does not match the panorama | FIXED (r6) | shots are the panorama reprojected through the proxy, then re-rendered (dLab 4.6, LPIPS 0.15 on co-visible pixels); windows / floors match across shots and Inside |
| 18 | 2D shots | broken furniture | FIXED (r1), see weak points | no orphan legs / melted chairs in any r13 shot; small pieces (toilet, stove, vanity) are drawn poorly or not recognised (plan / shot agreement 73 %) |
| 19 | 2D shots | nonsense wall decor, distorted mirror | FIXED (r10) | wall art proxies give plausible pictures; removed from kitchens (they became extra windows / niches); no mirrors |
| 20 | 2D shots | camera low, floor half the frame, cropped | FIXED (r1), see weak points | 1.5 m, level with lens shift, floor < 35 %; dense dining room still cuts foreground chairs at the frame edge (border penalty tried and reverted, r12) |
| 24, 37 | Dollhouse | only 2 pieces in a bedroom | FIXED (r2) | furniture count = layout in 6 / 6 rooms (coastal bedroom: 6 pieces) |
| 25 | Dollhouse | plain boxes, broken texture | FIXED (r2) | CC0 furniture models coloured from the panorama, smooth vertex colours |
| 26 | Dollhouse | inpainting patches, light streak | FIXED (r4) | nadir and highlights excluded, quilted floor colour-matched; r13 dollhouses |
| 27 | Dollhouse | walls stretched, windows small / misplaced | FIXED (r2) | windows and door are the layout's real openings with frames and curtains |
| 28 | Dollhouse | grey-blue fills | FIXED (r2) | unseen wall = the wall's own low-frequency colour |
| 29 | Top-down | blurry blob | FIXED (r5) | orthographic render >= 1024 px both sides, furniture outlines, plan orientation |
| 30 | Top-down | title says v3 / v4 | FIXED (M5) | tab label "Top view (orthographic, same orientation as the floor plan)" |
| 21 | UI | labels say v3 / v4 | FIXED (M5) | `results/figures/v6_ui_360_arabic_locale.png`: "Scene first ... (default)" / "Image first ... (previous version)" |
| 22 | UI | Quick-only option in 360 mode | FIXED (M5) | same screenshot: no Steps / CFG / hires in 360 mode |
| 23 | Run | 221-237 s per room | FIXED (M5) | fresh process 128 s (dining 44), 136 s with window repair (coastal 22); `v6_app_test.csv` |
| 35 | UI | duplicate panorama tab | FIXED (M5) | tabs Inside / Dollhouse / Top-down / Free orbit |
| 36 | UI | walkable mesh tears | FIXED (M5) | tab removed from the UI (code kept) |
| 40 | UI | CFG in Arabic-Indic digits | FIXED (M5) | `v6_ui_quick_arabic_locale.png` (ar-EG locale: 22, 30, 7.5) vs `..._before_fix.png` |
| 41 | Run | Quick models load at startup | FIXED (M5) | app test: startup_models empty; nothing loads until a request |
| 42 | Run | prompt > 77 tokens truncated | FIXED (M5) | v6 prompts asserted <= 77; v5 prompts cut at a word boundary; 0 truncation warnings in all app tests |
| 43 | Run | Mask2Former MISSING weights | FIXED (proof) | the layer is unused by SwinBackbone: random weights change logits by 0.0 (`v6_mask2former_missing_weights_check.json`); report silenced |
| 44 | Run | Hub requests; SD 1.5 loaded 3x | FIXED (M5) | offline mode + local snapshot paths: 0 Hub lines; one SDXL + one SD 1.5 load per v6 request |
