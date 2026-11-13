# RoomGen3D v2 plan: from a 2.5D relief to a full room with a floor plan

## Why this plan exists

The current 3D output is built from one perspective image with a 60° field of view. That can only ever be a relief of what the camera saw (2.5D), not a room. Comparable published systems all avoid this in one of three ways:

| System | How it gets a whole room |
|---|---|
| Text2Room (ICCV 2023) | Moves a virtual camera, inpaints the newly visible holes with a text-to-image model, estimates and aligns depth, fuses everything into one mesh |
| DreamScene360 (ECCV 2024) | Generates one 360° panorama, splits it into 20 perspective views, estimates depth per view, aligns the depths globally, back-projects to 3D |
| HunyuanWorld 1.0 (2025) | 360° panorama as the scene proxy, then layered depth and mesh reconstruction |
| Ctrl-Room (3DV 2025) | Generates the room layout first (walls + furniture boxes), then a panorama conditioned on that layout with ControlNet, then the mesh |

Common factors: 360° coverage, and in the strongest results a layout that exists before the image does.

The owner also wants the depth "heat map" replaced in the app by an architect-style floor plan with dimensions and furniture positions.

## Ground rules

- Follow CLAUDE.md. One model on the GPU at a time. 8 GB VRAM.
- Nothing existing may break. The current single-view pipeline stays as "Quick mode" and remains the source of the 2D evaluation and the NYU depth evaluation.
- New work goes in new modules (`src/floorplan.py`, `src/generate_pano.py`, `src/pano_depth.py`, `src/room_mesh.py`, `src/layout.py`).
- Work through the milestones in order. After each one: run it, save figures to `results/figures/` and tables to `results/tables/`, update the README, commit, push, and append a dated entry to `PROGRESS.md` (what was built, results, what failed, what is next). `PROGRESS.md` is how the work resumes after a usage-limit stop: read it first on every restart.
- If a milestone does not work after reasonable attempts, write down why as a negative result and move to the next one. Do not hide failures and do not tune on test data.
- Downloads: pre-approved up to 12 GB in total for this plan (LoRA, ControlNet, depth weights, upscaler, small validation sets). List every download with its size in `PROGRESS.md`. Ask before going over.
- Long jobs must be resumable scripts. The owner is asleep, so run them yourself in the foreground, one at a time.
- Every new stage needs at least one number that measures it, not only pictures.

## Milestone 1: floor plan from depth (works on the current pipeline)

Goal: replace the depth heat map in the app with a measured, architect-style plan.

1. `src/floorplan.py`
   - Input: metric point cloud + segmentation masks.
   - Align gravity using the fitted floor plane, then project top-down (orthographic) to a 2D grid in metres.
   - Walls: project wall-class points, clean the occupancy map with morphological closing/opening, extract wall lines (Hough or RANSAC line fitting), snap to the dominant directions when the data supports a Manhattan layout, merge collinear segments.
   - Furniture: split furniture masks into instances (connected components), project each one, take its footprint as a minimum-area rectangle. Label from the ADE20K class (bed, sofa, table, chair, cabinet, lamp, plant, ...).
   - Doors and windows: project those classes onto the nearest wall line.
2. Drawing: walls as thick black lines, door/window symbols, furniture outlines with labels, dimension lines in metres, a scale bar, a camera marker, and the floor area in m² when the room polygon is closed. Export PNG and SVG.
3. Mark dimensions as approximate: the metric depth model's absolute scale was about 22% off on NYU.
4. Validation with ground truth: on the NYU test images, build the plan once from ground-truth depth and once from predicted depth. Report wall-length error, furniture-footprint IoU and centroid error.
5. App: main outputs become the 2D image, the floor plan, and the 3D viewer. Move the depth map, edges and segmentation into a collapsed "Analysis" section, with the depth map in grayscale.

## Milestone 2: 360° panorama generation

1. `src/generate_pano.py`: SD v1.5 + a panorama LoRA (`ProGamerGov/360-Diffusion-LoRA-sd-v1-5`, trigger word `qxj`) + our MMIS LoRA, at 1024×512.
   - Seamless wrap: circular padding along x in the UNet and VAE convolutions. Fallback: roll the image by half its width and inpaint the seam.
   - Test our LoRA at weights 0, 0.5, 0.8, 1.0 next to the panorama LoRA.
   - Also try `archerfmy0831/sd-t2i-360panoimage` as an alternative generator if it fits in 8 GB.
2. Metrics on at least 30 prompts (5 room types × 6 styles, fixed seeds):
   - seam discontinuity (gradient across the wrap compared with the image interior)
   - CLIP score
   - FID/KID of 90° perspective crops against the MMIS hold-out, to test whether our LoRA still helps in panorama mode
3. Super-resolution to at least 2048×1024: Real-ESRGAN (tiled, wrap-aware padding) against bicubic + unsharp masking. Report sharpness (Laplacian variance) and time.
4. Choose the configuration from the numbers and save comparison grids.

## Milestone 3: panoramic depth

1. Method A, our own (image-processing contribution): equirectangular → overlapping perspective views (cube faces plus offset views, 90–100° FOV) → Depth Anything V2 Metric Indoor per view → convert z-depth to radial distance → global least-squares alignment of per-view scale (and shift) over the overlaps → feathered blending back to an equirectangular distance map. Report the disagreement in the overlaps before and after alignment.
2. Method B: DAP, "Depth Any Panoramas" (`Insta360-Research/DAP-weights`, code at `github.com/Insta360-Research-Team/DAP`), a metric panoramic depth model. Run it if it fits in 8 GB with fp16.
3. Compare A and B on the generated panoramas: wall planarity (RMS distance to RANSAC planes on wall-segmented points), floor flatness, angle between adjacent walls, ceiling height plausibility, time, VRAM. If a small real RGB-D panorama set with ground truth fits in the download budget (for example a Stanford2D3D or Matterport3D sample), also report AbsRel, RMSE and δ<1.25. If none is reachable, say so.
4. Apply the gated denoising and the depth refinement with wrap-aware filtering.

## Milestone 4: full-room 3D

1. `src/room_mesh.py`: equirectangular distance map → spherical back-projection → textured mesh. Reuse the textured-mesh logic: cuts at depth discontinuities, an inpainted background layer, plane completion. Stitch the wrap seam and handle the poles.
2. Room box: RANSAC planes for floor, ceiling and walls, Manhattan snap where supported, close the room.
3. Outputs: a full-room `.glb`, and a "dollhouse" variant with the ceiling removed for viewing from outside and above.
4. Viewer in the app: two modes.
   - Inside: camera at the panorama centre, look around freely, small movement radius.
   - Dollhouse: orbit from above with the ceiling hidden.
   - Keep `gr.Model3D` as an offline fallback.
5. Metrics: empty share from inside at 0, 0.5 and 1.0 m from the centre and from dollhouse views, compared with the single-view mesh. Timing and VRAM.

## Milestone 5: floor plan of the full room

1. Run `floorplan.py` on the 360° point cloud: a closed room polygon with wall lengths, area, doors, windows, and labelled furniture footprints. Architect-style drawing, PNG and SVG.
2. Consistency check: plan area against the mesh floor area.
3. Put the plan next to the dollhouse view in one figure per room.

## Milestone 6 (research experiment): layout first, like an architect

Idea: decide the plan first, then make the image obey it, so the 3D geometry is exact.

1. `src/layout.py`: a rule-based parametric layout generator per room type. Room dimensions sampled from realistic ranges; furniture as boxes with placement rules (bed against a wall with nightstands beside it, sofa facing the TV unit, minimum clearances, nothing blocking the door). Seeded. Output: JSON and a dimensioned floor plan.
2. Render the box layout from the room centre as an equirectangular depth map.
3. Generate the panorama conditioned on that depth: ControlNet depth for SD v1.5 (`lllyasviel/control_v11f1p_sd15_depth`) + the panorama LoRA + our LoRA.
4. Build the 3D by projecting the panorama texture onto the known layout geometry (exact walls, furniture boxes). Optionally refine furniture surfaces with estimated depth, constrained to stay near the boxes.
5. Measure layout adherence: depth estimated from the generated panorama against the layout depth (AbsRel), and segmentation of the generated panorama against the layout semantics (IoU). Compare the 3D result with milestone 4 on the same room descriptions.
6. If adherence is poor, document it and keep milestone 4 as the default.

## Final

- App with the modes that work: Quick (single view), 360° Room, Layout-first.
- README: new pipeline diagram, all new tables and figures, updated objectives and limitations.
- `PROGRESS.md` complete. Everything committed and pushed.
- Do not start the final course notebook; that comes after the owner reviews the results.
