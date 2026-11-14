# RoomGen3D v4 plan: clear 2D design shots, sharp 3D, real furniture shapes

## What is wrong now (from the owner's screenshots of the 360° mode, v3)

1. The "2D image" shown to the user is the raw equirectangular panorama. It is small and distorted, so the design and the furniture placement cannot be read from it.
2. The inside 3D view is blurry and grainy: floor and wall textures are noisy, furniture edges are stair-stepped, colours look washed out, and the viewer can open looking at the ceiling. Most of the design detail is lost.
3. Furniture is still textured blocks, not furniture shapes (carried over from v3).

## Ground rules

Same as PLAN_V2.md and PLAN_V3.md: follow CLAUDE.md, 8 GB VRAM, one model on the GPU at a time, nothing existing breaks, commit + push + `PROGRESS.md` entry after each milestone, negative results written down, long jobs resumable and run in the foreground, the owner is asleep so do not wait for answers.

- Disk: keep at least 10 GB free on C: at all times. Download budget 10 GB, each download listed. Delete intermediate files once tables and figures are saved.
- Keep the current pipeline callable as `v3` for before/after comparisons on the same seeds.
- Test bench: the six demo rooms plus the 40-room layout-first bench from v3 (same development / held-out split; tune only on the development half).
- Every milestone needs numbers, not only pictures. For image quality use both full-reference metrics where a reference exists (PSNR, SSIM, LPIPS) and no-reference metrics where it does not (Laplacian variance, BRISQUE or NIQE).

## Milestone 1: readable 2D design shots

Goal: the main 2D output is a set of normal-looking interior photos, not the panorama.

1. From the panorama, render perspective views (rectilinear, about 75–90° horizontal FOV, level horizon, camera at eye height).
   - Four cardinal views aligned with the walls of the fitted room box.
   - One or two "hero" views chosen automatically: score candidate directions by the amount and variety of furniture in view (instance masks), wall coverage, and how little of the frame is ceiling or floor; keep the best.
2. Render them from the highest-resolution panorama available (milestone 2 makes it sharper) at about 1024×768.
3. App: a gallery with the hero view first, then the four wall views. The panorama moves to an interactive 360° viewer tab (and the "Analysis" section). The floor plan stays beside the gallery.
4. On the floor plan, draw each shot's camera position and view cone with a letter (A, B, C, D, Hero), and caption the gallery images with the same letters.
5. Metric: hero-view selection against a simple baseline (always the first wall): furniture instances in view, and share of the frame that is furniture or wall.

## Milestone 2: a sharper panorama

Goal: more real detail in the texture that everything else is built from.

1. Measure first: how much detail a 75° view currently gets (source pixels per degree), and the sharpness of the current views.
2. Refinement by views: project the panorama to overlapping perspective views (enough to cover the sphere), upscale each (Real-ESRGAN, tiled), then run low-strength img2img on each view (strength about 0.2–0.35, same prompt) to add real texture detail. Try this with and without our MMIS LoRA: this is where our own LoRA can re-enter the 360° mode, so report honestly whether it helps (KID/FID of the views against the MMIS hold-out, CLIP score, sharpness).
3. Blend the refined views back into a 4096×2048 equirectangular panorama with feathered, multi-band blending; keep the wrap seam continuous; check that neighbouring views agree (overlap difference before and after blending).
4. Guard against drift: the refined panorama must stay geometrically consistent with the original (edges and depth should not move). Report depth disagreement and edge alignment between original and refined; lower the strength if it drifts.
5. Compare: bicubic, Real-ESRGAN only, Real-ESRGAN + img2img, with and without the LoRA. Sharpness, no-reference quality, time, VRAM. Choose from the development rooms.

## Milestone 3: a sharp, clean 3D view

1. Texture: use the refined 4096×2048 panorama. Check the whole texture path for losses (JPEG quality, texture size limits in the .glb, the viewer's filtering). Use mipmaps and anisotropic filtering. Target file size under about 25 MB, and report the size/quality trade-off.
2. Grain: find where the noise on floors and walls comes from (generator, upscaler, or sampling). Apply edge-preserving denoising to the plane textures in their own texture space if it helps, measured, without blurring real detail.
3. Geometry: remove the stair-stepping on furniture. Options to compare: edge-preserving smoothing of the depth inside each instance before meshing, a finer vertex grid near depth edges, Laplacian or Taubin smoothing of the furniture meshes with borders fixed.
4. Colour: make the viewer show the image colours unchanged (tone mapping and exposure), and compare a render against the source view to prove it.
5. Camera: the inside viewer opens level, at eye height, facing the hero direction from milestone 1.
6. Metric: render the room from inside at fixed poses (the milestone-1 views) and compare with the 2D design shots: PSNR, SSIM, LPIPS, sharpness ratio. Report v3 against v4. Take headed browser screenshots as in v3.

## Milestone 4: real furniture shapes (object-level image-to-3D)

Goal: replace the textured blocks with actual 3D furniture.

1. Choose an image-to-3D model that runs in 8 GB: test TripoSR first; also try Stable Fast 3D (if its licence can be accepted with the logged-in Hugging Face account) and Hunyuan3D-2mini if they fit. Report VRAM, seconds per object, and quality for each, and pick one.
2. Per furniture instance:
   - Pick the best perspective view of it (largest, least occluded, closest to frontal) from the refined panorama.
   - Crop, upscale, and cut it out with the instance mask (clean the mask with morphology and feather the edge).
   - If another object occludes it, inpaint the missing part before generation.
   - Generate the 3D object.
3. Placement:
   - Scale to the instance's measured footprint and height (with the class size priors from v3).
   - Orientation: render-and-compare. Render the object's silhouette from the panorama camera for candidate yaw angles and keep the one with the highest IoU against the instance mask, with a prior toward wall-aligned angles.
   - Rest it on the floor (or hang it from the ceiling for ceiling lamps), and resolve collisions with walls and other objects.
4. Fallbacks: if the silhouette IoU is below a threshold chosen on the development set, keep the v3 solid block for that instance. Keep the block for classes the model handles badly, and say which.
5. Recall: v3 rejects many uncertain instances (recall 0.38). Add a second tier: uncertain instances stay as relief on the wall or floor texture rather than disappearing.
6. Metrics, v3 against v4, on the held-out bench and the six demo rooms:
   - silhouette IoU from the panorama viewpoint
   - re-render the full room from the panorama centre and compare with the panorama (PSNR, SSIM, LPIPS)
   - footprint IoU, size and orientation error against the layout-first ground truth
   - share of instances that used the generated object, the block, or the relief
   - time and VRAM per room
7. Speed: objects are slow, so add a slider in the app for the maximum number of generated objects (0 = v3 blocks), largest objects first, and cache objects per seed.
8. Floor plan: unchanged in content, but check that symbols and the placed objects agree in position and orientation.

## Milestone 5: remaining v3 weaknesses (only after milestones 1–4)

1. Hidden floor and wall areas: replace the smooth fill and the dark patches with texture synthesis from the visible part of the same plane (patch-based or tiling of the detected pattern), compared with v3 by no-reference quality and by eye.
2. Open-plan rooms boxed too large: detect openings (regions where depth goes far beyond the fitted wall) and report "open side" on the plan instead of pushing the wall out. Default the candidate count to 3.
3. Doors and windows (recall 0.02): try detecting them on the refined perspective views from milestone 2 instead of on the panorama, and report whether recall improves.

## Final

- App: 360° mode runs v4 by default with a v3 toggle. Layout: design-shot gallery and floor plan on top, then the viewers (Inside, Dollhouse, Top-down, 360° panorama), then Analysis.
- One summary figure per demo room: hero shot, wall shots, floor plan with view cones, v3 dollhouse, v4 dollhouse, v3 inside render, v4 inside render.
- One summary table with every before/after metric.
- README and `PROGRESS.md` updated; everything pushed.
- Do not start the final course notebook.
