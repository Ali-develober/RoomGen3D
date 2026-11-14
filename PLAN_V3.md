# RoomGen3D v3 plan: clean rooms, solid furniture, a readable floor plan

## What is wrong now (from the owner's screenshots of the 360° mode)

Inside view
- White gaps at depth discontinuities (next to the door, around the TV unit, at the ceiling): the viewer background shows through.
- Smeared texture where surfaces are seen at grazing angles.

Dollhouse view
- The room is not a room: walls are broken into fragments, and the shape is a cross because the panorama shows openings into other spaces.
- Large flat single-colour slabs where planes were completed.
- Furniture is thin shells with no back or sides, so from above it looks like scraps.
- The near walls block the view or are cut arbitrarily.

Floor plan
- All four walls are "inferred, no wall line detected".
- Furniture rectangles are stretched along the viewing ray (depth-edge smear), overlap each other, and several lie outside the walls.
- Labels are wrong or implausible (a "lamp" of 1.2 m, three "coffee tables").
- The plan is drawn rotated, and furniture is not aligned with the walls.
- Furniture is plain rectangles, not architectural symbols.

## Ground rules

Same as PLAN_V2.md: follow CLAUDE.md, 8 GB VRAM, one model on the GPU at a time, nothing existing breaks, commit + push + `PROGRESS.md` entry after each milestone, negative results are written down, long jobs resumable and run in the foreground. Download budget for this plan: 8 GB, each download listed.

Measure every change. The layout-first mode gives ground-truth layouts (room size, furniture class, position, size), so use it as the test bench: generate a fixed set of layout-conditioned panoramas, split the seeds into a development set and a held-out set, tune only on the development set, and report on the held-out set.

Keep the current 360° pipeline callable as `v2` so every milestone has a before/after comparison on the same panoramas.

## Milestone 1: instance segmentation that works on panoramas

1. The current semantic segmentation runs on the distorted equirectangular image and splits instances by connected components. Replace it: project the panorama to overlapping perspective views, run a panoptic/instance model (for example Mask2Former or OneFormer, ADE20K panoptic, whichever fits in 8 GB), and merge the instances back onto the panorama (match instances across views by overlap).
2. Compare old vs new on the layout-first test bench: furniture detection precision and recall by class, and mask IoU against the rendered layout semantics.

## Milestone 2: room box regularization

Goal: the room becomes a clean box with straight, fully textured walls.

1. Dominant directions: estimate the two horizontal wall directions from wall-point normals (histogram), and rotate everything so the walls are axis-aligned.
2. Box fit: for each of the four sides, take the wall position from the peak of wall-point density along that axis; floor and ceiling from their planes. Report how many sides are detected rather than inferred. If a pretrained panorama layout estimator (HorizonNet, LGT-Net or similar) can be downloaded and run, compare it with this fit.
3. Rebuild the shell: walls, floor and ceiling are exact planes, textured by projecting the panorama onto them. Anything seen beyond a wall (through a doorway, window or opening) is flattened onto the wall plane as texture, not kept as geometry. No more arms sticking out of the room.
4. Hidden regions of the shell (behind furniture): fill by inpainting in the plane's own texture space (per wall, per floor), not with a flat colour. Compare Telea, Navier-Stokes and a patch-based fill.
5. Metrics, v2 vs v3: wall planarity, share of mesh area outside the room box, share of flat-colour area, right-angle error.

## Milestone 3: solid furniture

1. Per furniture instance (from milestone 1): keep the points that are clearly inside the box (not on the shell), erode the mask by a few pixels and drop flying pixels before back-projection, and trim outliers with percentiles so footprints are not stretched along the viewing ray.
2. Make each instance a solid: extrude the visible surface down to the floor (or up to the ceiling for hanging lamps), close the hidden sides using the footprint rectangle, and colour hidden faces by inpainting from the visible faces of the same object.
3. Class size priors: a table of plausible width/depth/height ranges per class (bed, sofa, armchair, chair, table, coffee table, cabinet, wardrobe, lamp, plant, TV...). Clamp or reject instances far outside the range, and report how many.
4. Snap furniture orientation to the wall axes when within about 20°, as real furniture usually is.
5. Resolve overlaps: merge duplicates of the same class with high overlap, shrink minor overlaps, drop instances outside the room.
6. Metrics on the held-out layout-first set: footprint IoU, centre error (m), size error (%), orientation error, v2 vs v3.

## Milestone 4: floor plan drawing

1. Draw the plan axis-aligned, with the longest wall horizontal.
2. Walls: solid where detected, hatched only where inferred. Door swings and window marks on the wall lines.
3. Furniture as architectural symbols per class (bed with pillows, sofa with back and arms, round or rectangular table with chairs, wardrobe with door swing line, lamp as a small circle, plant, TV unit, bathtub, toilet, sink, kitchen counter with hob). Fall back to a labelled rectangle for other classes.
4. No overlapping labels; labels inside the symbol when it is large enough, otherwise beside it with a leader line. Hide instances smaller than a set size.
5. Dimension lines for the room and for the main furniture clearances (for example bed to wall), a scale bar, area, and the "approximate" note.
6. Export PNG, SVG and DXF.
7. Figure: v2 plan vs v3 plan for the same six rooms.

## Milestone 5: panoramas that are actually one room

1. Prompt template for a single enclosed room (four walls, view from the centre) with a negative prompt against corridors, hallways, open-plan spaces and multiple rooms. Test whether it reduces openings.
2. Automatic selection: generate 4 candidates per request, score each by "boxiness" (share of wall points within 10 cm of the fitted box, share of depth beyond the box, number of detected sides), and keep the best. Report the score distribution with and without the template, and the time cost.
3. Make the candidate count a slider in the app (1 = fast).

## Milestone 6: viewers

1. Dollhouse: single-sided materials with inward-facing normals, so walls facing away from the viewer disappear automatically (like commercial dollhouse views). Ceiling hidden. Soft ambient lighting, neutral background.
2. Inside: set the scene background to the room's mean colour so thin gaps are not white; widen the background band at discontinuities so gaps close.
3. Add a third view: top-down orthographic, next to the floor plan, to show that plan and model agree.
4. Open each viewer in a real browser (headed, with screenshots at several angles) and include the screenshots in the report. If headed testing is impossible, say so clearly.

## Milestone 7: layout-first image quality

1. Replace box furniture in the conditioning depth with simple shaped proxies (bed with headboard and mattress, sofa with back and arms, table with legs, chairs with backs).
2. Sweep ControlNet strength (0.4–1.0) and the step range in which it is applied (for example only the first 50–70%), and report layout adherence against image quality (CLIP score, KID on perspective crops). Pick the best trade-off from the development seeds.
3. Add doors and windows to the layout and the conditioning.
4. Try the better panorama generator from PLAN_V2 milestone 2 with the same ControlNet if they are compatible.
5. Optional second pass: img2img at low strength without ControlNet to restore texture detail, checking that adherence does not drop.

## Final

- App: the 360° mode uses the v3 pipeline by default, with v2 available as a comparison toggle.
- One summary figure per room: panorama, v2 dollhouse, v3 dollhouse, v2 plan, v3 plan.
- One summary table of all before/after metrics.
- README and `PROGRESS.md` updated; everything pushed.
- Do not start the final course notebook.
