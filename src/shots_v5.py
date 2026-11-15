"""PLAN_V5 milestone 2: design shots that show the room from a distance, not close-ups.

v4 picked the hero by a furniture score alone, so a shot could be a close-up of a table top. v5 adds a validity test
(src/camera_v5.view_validity) and only chooses among VALID views:

    level horizon (pitch = roll = 0 in the room frame B), horizontal FOV 85° or 90°, camera at the panorama centre
    (moving the camera to 1.4-1.6 m would need re-rendering from the depth mesh, which tears: not done)
    valid  <=>  main piece covers 15-50 % of the frame  and  median distance in the frame > 1.5 m
                and  content nearer than 0.8 m covers < 15 % of the frame
    main piece = src/design_shots.main_piece (room type's main class, largest least occluded); without one, the
                largest furniture instance in the frame stands in for it
    hero = the valid view with the best v4 hero score; if no view is valid, the least bad one (smallest summed
           violation) and the caption says so. Wall views A-D stay (90°), each tagged valid / not valid.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import camera_v5 as cam
from . import design_shots as ds

FOVS = (85.0, 90.0)
STEP = 5.0
SW, SH = 256, 192


def measure(class_map, inst_map, families, dist, box, yaw, fov, sets, main_id):
    """v4 view measures + validity measures for one level view."""
    M = ds.view_to_cam(box, yaw)
    m = ds.view_measures(class_map, inst_map, families, M, fov, sets, dist=dist, main_id=main_id)
    imv = ds.render_view(inst_map.astype(np.float32), M, SW, SH, fov, cv2.INTER_NEAREST).astype(np.int32)
    dv = ds.render_view(dist.astype(np.float32), M, SW, SH, fov, cv2.INTER_NEAREST)
    if main_id is not None:
        share = float((imv == main_id).mean())
    else:
        furn = [i for i in np.unique(imv[imv >= 0])]
        share = float(max(((imv == i).mean() for i in furn), default=0.0))
    v = cam.view_validity(share, dv)
    return {**m, **{f"v_{k}": val for k, val in v.items()}}


def select_shots(texture, box, class_map, inst_map, instances, id2label, dist, room=None, weights=None,
                 size=(ds.SHOT_W, ds.SHOT_H), fovs=FOVS, step=STEP) -> dict:
    """Hero (best valid view) + wall views A-D. Returns the same structure as design_shots.design_shots, with the
    validity measures in every shot's 'measures' and 'valid' / 'caption_note' fields."""
    t0 = time.perf_counter()
    fam = {int(i["id"]): i.get("family", i.get("label", "")) for i in instances}
    sets = ds.class_sets(id2label)
    main = ds.main_piece(inst_map, instances, dist, box, room)
    mid = None if main is None else main["id"]
    w = ds.selected_weights() if weights is None else weights
    scan = []
    for fov in fovs:
        for y in np.arange(0.0, 360.0, step):
            m = measure(class_map, inst_map, fam, dist, box, float(y), fov, sets, mid)
            m.update({"yaw": float(y), "fov": float(fov), "score": float(ds.score_view(m, w))})
            scan.append(m)
    valid = [m for m in scan if m["v_valid"]]
    if valid:
        hero = max(valid, key=lambda m: m["score"])
        note = ""
    else:
        hero = min(scan, key=lambda m: (m["v_badness"], -m["score"]))
        note = "no view passes the distance test; least bad view (" + ", ".join(hero["v_fail"]) + ")"
    shots = [{"letter": "H", "kind": "hero", "yaw": hero["yaw"], "fov": hero["fov"], "measures": hero,
              "valid": bool(hero["v_valid"]), "note": note,
              "image": ds.render_view(texture, ds.view_to_cam(box, hero["yaw"]), *size, hero["fov"])}]
    for letter, yaw in ds.wall_yaws(box).items():
        m = measure(class_map, inst_map, fam, dist, box, yaw, ds.WALL_FOV, sets, mid)
        shots.append({"letter": letter, "kind": "wall", "yaw": yaw, "fov": ds.WALL_FOV, "measures": m,
                      "valid": bool(m["v_valid"]), "note": "",
                      "image": ds.render_view(texture, ds.view_to_cam(box, yaw), *size, ds.WALL_FOV)})
    return {"shots": shots, "scan": scan, "seconds": time.perf_counter() - t0, "main": main,
            "n_valid_yaws": len(valid), "n_yaws": len(scan)}


def validity_of(shot: dict, class_map, inst_map, instances, id2label, dist, box, room=None) -> dict:
    """Validity measures of an existing (e.g. v4) shot {'yaw', 'fov'}."""
    fam = {int(i["id"]): i.get("family", i.get("label", "")) for i in instances}
    main = ds.main_piece(inst_map, instances, dist, box, room)
    return measure(class_map, inst_map, fam, dist, box, shot["yaw"], shot["fov"], ds.class_sets(id2label),
                   None if main is None else main["id"])


def caption(shot: dict) -> str:
    m = shot["measures"]
    if shot["kind"] == "hero":
        s = f"H: hero view ({m['inst']} furniture pieces, main piece {100 * m['v_main_share']:.0f} % of the frame, " \
            f"median distance {m['v_median_depth']:.1f} m)"
        return s + (f" - {shot['note']}" if shot.get("note") else "")
    fail = [f for f in m.get("v_fail", []) if f != "main piece size"]
    return f"{shot['letter']}: wall view" + (f" ({', '.join(fail)})" if fail else "")
