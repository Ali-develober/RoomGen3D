"""Design-shot cameras placed in the v6 scene (PLAN_V6 milestone 3).

    hero (H)   candidates: the four corners (0.45 m from both walls) and the doorway (0.6 m inside the door),
               1.5 m high, level (pitch 0) with a vertical lens shift of 0 / 15 / 25 % of the image height
               (architectural shift lens: verticals stay vertical, more of the room below eye height),
               horizontal FOV 70 / 75 / 80°, yaw aimed at the main piece and
               swept ±25° in 5° steps; the best VALID candidate by the hero score; H2 = the best valid candidate
               from a different position (second hero)
    walls (A-D) one view per wall from the opposite side: camera 0.5 m in front of the opposite wall (moved
               along it to the free spot closest to the middle), 1.5 m, level, looking straight at the wall,
               FOV 75 / 80°
Validity (exact, from the rendered instance ids and planar depth of the proxy scene at 192 x 144):
    main piece share  the main piece and its group (bed + nightstands, table + chairs, sofa + TV unit /
                      coffee table / armchairs, ...) cover 15-50 % of the frame (hero only)
    near              content nearer than 0.8 m covers < 15 % of the frame
    floor             the floor covers < 35 % of the frame
    inside            the camera is >= 0.3 m from every furniture footprint and inside the room
Hero score = main share closeness to 30 % + visible pieces + depth of view (median depth), the near / floor
rules as hard constraints.
"""

from __future__ import annotations

import numpy as np

from . import layout_v6 as LV
from . import scene_v6 as S

EYE = 1.5
SHIFTS = (0.0, 0.15, 0.25)     # vertical lens shift (fraction of the image height), the camera stays level
VW, VH = 192, 144
ASPECT = 4 / 3


def _clear(lay: dict, x: float, z: float, margin: float = 0.3) -> bool:
    if not (0.25 <= x <= lay["W"] - 0.25 and 0.25 <= z <= lay["D"] - 0.25):
        return False
    for p in lay["items"]:
        if p.get("y0", 0) > 1.7:
            continue
        x0, z0, x1, z1 = LV.extent(p)
        if x0 - margin < x < x1 + margin and z0 - margin < z < z1 + margin:
            return False
    return True


def main_group(lay: dict) -> tuple[dict | None, set]:
    main = next((p for p in lay["items"] if p["cls"] == LV.MAIN[lay["room"]]), None)
    if main is None:
        return None, set()
    ids = {main["id"]} | {p["id"] for p in lay["items"] if p.get("group") == main["id"]}
    ids |= {p["id"] for p in lay["items"] if p.get("group") in ids}
    if lay["room"] == "kitchen":                                 # the counter run is the kitchen's main piece
        ids |= {p["id"] for p in lay["items"] if p["cls"] in ("counter", "stove", "sink", "fridge", "island")}
    return main, ids


def evaluate(rd: S.Renderer, lay: dict, pos, yaw: float, hfov: float, group: set, shift: float = 0.0) -> dict:
    vfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(hfov) / 2) / ASPECT))
    r = rd.perspective(pos, yaw, 0.0, hfov, VW, VH, shift)
    inst, z = r["inst"], r["z"]
    n = inst.size
    furn = inst >= S.FURN0
    main_share = float(np.isin(inst - S.FURN0, list(group)).sum() / n) if group else 0.0
    near = float((np.nan_to_num(z, nan=99, posinf=99) < 0.8).mean())
    floor = float((inst == S.FLOOR).mean())
    pieces = len({int(i) - S.FURN0 for i in np.unique(inst[furn])
                  if (inst == i).sum() > 0.003 * n})
    med = float(np.median(np.nan_to_num(z, nan=10, posinf=10)))
    return {"pos": [float(pos[0]), float(pos[1]), float(pos[2])], "yaw": float(yaw), "hfov": float(hfov),
            "vfov": float(vfov), "shift": float(shift), "main_share": main_share, "near_share": near, "floor_share": floor,
            "pieces": pieces, "median_depth": med, "furn_share": float(furn.mean())}


def valid(m: dict, hero: bool) -> bool:
    ok = m["near_share"] < 0.15 and m["floor_share"] < 0.35
    if hero:
        ok = ok and 0.15 <= m["main_share"] <= 0.50
    return ok


def hero_score(m: dict) -> float:
    return -abs(m["main_share"] - 0.30) * 4 + 0.15 * min(m["pieces"], 8) + 0.15 * min(m["median_depth"], 4) \
        - 2 * max(0, m["floor_share"] - 0.3)


def _aim(pos, target) -> float:
    return float(np.rad2deg(np.arctan2(target[0] - pos[0], target[1] - pos[2])))


def hero_candidates(lay: dict) -> list:
    W, D = lay["W"], lay["D"]
    pts = [(0.45, 0.45), (W - 0.45, 0.45), (0.45, D - 0.45), (W - 0.45, D - 0.45)]
    door = next(o for o in lay["openings"] if o["kind"] == "door")
    cx, cz, _ = LV.Layout.on_wall(type("L", (), {"W": W, "D": D})(), door["wall"], door["c"], 0.9, 1.2)
    pts.append((cx, cz))
    out = []
    for (x, z) in pts:
        for dx in (0, 0.25, -0.25):                          # nudge out of furniture if needed
            for dz in (0, 0.25, -0.25):
                if _clear(lay, x + dx, z + dz):
                    out.append((x + dx, z + dz))
                    break
            else:
                continue
            break
    return out


def choose_cameras(rd: S.Renderer, lay: dict) -> dict:
    """{"H": metrics, "H2": metrics | None, "A".."D": metrics} with "valid" flags."""
    main, group = main_group(lay)
    target = (main["cx"], main["cz"]) if main else (lay["W"] / 2, lay["D"] / 2)
    if main and lay["room"] != "kitchen":                   # aim between the main piece and the room centre
        target = (0.75 * target[0] + 0.25 * lay["W"] / 2, 0.75 * target[1] + 0.25 * lay["D"] / 2)
    cands = []
    for (x, z) in hero_candidates(lay):
        pos = (x, EYE, z)
        a0 = _aim(pos, target)
        for hf in (70, 75, 80):
            for da in range(-25, 30, 5):
                for sh in SHIFTS:
                    m = evaluate(rd, lay, pos, a0 + da, hf, group, sh)
                    m["valid"] = valid(m, True)
                    m["score"] = hero_score(m)
                    cands.append(m)
    cands.sort(key=lambda m: (m["valid"], m["score"]), reverse=True)
    shots = {"H": dict(cands[0], name="H")}
    p0 = np.array(cands[0]["pos"])
    others = [m for m in cands if np.linalg.norm(np.array(m["pos"]) - p0) > 1.0]
    shots["H2"] = dict(others[0], name="H2") if others else None
    names = {"N": "A", "E": "B", "S": "C", "W": "D"}
    used = [np.array(m["pos"]) for m in (shots["H"], shots["H2"]) if m]
    for wall, nm in names.items():
        best = None
        L = lay["W"] if wall in ("N", "S") else lay["D"]
        opp = LV.OPP[wall]
        yaw = {"N": 0, "S": 180, "E": 90, "W": 270}[wall]
        for off in (0.5, 0.8, 1.1, 1.4):
            for t in np.linspace(0.5, 0.1, 5).tolist() + np.linspace(0.5, 0.9, 5).tolist():
                s = t * L
                x, z = {"S": (s, off), "N": (s, lay["D"] - off), "W": (off, s), "E": (lay["W"] - off, s)}[opp]
                if not _clear(lay, x, z) or any(np.hypot(x - u[0], z - u[2]) < 0.6 for u in used):
                    continue                                   # not on top of another camera (plan markers)
                for hf, sh in ((75, 0.15), (80, 0.15), (70, 0.15), (65, 0.1)):
                    m = evaluate(rd, lay, (x, EYE, z), yaw, hf, group, sh)
                    m["valid"] = valid(m, False)
                    # valid first, then nothing near the lens (smeared edges), then centred, close to the wall
                    key = (m["valid"], m["near_share"] < 0.01, -round(m["near_share"], 2), -abs(t - 0.5), -off, hf)
                    if best is None or key > best[0]:
                        best = (key, m)
        if best:
            shots[nm] = dict(best[1], name=nm, wall=wall)
            used.append(np.array(best[1]["pos"]))
    return shots
