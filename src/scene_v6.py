"""Proxy 3D scene of a v6 layout (PLAN_V6 milestone 1): room shell with real door / window openings and
frames + the placed CC0 furniture models, rendered by CPU ray casting (src/raycast_v6.py).

Shell
    floor, ceiling, four walls as quads with rectangular holes at the openings; each opening has a reveal
    (jambs, head, sill) 0.15 m deep; windows: a pane 0.10 m behind the wall surface with a 5 cm frame and
    one mullion / transom (so a depth condition shows a framed window, not a hole to infinity); door: a
    closed door leaf 4 cm behind the wall surface, a 7 cm casing protruding 2 cm into the room and a
    handle; 8 cm baseboards along the walls (not across the door).
Furniture
    each layout piece gets a model of its class (src/assets_v6.CATALOG, one model per class per room,
    chosen by style and seed), scaled into the piece's footprint (proportions kept within 20 %), turned
    to its yaw and placed; the layout piece's w / d / h are then updated to the model's real extents so
    that the floor plan, the cameras and the renders all describe the same object.
Windows look out to a backdrop 3 m behind the wall (a flat pane was read as a painted panel by the depth
    ControlNet) and have two pleated curtain panels on a straight rod; a pendant lamp hangs over the dining
    table (else the room centre).
Instance ids (render "inst"):  0 wall (incl. baseboards, reveals), 1 floor, 2 ceiling, 3 window, 4 door,
    5 curtain, 6 lamp, 7 wall art, 10 + i = layout item i.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from . import assets_v6 as A
from . import layout_v6 as LV
from . import raycast_v6 as R

WALL, FLOOR, CEIL, WINDOW, DOOR, CURTAIN_ID, LAMP, ART = 0, 1, 2, 3, 4, 5, 6, 7
FURN0 = 10
REVEAL = 0.15
PANE_IN = 0.10             # frame plane of the window (behind the wall surface)
OUTSIDE = 3.0              # m: backdrop behind a window opening
STRUCT_COLORS = {WALL: (232, 228, 220), FLOOR: (176, 140, 104), CEIL: (245, 245, 242), WINDOW: (250, 250, 250),
                 DOOR: (200, 180, 150)}


def _tm():
    import trimesh
    return trimesh


def _mesh(v, f, color):
    m = _tm().Trimesh(np.asarray(v, float), np.asarray(f, np.int64), process=False)
    m.visual.face_colors = np.tile(np.array([*color, 255], np.uint8), (len(f), 1))
    return m


def _box(x0, y0, z0, x1, y1, z1, color=(200, 200, 200)):
    b = _tm().creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    b.visual.face_colors = np.tile(np.array([*color, 255], np.uint8), (len(b.faces), 1))
    return b


def _quad(p0, p1, p2, p3, color):
    return _mesh([p0, p1, p2, p3], [[0, 1, 2], [0, 2, 3]], color)


def wall_frame(lay: dict, wall: str):
    """(origin, along unit vector, inward normal, length) of a wall in world coordinates (y up)."""
    W, D = lay["W"], lay["D"]
    return {"S": (np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0]), W),
            "N": (np.array([0, 0, D]), np.array([1, 0, 0.0]), np.array([0, 0, -1.0]), W),
            "W": (np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), np.array([1, 0, 0.0]), D),
            "E": (np.array([W, 0, 0.0]), np.array([0, 0, 1.0]), np.array([-1, 0, 0.0]), D)}[wall]


def _wall_with_holes(lay, wall, holes):
    """Wall quad grid without the cells inside holes (s0, s1, y0, y1)."""
    o, a, n, L = wall_frame(lay, wall)
    H = lay["H"]
    ss = sorted({0.0, L, *[h[0] for h in holes], *[h[1] for h in holes]})
    ys = sorted({0.0, H, *[h[2] for h in holes], *[h[3] for h in holes]})
    v, f = [], []
    for i in range(len(ss) - 1):
        for j in range(len(ys) - 1):
            sc, yc = (ss[i] + ss[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(h[0] < sc < h[1] and h[2] < yc < h[3] for h in holes):
                continue
            k = len(v)
            for s, y in ((ss[i], ys[j]), (ss[i + 1], ys[j]), (ss[i + 1], ys[j + 1]), (ss[i], ys[j + 1])):
                v.append(o + a * s + np.array([0, y, 0]))
            f += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
    return _mesh(v, f, STRUCT_COLORS[WALL])


def _curtains(P, c0: float, c1: float, y1: float, H: float) -> list:
    """Two curtain panels (0.35 m each, pushed open) beside a window and a straight rod above it. Each panel is a
    folded fabric surface: depth offset 4 cm * sin(2 pi s / 0.12 m) across its width (pleats), so the depth
    condition reads as fabric (box-shaped pleats were painted as wooden boards in the first tests)."""
    tm = _tm()
    out = []
    top = min(H - 0.08, y1 + 0.2)
    col = (238, 234, 226)
    for side in (-1, 1):
        s0 = c0 - 0.3 if side < 0 else c1 - 0.05
        ss = np.linspace(s0, s0 + 0.35, 36)
        off = 0.10 + 0.04 * (0.5 + 0.5 * np.sin(2 * np.pi * (ss - s0) / 0.12))
        v, f = [], []
        for k, (sv, tv) in enumerate(zip(ss, off)):
            v += [P(sv, 0.03, tv), P(sv, top, tv)]
            if k:
                i = 2 * k
                f += [[i - 2, i - 1, i + 1], [i - 2, i + 1, i]]
        m = tm.Trimesh(np.array(v), np.array(f), process=False)
        m.visual.face_colors = np.tile(np.array([*col, 255], np.uint8), (len(f), 1))
        out.append(m)
    q0, q1 = P(c0 - 0.4, top, 0.12), P(c1 + 0.4, top + 0.03, 0.15)
    out.append(_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(60, 60, 60)))
    return out


def _art_spots(lay: dict) -> list:
    """Framed pictures (wall art): above the sofa / bed / sideboard, and centred on every blank wall stretch of at
    least 1.6 m (no opening, nothing taller than 1.2 m in front). Returns [(wall, s, width, y0, y1)]."""
    out = []
    for p in lay["items"]:
        if p["cls"] in ("sofa", "bed", "sideboard") and p["wall"]:
            s = p["cx"] if p["wall"] in ("N", "S") else p["cz"]
            w = min(1.0, 0.6 * p["w"])
            y0 = max(p["h"] + 0.25, 1.25) if p["cls"] != "bed" else 1.35
            out.append((p["wall"], s, w, y0, min(y0 + 0.65, lay["H"] - 0.4)))
    for wall in ("S", "N", "W", "E"):
        L = lay["W"] if wall in ("N", "S") else lay["D"]
        busy = [(o["c"] - o["w"] / 2 - 0.45, o["c"] + o["w"] / 2 + 0.45) for o in lay["openings"] if o["wall"] == wall]
        for p in lay["items"]:
            if p["wall"] == wall and p["h"] > 1.2 or p["wall"] == wall and p["cls"] in ("sofa", "bed", "sideboard"):
                x0, z0, x1, z1 = LV.extent(p)
                busy.append((x0, x1) if wall in ("N", "S") else (z0, z1))
        busy = sorted(busy) + [(L, L)]
        cur = 0.0
        for a, b in busy:
            if a - cur >= 1.6:
                out.append((wall, (cur + a) / 2, min(0.9, 0.45 * (a - cur)), 1.3, 1.95))
            cur = max(cur, b)
    return out


def _pendant(x: float, z: float, H: float, drop: float = 0.9, r: float = 0.22) -> list:
    """Pendant lamp: cord + conical shade hanging from the ceiling."""
    tm = _tm()
    cord = _box(x - 0.006, H - drop + 0.2, z - 0.006, x + 0.006, H, z + 0.006, color=(40, 40, 40))
    shade = tm.creation.cone(radius=r, height=0.22, sections=24)
    v = np.asarray(shade.vertices).copy()
    v[:, [1, 2]] = v[:, [2, 1]]                                # cone axis z -> y (apex up)
    shade = tm.Trimesh(v + np.array([x, H - drop, z]), np.asarray(shade.faces)[:, ::-1], process=False)
    shade.visual.face_colors = np.tile(np.array([230, 225, 210, 255], np.uint8), (len(shade.faces), 1))
    return [cord, shade]


def shell_parts(lay: dict, curtains: bool = True) -> list:
    """[(mesh, inst id)] of the room shell."""
    W, D, H = lay["W"], lay["D"], lay["H"]
    parts = [(_quad([0, 0, 0], [W, 0, 0], [W, 0, D], [0, 0, D], STRUCT_COLORS[FLOOR]), FLOOR),
             (_quad([0, H, 0], [0, H, D], [W, H, D], [W, H, 0], STRUCT_COLORS[CEIL]), CEIL)]
    for wall in ("S", "N", "W", "E"):
        o, a, n, L = wall_frame(lay, wall)
        ops = [x for x in lay["openings"] if x["wall"] == wall]
        holes = [(x["c"] - x["w"] / 2, x["c"] + x["w"] / 2, x["y0"], x["y1"]) for x in ops]
        parts.append((_wall_with_holes(lay, wall, holes), WALL))

        def P(s, y, t):                                      # wall coords -> world (t > 0 = into the room)
            return o + a * s + np.array([0, y, 0]) + n * t
        # baseboards (not across doors)
        cuts = sorted((x["c"] - x["w"] / 2, x["c"] + x["w"] / 2) for x in ops if x["kind"] == "door")
        s0 = 0.0
        for c0, c1 in cuts + [(L, L)]:
            if c0 - s0 > 0.05:
                q0, q1 = P(s0, 0, 0), P(c0, 0.08, 0.015)
                parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(245, 243, 238)), WALL))
            s0 = c1
        for x in ops:
            c0, c1, y0, y1 = x["c"] - x["w"] / 2, x["c"] + x["w"] / 2, x["y0"], x["y1"]
            col = STRUCT_COLORS[WALL]
            # reveal: jambs, head, sill (quads facing into the opening)
            parts.append((_quad(P(c0, y0, 0), P(c0, y0, -REVEAL), P(c0, y1, -REVEAL), P(c0, y1, 0), col), WALL))
            parts.append((_quad(P(c1, y0, 0), P(c1, y1, 0), P(c1, y1, -REVEAL), P(c1, y0, -REVEAL), col), WALL))
            parts.append((_quad(P(c0, y1, 0), P(c0, y1, -REVEAL), P(c1, y1, -REVEAL), P(c1, y1, 0), col), WALL))
            if x["kind"] == "window":
                parts.append((_quad(P(c0, y0, 0), P(c1, y0, 0), P(c1, y0, -REVEAL), P(c0, y0, -REVEAL), col), WALL))
                # no pane: the opening looks out to a backdrop OUTSIDE m behind the wall, so the depth
                # condition shows a real window (a flat pane read as a painted panel in the first tests)
                m_ = 0.6 * OUTSIDE
                bd = _quad(P(c0 - m_, y0 - m_, -OUTSIDE), P(c1 + m_, y0 - m_, -OUTSIDE), P(c1 + m_, y1 + m_, -OUTSIDE),
                           P(c0 - m_, y1 + m_, -OUTSIDE), (205, 225, 240))
                parts.append((bd, WINDOW))
                if curtains:
                    parts += [(m, CURTAIN_ID) for m in _curtains(P, c0, c1, y1, lay["H"])]
                fw, fd = 0.05, 0.06
                fr = (245, 245, 245)
                bars = [(c0, y0, c0 + fw, y1), (c1 - fw, y0, c1, y1), (c0, y0, c1, y0 + fw), (c0, y1 - fw, c1, y1)]
                xm = (c0 + c1) / 2
                bars.append((xm - 0.025, y0, xm + 0.025, y1))                       # mullion
                ym = y0 + (y1 - y0) * 0.72
                bars.append((c0, ym - 0.025, c1, ym + 0.025))                       # transom
                for b0, b1, b2, b3 in bars:
                    q0, q1 = P(b0, b1, -PANE_IN), P(b2, b3, -PANE_IN + fd)
                    parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=fr), WINDOW))
                q0, q1 = P(c0 - 0.03, y0 - 0.04, 0), P(c1 + 0.03, y0, 0.05)          # sill board
                parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(245, 243, 238)), WINDOW))
            else:
                leaf = _box(*np.minimum(P(c0, y0, -0.08), P(c1, y1, -0.04)),
                            *np.maximum(P(c0, y0, -0.08), P(c1, y1, -0.04)), color=STRUCT_COLORS[DOOR])
                parts.append((leaf, DOOR))
                cw = 0.07
                for b0, b1, b2, b3 in [(c0 - cw, y0, c0, y1 + cw), (c1, y0, c1 + cw, y1 + cw), (c0 - cw, y1, c1 + cw, y1 + cw)]:
                    q0, q1 = P(b0, b1, 0), P(b2, b3, 0.02)
                    parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(240, 238, 232)), DOOR))
                hs = c1 - 0.08
                q0, q1 = P(hs - 0.06, 1.0, -0.04), P(hs + 0.02, 1.03, 0.02)
                parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(90, 90, 90)), DOOR))
    return parts


def _place(mesh, p: dict):
    """Turn a normalised, fitted model (front +z) to the piece's yaw and move it to its centre."""
    a = np.deg2rad(p["yaw"])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    m = mesh.copy()
    m.vertices = np.asarray(m.vertices) @ Ry.T + np.array([p["cx"], p.get("y0", 0.0), p["cz"]])
    return m


def build(lay: dict, seed: int | None = None) -> dict:
    """Scene of a layout: {"layout" (updated copy), "parts" [(mesh, inst)], "models" {item id: model id},
    "fit" {item id: per-axis scale}, "build_s"}."""
    t0 = time.perf_counter()
    lay = copy.deepcopy(lay)
    rng = np.random.default_rng(lay["seed"] if seed is None else seed)
    chosen = {}
    parts = shell_parts(lay)
    main = next((i for i in lay["items"] if i["cls"] == LV.MAIN[lay["room"]]), None)
    if lay["room"] in ("dining room", "kitchen") and main is not None and lay["room"] != "kitchen":
        lx, lz = main["cx"], main["cz"]
    else:
        lx, lz = lay["W"] / 2, lay["D"] / 2
    for wall, s_, w_, y0_, y1_ in _art_spots(lay):                       # wall art (thin framed boards)
        o, a, n, L = wall_frame(lay, wall)
        q0 = o + a * (s_ - w_ / 2) + np.array([0, y0_, 0]) + n * 0.0
        q1 = o + a * (s_ + w_ / 2) + np.array([0, y1_, 0]) + n * 0.035
        parts.append((_box(*np.minimum(q0, q1), *np.maximum(q0, q1), color=(180, 170, 150)), ART))
    parts += [(m, LAMP) for m in _pendant(lx, lz, lay["H"], drop=0.95 if lay["room"] == "dining room" else 0.55)]
    lay["lamp"] = [round(lx, 3), round(lay["H"] - (0.95 if lay["room"] == "dining room" else 0.55), 3), round(lz, 3)]
    models, fits = {}, {}
    for p in lay["items"]:
        cls = p["cls"]
        if cls not in chosen:
            chosen[cls] = A.choose(cls, lay["style"], rng)
        src, mid, front = chosen[cls]
        m0 = A.load_asset(src, mid, front)
        m, s = A.fit_to(m0, p["w"], p["d"], p["h"])
        ext = m.extents
        p["w"], p["h"], p["d"] = round(float(ext[0]), 3), round(float(ext[1]), 3), round(float(ext[2]), 3)
        if p["wall"]:                                       # keep the back against its wall after shrinking
            f = LV.front_dir(p["yaw"])
            o, a, n, L = wall_frame(lay, p["wall"])
            back = np.array([p["cx"], p["cz"]]) - f * p["d"] / 2
            dist = float(np.dot(back - o[[0, 2]], n[[0, 2]]))
            p["cx"], p["cz"] = round(p["cx"] - f[0] * dist, 3), round(p["cz"] - f[1] * dist, 3)
        parts.append((_place(m, p), FURN0 + p["id"]))
        models[p["id"]], fits[p["id"]] = mid, [round(float(x), 3) for x in s]
    return {"layout": lay, "parts": parts, "models": models, "fit": fits, "build_s": time.perf_counter() - t0}


class Renderer:
    """Ray-cast renders of a scene (equirectangular and perspective)."""

    def __init__(self, scene: dict):
        self.scene = scene
        self.tracer = R.Tracer(scene["parts"])

    def equirect(self, center, w: int = 2048, h: int = 1024, yaw: float = 0.0) -> dict:
        d = R.equirect_rays(w, h, yaw)
        res = self.tracer.cast(np.asarray(center, float), d)
        res["dirs"] = d
        return res

    def perspective(self, center, yaw: float, pitch: float, hfov: float, w: int, h: int, shift: float = 0.0) -> dict:
        d = R.persp_rays(yaw, pitch, hfov, w, h, shift)
        res = self.tracer.cast(np.asarray(center, float), d)
        res["dirs"] = d
        z_axis = R.cam_rotation(yaw, pitch)[:, 2]
        res["z"] = res["t"] * (d @ z_axis)                  # planar depth (for depth ControlNets)
        return res


def depth_control(dist: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Depth-ControlNet condition: inverse depth normalised to 0-255 (near = white), 3 channels."""
    inv = 1.0 / np.maximum(dist, 0.1)
    lo, hi = np.percentile(inv, lo_pct), np.percentile(inv, hi_pct)
    g = (255 * np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


def inst_colors(inst: np.ndarray) -> np.ndarray:
    """Colour-coded instance map for figures."""
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 230, (256, 3)).astype(np.uint8)
    pal[WALL], pal[FLOOR], pal[CEIL], pal[WINDOW], pal[DOOR] = (210, 210, 210), (150, 120, 90), (240, 240, 240), \
        (140, 190, 230), (120, 90, 60)
    return pal[np.clip(inst, 0, 255)]
