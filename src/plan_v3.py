"""Architectural floor plan of the v3 room (PLAN_V3 milestone 4).

Input: the room box (src/room_box.py, walls axis-aligned in frame B), the furniture solids
(src/furniture.py) and the doors / windows found on the walls (openings_on_box). The drawing is built
once as a list of primitives (lines, polygons, circles, arcs, texts; metres) and then rendered to
PNG / SVG with matplotlib and written to DXF (R12 ASCII, own writer), so the three files agree.

Drawing rules
    orientation   axis-aligned, the longer room side horizontal
    walls         10 cm band outside the measured inner face; solid black where the side was detected,
                  hatched where it was inferred; gaps for doors (with leaf and 90 deg swing) and windows
                  (three parallel lines)
    furniture     architectural symbols per class (bed with pillows and fold, sofa with back and arms,
                  armchair, chair, round or rectangular table, wardrobe with door swings, cabinet,
                  shelf, lamp, plant, TV unit with screen, bathtub, toilet, sink, counter, hob,
                  fridge); other classes: labelled rectangle. The "back" of a symbol is the side
                  nearest a wall. Instances smaller than MIN_AREA are hidden
    labels        inside the symbol if the text fits, else at the first free position around it with a
                  leader line; positions are checked against all labels and symbols placed so far
    dimensions    room width and depth outside the walls, the clearance in front of the main items
                  (bed, sofa, table) to the facing wall, scale bar, area, "approximate" note
"""

from __future__ import annotations

import numpy as np

WALL_T = 0.10
MIN_AREA = 0.05            # m^2: smaller furniture is not drawn
LABEL_PT = 7.0
DIM_PT = 7.0
INK, DIM, HATCH, FILL = "#1b1b1b", "#2d5f8a", "#555555", "#ffffff"
OPENING_CLASSES = {"door": "door", "screen door": "door", "double door": "door", "windowpane": "window"}


# --------------------------------------------------------------------------- #
# Openings
# --------------------------------------------------------------------------- #
def openings_on_box(dist: np.ndarray, class_map: np.ndarray, id2label: dict, box: dict, bin_m: float = 0.05,
                    min_gap: float = 0.25) -> list[dict]:
    """Doors / windows: their pixels' rays are intersected with the box walls (camera inside the box);
    pixels whose observed distance is clearly shorter than the wall (a wardrobe door, a mirror on
    furniture) are ignored. Per wall, runs of 5 cm bins with door/window hits (gaps < min_gap joined)
    become openings. A door must reach within 30 cm of the floor, a window not."""
    from . import equirect as eq
    from .floorplan import class_ids
    H, W = dist.shape
    out = []
    dirs = (eq.pixel_dirs(W, H) @ box["R"].T) @ box["Q"]
    (x0, x1), (z0, z1) = box["x"], box["z"]
    walls = [("x1", 0, x1), ("x0", 0, x0), ("z1", 2, z1), ("z0", 2, z0)]
    kinds = {}
    for name, kind in OPENING_CLASSES.items():
        for i in class_ids(id2label, [name]):
            kinds[i] = kind
    m = np.isin(class_map, list(kinds))
    if m.sum() < 50:
        return out
    d = dirs[m]
    obs = dist[m]
    cls = np.array([kinds[c] for c in class_map[m]])
    t = np.full((len(d), 4), np.inf)
    for k, (_, ax, pos) in enumerate(walls):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = pos / d[:, ax]
        t[:, k] = np.where(tt > 0, tt, np.inf)
    k = np.argmin(t, 1)
    tw = t[np.arange(len(d)), k]
    P = d * tw[:, None]
    hgt = box["floor_y"] - P[:, 1]
    ok = np.isfinite(tw) & (obs > 0.85 * tw) & (hgt > -0.05) & (hgt < box["ceiling_height"] + 0.05)
    for wi, (wname, ax, pos) in enumerate(walls):
        s_ax = 2 if ax == 0 else 0
        sel = ok & (k == wi)
        if sel.sum() < 30:
            continue
        s = P[sel, s_ax]
        lo = (box["z"] if s_ax == 2 else box["x"])
        nb = int(np.ceil((lo[1] - lo[0]) / bin_m)) + 1
        idx = np.clip(((s - lo[0]) / bin_m).astype(int), 0, nb - 1)
        cnt = np.bincount(idx, minlength=nb)
        occ = cnt >= 3
        gap = int(round(min_gap / bin_m))
        runs, start, last = [], None, None
        for i in range(nb):
            if occ[i]:
                if start is None:
                    start = i
                elif i - last > gap:
                    runs.append((start, last))
                    start = i
                last = i
        if start is not None:
            runs.append((start, last))
        for a, b in runs:
            r = sel.copy()
            r[sel] = (idx >= a) & (idx <= b)
            width = (b - a + 1) * bin_m
            kind = "door" if (cls[r] == "door").mean() >= 0.5 else "window"
            low = np.percentile(hgt[r], 3)
            if kind == "door" and low > 0.3:
                kind = "window"
            if kind == "window" and low < 0.15:
                kind = "door"            # glazed door reaching the floor
            if width < (0.6 if kind == "door" else 0.4):
                continue
            width = min(width, 1.8 if kind == "door" else 3.0)
            c = lo[0] + (a + b + 1) / 2 * bin_m
            out.append({"kind": kind, "wall": wname, "s0": c - width / 2, "s1": c + width / 2, "width": width,
                        "pixels": int(r.sum())})
    return out


# --------------------------------------------------------------------------- #
# Primitive list
# --------------------------------------------------------------------------- #
class Drawing:
    def __init__(self):
        self.items = []

    def line(self, p0, p1, layer="FURNITURE", lw=0.8, color=INK, ls="-"):
        self.items.append({"t": "line", "p": [np.asarray(p0, float), np.asarray(p1, float)], "layer": layer,
                           "lw": lw, "color": color, "ls": ls})

    def poly(self, pts, layer="FURNITURE", lw=0.8, color=INK, fill=None, closed=True):
        self.items.append({"t": "poly", "p": np.asarray(pts, float), "layer": layer, "lw": lw, "color": color,
                           "fill": fill, "closed": closed})

    def circle(self, c, r, layer="FURNITURE", lw=0.8, color=INK, fill=None):
        self.items.append({"t": "circle", "c": np.asarray(c, float), "r": float(r), "layer": layer, "lw": lw,
                           "color": color, "fill": fill})

    def arc(self, c, r, a0, a1, layer="FURNITURE", lw=0.6, color=INK, ls="-"):
        """Arc from angle a0 to a1 (degrees, counter-clockwise)."""
        self.items.append({"t": "arc", "c": np.asarray(c, float), "r": float(r), "a0": float(a0), "a1": float(a1),
                           "layer": layer, "lw": lw, "color": color, "ls": ls})

    def text(self, p, s, pt=LABEL_PT, layer="TEXT", ha="center", va="center", rot=0.0, color=INK, bold=False):
        self.items.append({"t": "text", "p": np.asarray(p, float), "s": s, "pt": pt, "layer": layer, "ha": ha,
                           "va": va, "rot": rot, "color": color, "bold": bold})

    def bounds(self):
        pts = []
        for it in self.items:
            if it["t"] in ("line", "poly"):
                pts.append(np.atleast_2d(it["p"]))
            elif it["t"] in ("circle", "arc"):
                pts.append(np.array([it["c"] - it["r"], it["c"] + it["r"]]))
            else:
                pts.append(it["p"][None])
        P = np.vstack(pts)
        return P.min(0), P.max(0)


def _clip(p0, p1, lo, hi):
    """Liang-Barsky clip of segment p0-p1 to the axis-aligned box [lo, hi]; None if outside."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for ax in range(2):
        for p, q in ((-d[ax], p0[ax] - lo[ax]), (d[ax], hi[ax] - p0[ax])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    if t0 > t1:
        return None
    return p0 + t0 * d, p0 + t1 * d


def hatch_rect(D: Drawing, lo, hi, spacing=0.08, layer="WALLS"):
    """45 degree hatch lines clipped to an axis-aligned rectangle (written as plain lines, also in DXF)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = (hi - lo).sum()
    for k in np.arange(-span, span, spacing):
        a = np.array([lo[0] + k, lo[1]])
        b = a + np.array([span, span])
        c = _clip(a, b, lo, hi)
        if c is not None:
            D.line(c[0], c[1], layer=layer, lw=0.4, color=HATCH)


# --------------------------------------------------------------------------- #
# Furniture symbols (local frame: s along the back, t from the back (0) to the front (Dp))
# --------------------------------------------------------------------------- #
class Frame:
    def __init__(self, c, a, b, W, Dp):
        self.c, self.a, self.b, self.W, self.D = np.asarray(c, float), np.asarray(a, float), np.asarray(b, float), W, Dp

    def __call__(self, s, t):
        return self.c + s * self.a + (t - self.D / 2) * self.b

    def rect(self, s0, s1, t0, t1):
        return np.array([self(s0, t0), self(s1, t0), self(s1, t1), self(s0, t1)])

    def bbox(self):
        r = self.rect(-self.W / 2, self.W / 2, 0, self.D)
        return r.min(0), r.max(0)


def _rounded(F: Frame, s0, s1, t0, t1, rad, n=6):
    rad = min(rad, (s1 - s0) / 2, (t1 - t0) / 2)
    pts = []
    for (cs, ct, a0) in ((s1 - rad, t0 + rad, -90), (s1 - rad, t1 - rad, 0), (s0 + rad, t1 - rad, 90), (s0 + rad, t0 + rad, 180)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            pts.append(F(cs + rad * np.cos(a), ct + rad * np.sin(a)))
    return np.array(pts)


def _ellipse(F: Frame, cs, ct, rs, rt, n=28):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([F(cs + rs * np.cos(x), ct + rt * np.sin(x)) for x in a])


def symbol(D: Drawing, F: Frame, label: str) -> None:
    W, Dp, h = F.W, F.D, F.W / 2
    outline = F.rect(-h, h, 0, Dp)
    L = label
    if L == "bed":
        D.poly(outline, fill=FILL, lw=1.0)
        D.poly(F.rect(-h, h, 0, 0.07 * Dp), fill="#e8e8e8", lw=0.8)                 # headboard
        n = 2 if W > 1.25 else 1
        pw = (W - 0.15 * (n + 1)) / n
        for i in range(n):
            s0 = -h + 0.15 + i * (pw + 0.15)
            D.poly(_rounded(F, s0, s0 + pw, 0.1 * Dp, 0.24 * Dp, 0.06), lw=0.6)    # pillows
        D.line(F(-h, 0.32 * Dp), F(h, 0.32 * Dp), lw=0.6)                            # blanket fold
        D.line(F(h - 0.35 * W, 0.32 * Dp), F(h, 0.32 * Dp + 0.35 * W), lw=0.5)       # turned-down corner
    elif L in ("sofa", "armchair"):
        arm = min(0.15 * W if L == "sofa" else 0.18 * W, 0.22)
        D.poly(outline, fill=FILL, lw=1.0)
        D.poly(F.rect(-h, h, 0, 0.25 * Dp), lw=0.7)                                   # back
        D.poly(F.rect(-h, -h + arm, 0.25 * Dp, Dp), lw=0.7)                           # arms
        D.poly(F.rect(h - arm, h, 0.25 * Dp, Dp), lw=0.7)
        seats = max(1, int(round((W - 2 * arm) / 0.7))) if L == "sofa" else 1
        for i in range(1, seats):
            s = -h + arm + i * (W - 2 * arm) / seats
            D.line(F(s, 0.25 * Dp), F(s, Dp), lw=0.5)
    elif L in ("chair", "swivel chair", "seat", "stool", "ottoman", "bench"):
        if L in ("stool", "ottoman") and abs(W - Dp) < 0.15:
            D.circle(F(0, Dp / 2), min(W, Dp) / 2, fill=FILL)
        else:
            D.poly(_rounded(F, -h, h, 0.15 * Dp, Dp, 0.05), fill=FILL, lw=0.9)        # seat
            D.poly(F.rect(-h, h, 0, 0.15 * Dp), fill="#e8e8e8", lw=0.8)              # back
    elif L in ("table", "coffee table", "desk", "pool table", "kitchen island"):
        if L in ("table", "coffee table") and max(W, Dp) / max(min(W, Dp), 1e-6) < 1.15:
            D.circle(F(0, Dp / 2), min(W, Dp) / 2, fill=FILL, lw=1.0)
            D.circle(F(0, Dp / 2), min(W, Dp) / 2 - 0.04, lw=0.4)
        else:
            D.poly(outline, fill=FILL, lw=1.0)
            D.poly(F.rect(-h + 0.04, h - 0.04, 0.04, Dp - 0.04), lw=0.4)
    elif L == "wardrobe":
        D.poly(outline, fill=FILL, lw=1.0)
        D.line(F(-h, 0.08 * Dp), F(h, 0.08 * Dp), lw=0.5)                             # hanging rail side
        nd = max(1, int(round(W / 0.5)))
        dw = W / nd
        for i in range(nd):                                                          # door swings (dashed)
            s0 = -h + i * dw
            hinge = F(s0, Dp) if i % 2 == 0 else F(s0 + dw, Dp)
            tip = hinge + F.b * dw
            D.line(hinge, tip, lw=0.4, ls="--")
            a_back = np.degrees(np.arctan2(*(F.a if i % 2 == 0 else -F.a)[::-1]))
            a_out = np.degrees(np.arctan2(*F.b[::-1]))
            a0, a1 = sorted([a_back, a_out])
            if a1 - a0 > 180:
                a0, a1 = a1, a0 + 360
            D.arc(hinge, dw, a0, a1, lw=0.4, ls="--")
    elif L in ("cabinet", "chest of drawers", "buffet"):
        D.poly(outline, fill=FILL, lw=1.0)
        D.line(F(-h, 0), F(h, Dp), lw=0.4)
        D.line(F(-h, Dp), F(h, 0), lw=0.4)
    elif L in ("shelf", "bookcase"):
        D.poly(outline, fill=FILL, lw=1.0)
        for k in (0.33, 0.66):
            D.line(F(-h, k * Dp), F(h, k * Dp), lw=0.4)
    elif L == "lamp":
        r = min(W, Dp) / 2
        D.circle(F(0, Dp / 2), r, fill=FILL, lw=0.8)
        D.line(F(-r * 0.7, Dp / 2 - r * 0.7), F(r * 0.7, Dp / 2 + r * 0.7), lw=0.5)
        D.line(F(-r * 0.7, Dp / 2 + r * 0.7), F(r * 0.7, Dp / 2 - r * 0.7), lw=0.5)
    elif L in ("plant", "pot", "flower"):
        r = min(W, Dp) / 2
        c = (0, Dp / 2)
        D.circle(F(*c), r * 0.35, fill=FILL, lw=0.7)
        for a in np.linspace(0, 2 * np.pi, 7, endpoint=False):
            D.circle(F(c[0] + 0.6 * r * np.cos(a), c[1] + 0.6 * r * np.sin(a)), r * 0.4, lw=0.4)
    elif L == "television receiver":
        D.poly(outline, fill=FILL, lw=0.9)
        D.poly(F.rect(-0.4 * W, 0.4 * W, 0.05 * Dp, 0.05 * Dp + min(0.06, 0.3 * Dp)), fill=INK, lw=0.5)  # screen
    elif L in ("bathtub", "shower"):
        D.poly(outline, fill=FILL, lw=1.0)
        if L == "bathtub":
            D.poly(_rounded(F, -h + 0.07, h - 0.07, 0.07, Dp - 0.07, 0.2), lw=0.6)
            D.circle(F(-h + 0.22, Dp / 2), 0.03, lw=0.5)
        else:
            D.line(F(-h, 0), F(h, Dp), lw=0.4)
            D.line(F(-h, Dp), F(h, 0), lw=0.4)
    elif L == "toilet":
        D.poly(F.rect(-0.45 * W, 0.45 * W, 0, 0.28 * Dp), fill=FILL, lw=0.9)          # tank
        D.poly(_ellipse(F, 0, 0.28 * Dp + 0.36 * Dp, 0.42 * W, 0.36 * Dp), fill=FILL, lw=0.9)
    elif L == "sink":
        D.poly(outline, fill=FILL, lw=1.0)
        D.poly(_ellipse(F, 0, 0.55 * Dp, 0.32 * W if W < 0.9 else 0.25, 0.3 * Dp), lw=0.6)
        D.circle(F(0, 0.12 * Dp), 0.02, lw=0.5)
    elif L in ("stove", "oven"):
        D.poly(outline, fill=FILL, lw=1.0)
        for ss in (-0.22, 0.22):
            for tt in (0.3, 0.7):
                D.circle(F(ss * W, tt * Dp), 0.13 * min(W, Dp), lw=0.6)
    elif L in ("counter", "countertop"):
        D.poly(outline, fill=FILL, lw=1.0)
        D.line(F(-h, 0.85 * Dp), F(h, 0.85 * Dp), lw=0.4, ls="--")                    # front edge of wall units
    elif L in ("refrigerator", "washer", "dishwasher"):
        D.poly(outline, fill=FILL, lw=1.0)
        D.poly(F.rect(-h + 0.03, h - 0.03, 0.03, Dp - 0.03), lw=0.4)
        D.line(F(-h, Dp), F(-h + 0.12 * W, Dp - 0.12 * Dp), lw=0.4)
    else:
        D.poly(outline, fill=FILL, lw=1.0)


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
SHORT = {"television receiver": "TV", "chest of drawers": "drawers", "refrigerator": "fridge", "buffet": "sideboard",
         "kitchen island": "island", "swivel chair": "chair", "coffee table": "coffee tbl", "countertop": "counter",
         "pool table": "pool table"}


def _to_plan(p, rot):
    """frame B plan coords (x, z) -> drawing coords; rot = rotate 90 deg so the long side is horizontal."""
    p = np.asarray(p, float)
    return np.stack([p[..., 1], -p[..., 0]], -1) if rot else p


def build_drawing(box: dict, solids: list[dict], openings: list[dict], title: str | None = None,
                  note: str = "Approximate: reconstructed from one generated 360° panorama") -> tuple[Drawing, dict]:
    from .furniture import _corners
    (x0, x1), (z0, z1) = box["x"], box["z"]
    rot = (z1 - z0) > (x1 - x0)
    Dw = Drawing()
    corners = _to_plan(np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]]), rot)
    lo, hi = corners.min(0), corners.max(0)
    Wr, Hr = hi - lo
    # walls: band per side, measured inner face; map side names to drawing sides
    side_of = {}
    for name, p in (("x0", [x0, (z0 + z1) / 2]), ("x1", [x1, (z0 + z1) / 2]), ("z0", [(x0 + x1) / 2, z0]),
                    ("z1", [(x0 + x1) / 2, z1])):
        q = _to_plan(np.array(p), rot)
        side_of[name] = "left" if abs(q[0] - lo[0]) < 1e-6 else "right" if abs(q[0] - hi[0]) < 1e-6 else \
            "bottom" if abs(q[1] - lo[1]) < 1e-6 else "top"
    det = {"+x": "x1", "-x": "x0", "+z": "z1", "-z": "z0"}
    detected = {side_of[v]: box["sides"][k]["detected"] for k, v in det.items()}
    t = WALL_T                                       # band per side: (lower corner, upper corner, axis it runs along)
    bands = {"bottom": (np.array([lo[0] - t, lo[1] - t]), np.array([hi[0] + t, lo[1]]), 0),
             "top": (np.array([lo[0] - t, hi[1]]), np.array([hi[0] + t, hi[1] + t]), 0),
             "left": (np.array([lo[0] - t, lo[1]]), np.array([lo[0], hi[1]]), 1),
             "right": (np.array([hi[0], lo[1]]), np.array([hi[0] + t, hi[1]]), 1)}
    gaps = {k: [] for k in bands}
    for o in openings:
        side = side_of[o["wall"]]
        ax = bands[side][2]                          # coordinate along that wall in drawing coords
        # opening s-range lives on the B axis along the wall; map its two ends
        if o["wall"] in ("x0", "x1"):
            pa = _to_plan(np.array([x0 if o["wall"] == "x0" else x1, o["s0"]]), rot)
            pb = _to_plan(np.array([x0 if o["wall"] == "x0" else x1, o["s1"]]), rot)
        else:
            pa = _to_plan(np.array([o["s0"], z0 if o["wall"] == "z0" else z1]), rot)
            pb = _to_plan(np.array([o["s1"], z0 if o["wall"] == "z0" else z1]), rot)
        g0, g1 = sorted([pa[ax], pb[ax]])
        gaps[side].append((g0, g1, o["kind"]))
    stats = {"walls_detected": int(sum(detected.values())), "openings": len(openings)}
    for side, (blo, bhi, ax) in bands.items():
        segs, cur = [], blo[ax]
        for g0, g1, kind in sorted(gaps[side]):
            segs.append((cur, max(cur, g0)))
            cur = max(cur, g1)
        segs.append((cur, bhi[ax]))
        for s0, s1 in segs:
            if s1 - s0 < 1e-3:
                continue
            a, b = blo.copy(), bhi.copy()
            a[ax], b[ax] = s0, s1
            if detected[side]:
                Dw.poly(np.array([a, [b[0], a[1]], b, [a[0], b[1]]]), layer="WALLS", fill=INK, lw=0.8)
            else:
                Dw.poly(np.array([a, [b[0], a[1]], b, [a[0], b[1]]]), layer="WALLS", fill=FILL, lw=0.8)
                hatch_rect(Dw, a, b)
        inner = {"bottom": lo[1], "top": hi[1], "left": lo[0], "right": hi[0]}[side]
        inward = {"bottom": 1, "top": -1, "left": 1, "right": -1}[side]
        for g0, g1, kind in gaps[side]:
            w = g1 - g0
            if kind == "window":
                for f in (0.0, 0.5, 1.0):
                    off = inner - inward * t * f
                    p0 = [g0, off] if ax == 0 else [off, g0]
                    p1 = [g1, off] if ax == 0 else [off, g1]
                    Dw.line(p0, p1, layer="OPENINGS", lw=0.6)
                e0 = [g0, inner] if ax == 0 else [inner, g0]
                e1 = [g0, inner - inward * t] if ax == 0 else [inner - inward * t, g0]
                Dw.line(e0, e1, layer="OPENINGS", lw=0.6)
                e0 = [g1, inner] if ax == 0 else [inner, g1]
                e1 = [g1, inner - inward * t] if ax == 0 else [inner - inward * t, g1]
                Dw.line(e0, e1, layer="OPENINGS", lw=0.6)
            else:                                       # door: leaf + 90 deg swing; > 1.1 m: double door
                into = np.array([0.0, inward] if ax == 0 else [inward, 0.0])
                leaves = [(g0, 1.0, w)] if w <= 1.1 else [(g0, 1.0, w / 2), (g1, -1.0, w / 2)]
                for h0, sgn, lw_ in leaves:
                    hinge = np.array([h0, inner] if ax == 0 else [inner, h0])
                    along = sgn * np.array([1.0, 0.0] if ax == 0 else [0.0, 1.0])
                    Dw.line(hinge, hinge + into * lw_, layer="OPENINGS", lw=0.8)
                    a_al = np.degrees(np.arctan2(along[1], along[0]))
                    a_in = np.degrees(np.arctan2(into[1], into[0]))
                    a0, a1 = sorted([a_al, a_in])
                    if a1 - a0 > 180:
                        a0, a1 = a1, a0 + 360
                    Dw.arc(hinge, lw_, a0, a1, layer="OPENINGS", lw=0.5)
    # furniture
    walls_d = {"left": lo[0], "right": hi[0], "bottom": lo[1], "top": hi[1]}
    placed = []
    sol = sorted([s for s in solids if s["e"][0] * s["e"][1] >= MIN_AREA], key=lambda s: -s["e"][0] * s["e"][1])
    stats["furniture_hidden_small"] = len(solids) - len(sol)
    for s in sol:
        c = _to_plan(_corners(s), rot)
        ctr = c.mean(0)
        e1, e2 = c[1] - c[0], c[3] - c[0]
        best = None                                     # the back of the symbol = the side closest to a wall
        sides = [(c[0], c[1], e2), (c[1], c[2], -e1), (c[2], c[3], -e2), (c[3], c[0], e1)]   # (p, q, inward dir)
        for p, q, inward in sides:
            mid = (p + q) / 2
            n_out = -inward / np.linalg.norm(inward)
            dist_w = np.inf
            for side, val in walls_d.items():
                ax = 0 if side in ("left", "right") else 1
                wn = np.array([-1.0, 0.0] if side == "left" else [1.0, 0.0] if side == "right" else
                              [0.0, -1.0] if side == "bottom" else [0.0, 1.0])
                if n_out @ wn > 0.9:
                    dist_w = min(dist_w, abs(mid[ax] - val))
            if best is None or dist_w < best[0]:
                best = (dist_w, p, q, inward)
        _, p, q, inward = best
        a = (q - p) / np.linalg.norm(q - p)
        Dp = np.linalg.norm(inward)
        b = inward / Dp
        F = Frame(ctr, a, b, float(np.linalg.norm(q - p)), float(Dp))
        symbol(Dw, F, s["label"])
        placed.append({"F": F, "label": SHORT.get(s["label"], s["label"]), "solid": s, "back_gap": best[0]})
    return Dw, {"rot": rot, "lo": lo, "hi": hi, "placed": placed, "stats": stats, "title": title, "note": note,
                "area": float((x1 - x0) * (z1 - z0)), "W": float(Wr), "H": float(Hr), "detected": detected}


# --------------------------------------------------------------------------- #
# Labels, dimensions and rendering
# --------------------------------------------------------------------------- #
def _text_box(p, s, pt, m_per_pt, ha="center", va="center"):
    w, h = 0.58 * pt * len(s) * m_per_pt, 1.25 * pt * m_per_pt
    x = p[0] - (w / 2 if ha == "center" else 0 if ha == "left" else w)
    y = p[1] - (h / 2 if va == "center" else 0 if va == "bottom" else h)
    return np.array([x, y]), np.array([x + w, y + h])


def _hit(a, b):
    return not (a[1][0] <= b[0][0] or b[1][0] <= a[0][0] or a[1][1] <= b[0][1] or b[1][1] <= a[0][1])


def annotate(Dw: Drawing, info: dict, m_per_pt: float) -> dict:
    lo, hi = info["lo"], info["hi"]
    obstacles = [p["F"].bbox() for p in info["placed"]]
    labels = []
    fails = 0
    for i, p in enumerate(info["placed"]):
        F, s = p["F"], p["label"]
        ctr = F(0, F.D / 2)
        box_in = _text_box(ctr, s, LABEL_PT, m_per_pt)
        flo, fhi = F.bbox()
        fits = box_in[0][0] > flo[0] + 0.02 and box_in[1][0] < fhi[0] - 0.02 and box_in[0][1] > flo[1] + 0.02 \
            and box_in[1][1] < fhi[1] - 0.02
        if fits and not any(_hit(box_in, l_) for l_ in labels):
            Dw.text(ctr, s)
            labels.append(box_in)
            continue
        done = False
        for k in range(16):
            ang = np.radians(90 + 45 * k / 2)
            for rr in (0.25, 0.45, 0.7):
                q = ctr + (np.linalg.norm(fhi - flo) / 2 + rr) * np.array([np.cos(ang), np.sin(ang)])
                bx = _text_box(q, s, LABEL_PT, m_per_pt)
                if bx[0][0] < lo[0] - 0.9 or bx[1][0] > hi[0] + 0.9 or bx[0][1] < lo[1] - 0.9 or bx[1][1] > hi[1] + 0.9:
                    continue
                if any(_hit(bx, l_) for l_ in labels) or any(_hit(bx, o) for j, o in enumerate(obstacles)):
                    continue
                edge = np.clip(q, flo, fhi)
                Dw.line(edge, q - (q - edge) / max(np.linalg.norm(q - edge), 1e-6) * (bx[1][1] - bx[0][1]) / 2,
                        layer="TEXT", lw=0.4, color=DIM)
                Dw.text(q, s)
                labels.append(bx)
                done = True
                break
            if done:
                break
        if not done:
            fails += 1
    # dimensions: room
    off = WALL_T + 0.35

    def dim(p0, p1, normal, text):
        p0, p1, n = np.asarray(p0, float), np.asarray(p1, float), np.asarray(normal, float)
        Dw.line(p0, p1, layer="DIMS", lw=0.5, color=DIM)
        for pp in (p0, p1):
            Dw.line(pp - 0.08 * n, pp + 0.08 * n, layer="DIMS", lw=0.5, color=DIM)
            tick = np.array([1, 1]) * 0.05 * np.sign(n.sum() or 1)
            Dw.line(pp - tick, pp + tick, layer="DIMS", lw=0.9, color=DIM)
        mid = (p0 + p1) / 2 + n * 0.12
        rot = 90.0 if abs(p1[0] - p0[0]) < 1e-6 else 0.0
        Dw.text(mid, text, pt=DIM_PT, layer="DIMS", rot=rot, color=DIM)

    dim([lo[0], lo[1] - off], [hi[0], lo[1] - off], [0, -1], f"≈ {hi[0] - lo[0]:.2f} m")
    dim([lo[0] - off, lo[1]], [lo[0] - off, hi[1]], [-1, 0], f"≈ {hi[1] - lo[1]:.2f} m")
    # clearances in front of the main items
    n_cl = 0
    for p in info["placed"]:
        if p["solid"]["label"] not in ("bed", "sofa", "table", "desk") or n_cl >= 2:
            continue
        F = p["F"]
        front = F(0, F.D)
        b = F.b
        ax = int(np.argmax(np.abs(b)))
        wall = hi[ax] if b[ax] > 0 else lo[ax]
        gap = abs(wall - front[ax])
        if gap < 0.3:
            continue
        end = front.copy()
        end[ax] = wall
        nrm = np.array([1.0, 0.0]) if ax == 1 else np.array([0.0, 1.0])
        s0, s1 = front + 0.12 * F.a, end + 0.12 * F.a
        dim(s0, s1, nrm, f"{gap:.2f}")
        n_cl += 1
    # scale bar, area, title, note
    sb = np.array([lo[0], lo[1] - off - 0.55])
    for k in range(2):
        Dw.poly([sb + [k, 0], sb + [k + 1, 0], sb + [k + 1, 0.06], sb + [k, 0.06]], layer="DIMS", lw=0.5,
                fill=INK if k == 0 else FILL)
    Dw.text(sb + [0, -0.12], "0", pt=6, layer="DIMS")
    Dw.text(sb + [1, -0.12], "1", pt=6, layer="DIMS")
    Dw.text(sb + [2, -0.12], "2 m", pt=6, layer="DIMS")
    Dw.text([hi[0], lo[1] - off - 0.5], f"Area ≈ {info['area']:.1f} m²", pt=8, layer="TEXT", ha="right", bold=True)
    if info["title"]:
        Dw.text([(lo[0] + hi[0]) / 2, hi[1] + WALL_T + 0.45], info["title"], pt=10, layer="TEXT", bold=True)
    Dw.text([(lo[0] + hi[0]) / 2, hi[1] + WALL_T + 0.2], info["note"] + "  ·  hatched walls: inferred", pt=6,
            layer="TEXT", color="#666666")
    return {"labels_placed": len(labels), "label_failures": fails, "clearances": n_cl}


def render(Dw: Drawing, png=None, svg=None, dpi: int = 150, scale_in_per_m: float = 1.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, Polygon
    lo, hi = Dw.bounds()
    lo, hi = lo - 0.3, hi + 0.3
    fig = plt.figure(figsize=((hi[0] - lo[0]) * scale_in_per_m, (hi[1] - lo[1]) * scale_in_per_m), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    for it in Dw.items:
        t = it["t"]
        if t == "line":
            ax.plot(*np.array(it["p"]).T, color=it["color"], lw=it["lw"], ls=it["ls"], solid_capstyle="butt")
        elif t == "poly":
            ax.add_patch(Polygon(it["p"], closed=it["closed"], fc=it["fill"] or "none", ec=it["color"], lw=it["lw"],
                                 joinstyle="miter"))
        elif t == "circle":
            ax.add_patch(Circle(it["c"], it["r"], fc=it["fill"] or "none", ec=it["color"], lw=it["lw"]))
        elif t == "arc":
            ax.add_patch(Arc(it["c"], 2 * it["r"], 2 * it["r"], theta1=it["a0"], theta2=it["a1"], color=it["color"],
                             lw=it["lw"], ls=it["ls"]))
        else:
            ax.text(*it["p"], it["s"], fontsize=it["pt"], ha=it["ha"], va=it["va"], rotation=it["rot"],
                    color=it["color"], fontweight="bold" if it["bold"] else "normal",
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85) if it["layer"] in ("TEXT", "DIMS") else None)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    if png:
        fig.savefig(png, facecolor="white", dpi=dpi)
    if svg:
        fig.savefig(svg, facecolor="white")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def write_dxf(Dw: Drawing, path, m_per_pt: float) -> None:
    """Minimal DXF R12 (ASCII): LINE, CIRCLE, ARC, TEXT and closed POLYLINEs, units = metres, one layer per kind."""
    out = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "6", "0", "ENDSEC",
           "0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", "6"]
    colors = {"WALLS": 7, "OPENINGS": 4, "FURNITURE": 8, "TEXT": 7, "DIMS": 5}
    for name, col in colors.items():
        out += ["0", "LAYER", "2", name, "70", "0", "62", str(col), "6", "CONTINUOUS"]
    out += ["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    f = lambda v: f"{float(v):.4f}"  # noqa: E731
    for it in Dw.items:
        L = it["layer"]
        if it["t"] == "line":
            (a, b) = it["p"]
            out += ["0", "LINE", "8", L, "10", f(a[0]), "20", f(a[1]), "30", "0", "11", f(b[0]), "21", f(b[1]), "31", "0"]
        elif it["t"] == "poly":
            out += ["0", "POLYLINE", "8", L, "66", "1", "70", "1" if it["closed"] else "0"]
            for p in it["p"]:
                out += ["0", "VERTEX", "8", L, "10", f(p[0]), "20", f(p[1]), "30", "0"]
            out += ["0", "SEQEND", "8", L]
        elif it["t"] == "circle":
            out += ["0", "CIRCLE", "8", L, "10", f(it["c"][0]), "20", f(it["c"][1]), "30", "0", "40", f(it["r"])]
        elif it["t"] == "arc":
            out += ["0", "ARC", "8", L, "10", f(it["c"][0]), "20", f(it["c"][1]), "30", "0", "40", f(it["r"]),
                    "50", f(it["a0"] % 360), "51", f(it["a1"] % 360)]
        else:
            h = it["pt"] * m_per_pt
            hj = {"left": 0, "center": 1, "right": 2}[it["ha"]]
            vj = {"bottom": 1, "center": 2, "top": 3}[it["va"]]
            s = it["s"].replace("≈", "~").replace("²", "2").replace("·", "-").replace("°", " deg")
            out += ["0", "TEXT", "8", L, "10", f(it["p"][0]), "20", f(it["p"][1]), "30", "0", "40", f(h), "1", s,
                    "50", f(it["rot"]), "72", str(hj), "73", str(vj), "11", f(it["p"][0]), "21", f(it["p"][1]), "31", "0"]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(out) + "\n")


def draw_plan(box: dict, solids: list[dict], openings: list[dict], png=None, svg=None, dxf=None,
              title: str | None = None, scale_in_per_m: float = 1.1, dpi: int = 150):
    """Whole milestone-4 plan: returns (RGB image, stats)."""
    Dw, info = build_drawing(box, solids, openings, title)
    m_per_pt = 1.0 / (72.0 * scale_in_per_m)
    st = annotate(Dw, info, m_per_pt)
    img = render(Dw, png, svg, dpi, scale_in_per_m)
    if dxf:
        write_dxf(Dw, dxf, m_per_pt)
    return img, {**info["stats"], **st, "rotated_90": bool(info["rot"]), "items": len(Dw.items)}
