"""Layout first (milestone 6): a parametric room layout decides the geometry before any image exists.

    room type + seed
      -> room size from realistic ranges, one door, one or two windows
      -> furniture as boxes, placed by rules per room type (against walls, facing each other,
         clearances, nothing in front of the door), checked for overlaps; seeded and repeatable
      -> camera: the free spot closest to the room centre, 1.4 m above the floor
      -> equirectangular DEPTH (distance) and SEMANTIC (ADE20K ids) maps rendered from the camera by
         exact ray-box intersection: the condition for the image model and the ground truth
      -> layout JSON and a dimensioned floor plan (drawn with src/floorplan.py)

World frame: x to the right, z forward (plan coordinates x, y), y up, floor at y = 0, the room is
[0, W] x [0, D]. Camera frame of the panorama (equirect.py): X right, Y down, Z forward.
"""

from __future__ import annotations

import json

import numpy as np

from . import equirect as eq

CAMERA_HEIGHT = 1.4
CAMERA_CLEARANCE = 0.45      # m from any furniture box
DOOR_W, DOOR_H, DOOR_CLEAR = 0.9, 2.1, 1.0
WINDOW_W, WINDOW_H, WINDOW_SILL = 1.2, 1.3, 0.9
GAP = 0.08                   # m between neighbouring boxes

# ADE20K class ids of the SegFormer model (analysis.SEG_MODEL)
ADE = {"wall": 0, "floor": 3, "ceiling": 5, "bed": 7, "windowpane": 8, "cabinet": 10, "door": 14, "table": 15,
       "plant": 17, "chair": 19, "sofa": 23, "shelf": 24, "armchair": 30, "wardrobe": 35, "bathtub": 37,
       "chest of drawers": 44, "counter": 45, "sink": 47, "refrigerator": 50, "coffee table": 64, "toilet": 65,
       "stove": 71, "kitchen island": 73, "television receiver": 89, "buffet": 99}

ROOMS = {  # room type: (width range, depth range, height range)
    "bedroom": ((3.2, 4.5), (3.5, 5.0), (2.6, 3.0)),
    "living room": ((4.0, 6.0), (4.5, 7.0), (2.7, 3.1)),
    "kitchen": ((3.0, 4.5), (3.5, 5.0), (2.6, 3.0)),
    "dining room": ((3.5, 5.0), (3.5, 5.5), (2.6, 3.0)),
    "bathroom": ((2.2, 3.0), (2.4, 3.5), (2.5, 2.8)),
}


# --------------------------------------------------------------------------- #
# Placement helpers
# --------------------------------------------------------------------------- #
def _box(label, ade, x0, z0, w, d, h):
    return {"label": label, "ade": ade, "x0": float(x0), "z0": float(z0), "x1": float(x0 + w), "z1": float(z0 + d),
            "h": float(h)}


def _against(wall: str, W: float, D: float, length: float, depth: float, t: float):
    """Footprint (x0, z0, w, d) of a box `length` wide along `wall` ("N" z=D, "S" z=0, "E" x=W, "W" x=0),
    `depth` deep into the room, its centre at fraction t along the wall."""
    if wall in ("N", "S"):
        x0 = t * (W - length)
        return x0, (D - depth if wall == "N" else 0.0), length, depth
    z0 = t * (D - length)
    return (W - depth if wall == "E" else 0.0), z0, depth, length


def _overlap(a, b, gap=GAP):
    return not (a["x1"] + gap <= b["x0"] or b["x1"] + gap <= a["x0"] or a["z1"] + gap <= b["z0"] or b["z1"] + gap <= a["z0"])


def _inside(b, W, D):
    return b["x0"] >= -1e-6 and b["z0"] >= -1e-6 and b["x1"] <= W + 1e-6 and b["z1"] <= D + 1e-6


class Layout:
    def __init__(self, room: str, seed: int):
        self.rng = np.random.default_rng(seed)
        (w0, w1), (d0, d1), (h0, h1) = ROOMS[room]
        r = self.rng
        self.room, self.seed = room, seed
        self.W, self.D, self.H = round(r.uniform(w0, w1), 2), round(r.uniform(d0, d1), 2), round(r.uniform(h0, h1), 2)
        self.boxes, self.openings = [], []
        self.door = self._door()

    # openings ------------------------------------------------------------- #
    def _door(self):
        wall = self.rng.choice(["S", "E", "W"])
        L = self.W if wall in ("N", "S") else self.D
        c = float(self.rng.uniform(DOOR_W / 2 + 0.2, L - DOOR_W / 2 - 0.2))
        d = {"kind": "door", "wall": wall, "c": c, "w": DOOR_W, "y0": 0.0, "y1": DOOR_H}
        self.openings.append(d)
        x0, z0, w, dd = _against(wall, self.W, self.D, DOOR_W + 0.4, DOOR_CLEAR, (c - DOOR_W / 2 - 0.2) / max(L - DOOR_W - 0.4, 1e-6))
        self.keepout = [_box("door clearance", -1, x0, z0, w, dd, 0)]
        return d

    def window(self, wall: str):
        L = self.W if wall in ("N", "S") else self.D
        if L < WINDOW_W + 0.6:
            return
        c = float(self.rng.uniform(WINDOW_W / 2 + 0.3, L - WINDOW_W / 2 - 0.3))
        if wall == self.door["wall"] and abs(c - self.door["c"]) < (DOOR_W + WINDOW_W) / 2 + 0.3:
            return
        self.openings.append({"kind": "window", "wall": wall, "c": c, "w": WINDOW_W, "y0": WINDOW_SILL,
                              "y1": WINDOW_SILL + WINDOW_H})

    # furniture ------------------------------------------------------------ #
    def place(self, label, ade, length, depth, h, walls=None, t=None, footprint=None, tries=40):
        """Place a box against one of `walls` (random position) or at a given footprint; returns it or None."""
        for _ in range(tries if footprint is None else 1):
            if footprint is None:
                wall = self.rng.choice(walls)
                tt = float(self.rng.uniform(0.05, 0.95)) if t is None else t
                fp_ = _against(wall, self.W, self.D, length, depth, tt)
            else:
                fp_ = footprint
            b = _box(label, ade, *fp_, h)
            if _inside(b, self.W, self.D) and not any(_overlap(b, o) for o in self.boxes + self.keepout):
                self.boxes.append(b)
                return b
        return None

    def free_walls(self):
        return [w for w in ("N", "S", "E", "W") if w != self.door["wall"]]

    # camera ------------------------------------------------------------- #
    def camera(self):
        best, bd = None, np.inf
        for x in np.linspace(0.6, self.W - 0.6, 25):
            for z in np.linspace(0.6, self.D - 0.6, 25):
                clear = all(x < b["x0"] - CAMERA_CLEARANCE or x > b["x1"] + CAMERA_CLEARANCE or
                            z < b["z0"] - CAMERA_CLEARANCE or z > b["z1"] + CAMERA_CLEARANCE for b in self.boxes)
                d = (x - self.W / 2) ** 2 + (z - self.D / 2) ** 2
                if clear and d < bd:
                    best, bd = (float(x), float(z)), d
        if best is None:
            best = (self.W / 2, self.D / 2)
        return np.array([best[0], CAMERA_HEIGHT, best[1]])

    def to_dict(self):
        return {"room": self.room, "seed": self.seed, "W": self.W, "D": self.D, "H": self.H,
                "camera": self.cam.tolist(), "boxes": self.boxes, "openings": self.openings}


def _bedroom(L: Layout):
    walls = L.free_walls()
    head = L.rng.choice(walls)
    bw = 1.6 if L.W > 3.4 and L.D > 3.8 else 1.4
    bed = None
    for _ in range(20):
        bed = L.place("bed", ADE["bed"], bw, 2.0, 0.55, walls=[head], t=float(L.rng.uniform(0.3, 0.7)))
        if bed:
            break
    if bed:                                                       # nightstands left and right of the headboard
        for side in (-1, 1):
            if head in ("N", "S"):
                x0 = bed["x0"] - 0.45 - GAP if side < 0 else bed["x1"] + GAP
                fp_ = (x0, bed["z0"] if head == "S" else bed["z1"] - 0.4, 0.45, 0.4)
            else:
                z0 = bed["z0"] - 0.45 - GAP if side < 0 else bed["z1"] + GAP
                fp_ = (bed["x0"] if head == "W" else bed["x1"] - 0.4, z0, 0.4, 0.45)
            L.place("nightstand", ADE["chest of drawers"], 0, 0, 0.55, footprint=fp_)
    L.place("wardrobe", ADE["wardrobe"], 1.8, 0.6, 2.1, walls=[w for w in walls if w != head] or walls)
    if L.rng.random() < 0.6:
        if L.place("desk", ADE["table"], 1.2, 0.6, 0.75, walls=walls):
            d = L.boxes[-1]                                       # chair in front of the desk, 10 cm away
            cx, cz, c = (d["x0"] + d["x1"]) / 2, (d["z0"] + d["z1"]) / 2, 0.46
            fp_ = ((cx - c / 2, d["z1"] + 0.1) if d["z0"] < 0.01 else (cx - c / 2, d["z0"] - 0.1 - c) if d["z1"] > L.D - 0.01
                   else (d["x1"] + 0.1, cz - c / 2) if d["x0"] < 0.01 else (d["x0"] - 0.1 - c, cz - c / 2))
            L.place("chair", ADE["chair"], 0, 0, 0.9, footprint=(*fp_, c, c))
    L.window(L.rng.choice(walls))


def _living(L: Layout):
    walls = L.free_walls()
    sofa = L.place("sofa", ADE["sofa"], 2.2, 0.9, 0.85, walls=walls, t=float(L.rng.uniform(0.3, 0.7)))
    if sofa:
        opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
        sw = "N" if sofa["z1"] >= L.D - 1e-6 else "S" if sofa["z0"] <= 1e-6 else "E" if sofa["x1"] >= L.W - 1e-6 else "W"
        L.place("TV unit", ADE["cabinet"], 1.8, 0.45, 0.55, walls=[opp[sw]], t=0.5) or \
            L.place("TV unit", ADE["cabinet"], 1.8, 0.45, 0.55, walls=[opp[sw]])
        cx, cz = (sofa["x0"] + sofa["x1"]) / 2, (sofa["z0"] + sofa["z1"]) / 2
        if sw in ("N", "S"):
            z0 = sofa["z0"] - 0.45 - 0.6 if sw == "N" else sofa["z1"] + 0.45
            L.place("coffee table", ADE["coffee table"], 0, 0, 0.45, footprint=(cx - 0.55, z0, 1.1, 0.6))
        else:
            x0 = sofa["x0"] - 0.45 - 0.6 if sw == "E" else sofa["x1"] + 0.45
            L.place("coffee table", ADE["coffee table"], 0, 0, 0.45, footprint=(x0, cz - 0.55, 0.6, 1.1))
    for _ in range(int(L.rng.integers(1, 3))):
        L.place("armchair", ADE["armchair"], 0.8, 0.8, 0.85, walls=walls)
    L.place("plant", ADE["plant"], 0.4, 0.4, 1.2, walls=walls, t=float(L.rng.choice([0.0, 1.0])))
    L.place("shelf", ADE["shelf"], 1.2, 0.35, 1.9, walls=walls)
    L.window(L.rng.choice(walls))
    L.window(L.rng.choice(walls))


def _kitchen(L: Layout):
    walls = L.free_walls()
    w1 = L.rng.choice(walls)
    run = (L.W if w1 in ("N", "S") else L.D) - 0.2
    pieces = [("counter", ADE["counter"], 0.9), ("stove", ADE["stove"], 0.6), ("counter", ADE["counter"], 0.6),
              ("sink", ADE["sink"], 0.8), ("counter", ADE["counter"], 0.9)]
    total = sum(p[2] for p in pieces)
    s = min(1.0, run / total)
    pos = 0.1
    for label, ade, length in pieces:                             # a counter run along one wall
        length *= s
        if w1 in ("N", "S"):
            fp_ = (pos, L.D - 0.6 if w1 == "N" else 0.0, length - 0.01, 0.6)
        else:
            fp_ = (L.W - 0.6 if w1 == "E" else 0.0, pos, 0.6, length - 0.01)
        L.boxes.append(_box(label, ade, *fp_, 0.9)) if not any(_overlap(_box(label, ade, *fp_, 0.9), o, 0.0) for o in L.keepout) else None
        pos += length
    L.place("fridge", ADE["refrigerator"], 0.7, 0.7, 1.85, walls=[w for w in walls if w != w1] or walls)
    if min(L.W, L.D) >= 3.4:
        L.place("island", ADE["kitchen island"], 0, 0, 0.9, footprint=(L.W / 2 - 0.8, L.D / 2 - 0.45 + 0.6, 1.6, 0.9))
    L.window(w1)


def _dining(L: Layout):
    walls = L.free_walls()
    tw, td = (1.8, 0.9) if L.W >= L.D else (0.9, 1.8)
    t = L.place("dining table", ADE["table"], 0, 0, 0.75, footprint=(L.W / 2 - tw / 2, L.D / 2 - td / 2 + 0.7, tw, td))
    if t:
        cw = 0.45
        if tw > td:
            for x in (t["x0"] + 0.2, t["x1"] - 0.2 - cw):
                L.place("chair", ADE["chair"], 0, 0, 0.9, footprint=(x, t["z0"] - cw - 0.1, cw, cw))
                L.place("chair", ADE["chair"], 0, 0, 0.9, footprint=(x, t["z1"] + 0.1, cw, cw))
        else:
            for z in (t["z0"] + 0.2, t["z1"] - 0.2 - cw):
                L.place("chair", ADE["chair"], 0, 0, 0.9, footprint=(t["x0"] - cw - 0.1, z, cw, cw))
                L.place("chair", ADE["chair"], 0, 0, 0.9, footprint=(t["x1"] + 0.1, z, cw, cw))
    L.place("sideboard", ADE["buffet"], 1.6, 0.45, 0.85, walls=walls)
    L.place("plant", ADE["plant"], 0.4, 0.4, 1.2, walls=walls, t=float(L.rng.choice([0.0, 1.0])))
    L.window(L.rng.choice(walls))


def _bathroom(L: Layout):
    walls = L.free_walls()
    L.place("bathtub", ADE["bathtub"], 1.7, 0.75, 0.55, walls=walls, t=float(L.rng.choice([0.0, 1.0])))
    L.place("toilet", ADE["toilet"], 0.4, 0.65, 0.75, walls=walls)
    L.place("vanity", ADE["sink"], 0.8, 0.5, 0.85, walls=walls)
    L.window(L.rng.choice(walls))


BUILDERS = {"bedroom": _bedroom, "living room": _living, "kitchen": _kitchen, "dining room": _dining,
            "bathroom": _bathroom}


def generate_layout(room: str, seed: int) -> dict:
    """Seeded layout of a room type; returns the layout dict (JSON-serialisable)."""
    L = Layout(room, seed)
    BUILDERS[room](L)
    L.cam = L.camera()
    return L.to_dict()


def room_type(prompt: str) -> str:
    """'a coastal style bedroom' -> 'bedroom'."""
    for r in sorted(ROOMS, key=len, reverse=True):
        if r in prompt:
            return r
    raise ValueError(prompt)


# --------------------------------------------------------------------------- #
# Rendering from the camera
# --------------------------------------------------------------------------- #
def render(layout: dict, w: int = 1024, h: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular distance (m) and ADE20K label map of the layout seen from its camera."""
    dist, label, _ = render_full(layout, w, h)
    return dist, label


DOOR_RECESS, WINDOW_RECESS = 0.12, 0.25     # m: openings drawn as recesses in the depth condition (shapes=True)


def _back_dir(b: dict, W: float, D: float) -> str:
    """Side of a furniture box that faces the nearest wall: '-x', '+x', '-z' or '+z'."""
    gaps = {"-x": b["x0"], "+x": W - b["x1"], "-z": b["z0"], "+z": D - b["z1"]}
    return min(gaps, key=gaps.get)


def proxy_parts(b: dict, W: float, D: float) -> list[tuple]:
    """Shaped proxy of a furniture box as part boxes (x0, y0, z0, x1, y1, z1), y up from the floor
    (PLAN_V3 milestone 7): bed = mattress + headboard, sofa / armchair = seat + back + arms,
    chair = seat + 4 legs + back, table / desk = top + 4 legs; other classes stay one box."""
    lab, h = b["label"], b["h"]
    back = _back_dir(b, W, D)
    ax = 0 if back[1] == "x" else 2                                   # depth axis (back -> front)
    lo_d, hi_d = (b["x0"], b["x1"]) if ax == 0 else (b["z0"], b["z1"])
    lo_w, hi_w = (b["z0"], b["z1"]) if ax == 0 else (b["x0"], b["x1"])
    sgn = -1 if back[0] == "-" else 1                                  # the back lies at lo_d if sgn < 0

    def part(d0, d1, w0, w1, y0, y1):
        """Depth fractions from the back (0..1), width fractions, absolute heights -> part box."""
        L = hi_d - lo_d
        a0, a1 = (lo_d + d0 * L, lo_d + d1 * L) if sgn < 0 else (hi_d - d1 * L, hi_d - d0 * L)
        Wd = hi_w - lo_w
        c0, c1 = lo_w + w0 * Wd, lo_w + w1 * Wd
        x0, x1, z0, z1 = (a0, a1, c0, c1) if ax == 0 else (c0, c1, a0, a1)
        return (x0, y0, z0, x1, y1, z1)

    leg = 0.05
    if lab == "bed":
        return [part(0, 1, 0, 1, 0, 0.5), part(0, 0.05, 0, 1, 0, 1.0)]
    if lab in ("sofa", "armchair"):
        arm = 0.12 if lab == "sofa" else 0.18
        return [part(0, 1, 0, 1, 0, 0.42), part(0, 0.25, 0, 1, 0, h), part(0, 1, 0, arm, 0, 0.62),
                part(0, 1, 1 - arm, 1, 0, 0.62)]
    if lab == "chair":
        Ld, Lw = (hi_d - lo_d), (hi_w - lo_w)
        fd, fw = leg / max(Ld, 1e-6), leg / max(Lw, 1e-6)
        return [part(0, 1, 0, 1, 0.42, 0.47), part(0, 0.12, 0, 1, 0.47, h)] + \
            [part(a, a + fd, c, c + fw, 0, 0.42) for a in (0, 1 - fd) for c in (0, 1 - fw)]
    if lab in ("desk", "dining table", "coffee table"):
        Ld, Lw = (hi_d - lo_d), (hi_w - lo_w)
        fd, fw = leg / max(Ld, 1e-6), leg / max(Lw, 1e-6)
        return [part(0, 1, 0, 1, h - 0.04, h)] + \
            [part(a, a + fd, c, c + fw, 0, h - 0.04) for a in (0.03, 0.97 - fd) for c in (0.03, 0.97 - fw)]
    return [(b["x0"], 0.0, b["z0"], b["x1"], h, b["z1"])]


def render_full(layout: dict, w: int = 1024, h: int = 512, shapes: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As render(), plus the instance map: index of the furniture box seen at each pixel (-1 = room shell).
    shapes=True (PLAN_V3 milestone 7): furniture as shaped proxies (proxy_parts) and doors / windows as
    recesses of DOOR_RECESS / WINDOW_RECESS behind the wall, so that the depth condition shows them."""
    W, D, H = layout["W"], layout["D"], layout["H"]
    C = np.array(layout["camera"])
    dc = eq.pixel_dirs(w, h).reshape(-1, 3)
    d = np.stack([dc[:, 0], -dc[:, 1], dc[:, 2]], 1)              # world direction (y up)
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    # room: exit distance of the box [0,W]x[0,H]x[0,D] (camera inside)
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([W, H, D])
    t_far = np.maximum((lo - C) * inv, (hi - C) * inv)            # per axis, the plane the ray leaves through
    axis = np.argmin(t_far, 1)
    dist = t_far[np.arange(len(d)), axis]
    P = C + d * dist[:, None]
    label = np.where(axis == 1, np.where(d[:, 1] < 0, ADE["floor"], ADE["ceiling"]), ADE["wall"]).astype(np.int32)
    for o in layout["openings"]:                                  # doors / windows painted on their wall
        wall = o["wall"]
        on = {"N": (axis == 2) & (d[:, 2] > 0), "S": (axis == 2) & (d[:, 2] < 0),
              "E": (axis == 0) & (d[:, 0] > 0), "W": (axis == 0) & (d[:, 0] < 0)}[wall]
        along = P[:, 0] if wall in ("N", "S") else P[:, 2]
        m = on & (np.abs(along - o["c"]) <= o["w"] / 2) & (P[:, 1] >= o["y0"]) & (P[:, 1] <= o["y1"])
        label[m] = ADE["door"] if o["kind"] == "door" else ADE["windowpane"]
        if shapes:                                                 # recess: the wall plane moves back
            k = 2 if wall in ("N", "S") else 0
            pos = {"N": D, "S": 0.0, "E": W, "W": 0.0}[wall] + (1 if wall in ("N", "E") else -1) * \
                (DOOR_RECESS if o["kind"] == "door" else WINDOW_RECESS)
            dist[m] = (pos - C[k]) * inv[m, k]
    inst = np.full(len(d), -1, np.int32)
    for bi, b in enumerate(layout["boxes"]):                      # furniture: entry distance of each box / part
        parts = proxy_parts(b, W, D) if shapes else [(b["x0"], 0.0, b["z0"], b["x1"], b["h"], b["z1"])]
        for x0, y0, z0, x1, y1, z1 in parts:
            blo, bhi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
            t1, t2 = (blo - C) * inv, (bhi - C) * inv
            tn, tf = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
            hit = (tf >= np.maximum(tn, 0)) & (tn > 0) & (tn < dist)
            dist[hit], label[hit], inst[hit] = tn[hit], b["ade"], bi
    return dist.reshape(h, w).astype(np.float32), label.reshape(h, w), inst.reshape(h, w)


def control_image(dist: np.ndarray) -> np.ndarray:
    """ControlNet depth condition: inverse depth scaled to 0-255 (near = white), 3 channels."""
    inv = 1.0 / np.maximum(dist, 0.1)
    lo, hi = np.percentile(inv, 0.5), np.percentile(inv, 99.5)
    g = (255 * np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


# --------------------------------------------------------------------------- #
# Ground-truth floor plan (drawn with floorplan.draw_floorplan)
# --------------------------------------------------------------------------- #
def plan_dict(layout: dict) -> dict:
    """The layout in the plan format of src/floorplan.py, in the panorama's plan coordinates
    (camera at the origin, x right, y forward)."""
    from . import floorplan as fp
    C = np.array(layout["camera"])
    W, D = layout["W"], layout["D"]
    off = np.array([C[0], C[2]])
    corners = np.array([[0, 0], [W, 0], [W, D], [0, D]], float) - off
    room = {"corners": corners, "width": W, "depth": D, "area": W * D, "inferred": [False] * 4, "n_observed_sides": 4}
    walls = fp.room_edges(room)
    side_of = {"S": 0, "E": 1, "N": 2, "W": 3}
    openings = []
    for o in layout["openings"]:
        wi = side_of[o["wall"]]
        w = walls[wi]
        # position along the wall segment p0 -> p1 (S: x increasing, E: z increasing, N: x decreasing, W: z decreasing)
        L = w["length"]
        t = o["c"] if o["wall"] in ("S", "E") else L - o["c"]
        openings.append({"kind": o["kind"], "wall": wi, "t0": t - o["w"] / 2, "t1": t + o["w"] / 2, "width": o["w"]})
    furn = []
    for b in layout["boxes"]:
        r = np.array([[b["x0"], b["z0"]], [b["x1"], b["z0"]], [b["x1"], b["z1"]], [b["x0"], b["z1"]]]) - off
        furn.append({"rect": r, "observed": r, "completed": False, "centroid": r.mean(0), "label": b["label"]})

    class _G:                                                     # empty floor grid (no tint)
        x0 = y0 = 0.0
        w = h = 1
        cell = 0.01
    return {"walls": walls, "furniture": furn, "openings": openings, "room": room, "grid": _G(),
            "floor_occ": np.zeros((1, 1), np.uint8), "fov_deg": 360.0, "visible_floor_m2": W * D}


def save_json(layout: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=1)
