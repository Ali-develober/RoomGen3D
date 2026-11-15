"""v6 layout generator (PLAN_V6 milestone 1): prompt -> room type + style -> room size, door, windows and
oriented furniture placed by rules. The layout is the ground truth of everything downstream (proxy scene,
depth condition, cameras, floor plan), so geometry is known by construction instead of estimated.

World frame (as src/layout.py): x right, y up, z forward; the room is [0, W] x [0, D] (plan x, y),
floor y = 0, ceiling y = H. Walls: S (z = 0), N (z = D), W (x = 0), E (x = W).

Furniture piece: {"cls", "label", "cx", "cz", "w", "d", "h", "y0", "yaw", "wall", "group"}
    w = width across its front, d = depth (back to front), h = height, y0 = bottom height (0 = floor)
    yaw in {0, 90, 180, 270}: the direction the FRONT faces, 0 = +z, 90 = +x, 180 = -z, 270 = -x
    (wall-aligned by construction); wall = the wall its back touches (or None); group = id of the main
    piece it belongs to (nightstands -> bed, chairs -> table, TV unit / coffee table / armchairs -> sofa).

Rules per room type (see the builders): bed against a wall with nightstands on both sides; dining table
centred with chairs around it facing it; sofa facing a TV unit across a coffee table, armchairs beside the
coffee table facing it; cabinets, wardrobes, shelves, sideboards, counters against walls; tall pieces never
in front of a window or a door; nothing in the door's swing zone; every piece that needs access keeps a
reachable strip in front of it (walking clearance, checked on a 5 cm grid from the door).
Deterministic per seed (numpy Generator), varied across seeds.
"""

from __future__ import annotations

import json

import numpy as np

ROOMS = {   # room type: (width range, depth range); ceiling 2.6-3.0 m for all
    "bedroom": ((3.4, 4.6), (3.6, 5.0)),
    "living room": ((4.0, 5.2), (4.4, 5.8)),
    "kitchen": ((3.2, 4.4), (3.6, 5.0)),
    "dining room": ((3.6, 4.8), (3.8, 5.2)),
    "bathroom": ((2.4, 3.0), (2.8, 3.6)),
}
CEILING = (2.6, 3.0)
DOOR_W, DOOR_H = 0.9, 2.1
DOOR_SWING = 1.0            # m kept free in front of the door
GAP = 0.05                  # m minimum gap between pieces
WALK = 0.6                  # m walking clearance (path width)
ACCESS = 0.45               # m deep strip in front of a piece kept free of other pieces
REACH = 0.8                 # m: a person must be able to stand within this distance of a piece's front
TALL = 1.0                  # pieces taller than this never stand in front of a window
WALL_YAW = {"S": 0, "N": 180, "W": 90, "E": 270}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
STYLES = ["contemporary", "coastal", "rustic", "mid-century modern", "art deco", "mediterranean", "scandinavian",
          "farmhouse", "industrial", "bohemian", "modern", "minimalist", "traditional", "japandi"]


# --------------------------------------------------------------------------- #
# Prompt parsing
# --------------------------------------------------------------------------- #
def parse_prompt(prompt: str) -> tuple[str, str]:
    """'a coastal style bedroom' -> ('bedroom', 'coastal'); unknown room -> living room."""
    p = prompt.lower()
    room = next((r for r in sorted(ROOMS, key=len, reverse=True) if r in p), None)
    if room is None:
        room = "bedroom" if "bed" in p else "kitchen" if "kitchen" in p else "bathroom" if "bath" in p else \
            "dining room" if "dining" in p else "living room"
    style = next((s for s in sorted(STYLES, key=len, reverse=True) if s in p), None)
    if style is None:
        style = p.split(" style")[0].replace("a ", "", 1).replace("an ", "", 1).strip() if " style" in p else "modern"
    return room, style


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def extent(p: dict) -> tuple[float, float, float, float]:
    """Axis-aligned footprint (x0, z0, x1, z1) of a piece."""
    ex, ez = (p["w"], p["d"]) if p["yaw"] in (0, 180) else (p["d"], p["w"])
    return p["cx"] - ex / 2, p["cz"] - ez / 2, p["cx"] + ex / 2, p["cz"] + ez / 2


def front_dir(yaw: float) -> np.ndarray:
    a = np.deg2rad(yaw)
    return np.array([np.sin(a), np.cos(a)])


def _overlap(a, b, gap=GAP) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def front_strip(p: dict, depth: float = ACCESS) -> tuple:
    """Footprint of the strip `depth` deep in front of a piece."""
    x0, z0, x1, z1 = extent(p)
    f = front_dir(p["yaw"])
    if f[1] > 0.5:
        return x0, z1, x1, z1 + depth
    if f[1] < -0.5:
        return x0, z0 - depth, x1, z0
    if f[0] > 0.5:
        return x1, z0, x1 + depth, z1
    return x0 - depth, z0, x0, z1


class Layout:
    def __init__(self, room: str, style: str, seed: int):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        (w0, w1), (d0, d1) = ROOMS[room]
        self.room, self.style, self.seed = room, style, seed
        self.W, self.D = round(float(r.uniform(w0, w1)), 2), round(float(r.uniform(d0, d1)), 2)
        self.H = round(float(r.uniform(*CEILING)), 2)
        self.items, self.openings, self.keepout = [], [], []
        self.door = self._door()

    # walls ---------------------------------------------------------------- #
    def wall_len(self, wall: str) -> float:
        return self.W if wall in ("N", "S") else self.D

    def on_wall(self, wall: str, s: float, w: float, d: float) -> tuple[float, float, int]:
        """Centre (cx, cz) and yaw of a piece w wide, d deep, back against `wall`, centred at s along it
        (s along x for N / S, along z for E / W)."""
        if wall == "S":
            return s, d / 2, 0
        if wall == "N":
            return s, self.D - d / 2, 180
        if wall == "W":
            return d / 2, s, 90
        return self.W - d / 2, s, 270

    def spans(self, wall: str, kinds=("door", "window")) -> list[tuple[float, float]]:
        return [(o["c"] - o["w"] / 2, o["c"] + o["w"] / 2) for o in self.openings if o["wall"] == wall and o["kind"] in kinds]

    # openings ------------------------------------------------------------- #
    def _door(self) -> dict:
        wall = str(self.rng.choice(["S", "E", "W", "N"]))
        L = self.wall_len(wall)
        near_corner = self.rng.random() < 0.6                     # doors are usually near a corner
        c = float(self.rng.uniform(0.25 + DOOR_W / 2, 0.6 + DOOR_W / 2)) if near_corner else \
            float(self.rng.uniform(DOOR_W / 2 + 0.3, L - DOOR_W / 2 - 0.3))
        if near_corner and self.rng.random() < 0.5:
            c = L - c
        o = {"kind": "door", "wall": wall, "c": round(c, 3), "w": DOOR_W, "y0": 0.0, "y1": DOOR_H}
        self.openings.append(o)
        cx, cz, yaw = self.on_wall(wall, c, DOOR_W + 0.2, DOOR_SWING)
        self.keepout.append({"cx": cx, "cz": cz, "w": DOOR_W + 0.2, "d": DOOR_SWING, "yaw": yaw})
        return o

    def add_window(self, wall: str, width: float | None = None) -> dict | None:
        L = self.wall_len(wall)
        w = round(float(width or self.rng.uniform(1.0, 1.5)), 2)
        if L < w + 0.8:
            return None
        sill = round(float(self.rng.uniform(0.85, 0.95)), 2)
        top = min(self.H - 0.35, sill + float(self.rng.uniform(1.2, 1.4)))
        for _ in range(30):
            c = float(self.rng.uniform(w / 2 + 0.4, L - w / 2 - 0.4)) if self.rng.random() < 0.5 else L / 2
            if all(c + w / 2 + 0.3 <= a or c - w / 2 - 0.3 >= b for a, b in self.spans(wall)):
                o = {"kind": "window", "wall": wall, "c": round(c, 3), "w": w, "y0": sill, "y1": round(top, 2)}
                self.openings.append(o)
                return o
        return None

    # placement ------------------------------------------------------------ #
    def free(self, p: dict, ignore=()) -> bool:
        e = extent(p)
        if e[0] < -1e-6 or e[1] < -1e-6 or e[2] > self.W + 1e-6 or e[3] > self.D + 1e-6:
            return False
        for q in self.items:
            if q["label"] in ignore:
                continue
            if q.get("y0", 0) >= p.get("y0", 0) + p["h"] or p.get("y0", 0) >= q.get("y0", 0) + q["h"]:
                continue                                               # stacked (TV on its unit)
            if _overlap(e, extent(q), 0.0 if p.get("run") and q.get("run") else GAP):
                return False
        if any(_overlap(e, extent(k), 0.0) for k in self.keepout):
            return False
        if p["wall"] and p["h"] > TALL:                               # tall pieces never in front of openings
            s0, s1 = (e[0], e[2]) if p["wall"] in ("N", "S") else (e[1], e[3])
            if any(s1 > a - 0.1 and s0 < b + 0.1 for a, b in self.spans(p["wall"])):
                return False
        if p["wall"] and p["h"] > 0.95 and p.get("y0", 0) == 0:       # nothing above sill height under a window
            s0, s1 = (e[0], e[2]) if p["wall"] in ("N", "S") else (e[1], e[3])
            if any(s1 > a and s0 < b for a, b in self.spans(p["wall"], ("window",))):
                return False
        for q in self.items:                                          # keep the access strips of others free
            if q.get("access") and q["label"] not in ignore and p.get("group") != q.get("id") \
                    and _overlap(e, front_strip(q), 0.0):
                return False
        if p.get("access") and any(_overlap(front_strip(p), extent(q), 0.0) for q in self.items
                                   if q.get("group") != p.get("id") and q.get("id") != p.get("group")):
            return False
        return True

    def add(self, cls: str, w: float, d: float, h: float, cx: float, cz: float, yaw: int, wall=None,
            group=None, label=None, y0: float = 0.0, access: bool = True, ignore=(), run: bool = False) -> dict | None:
        p = {"id": len(self.items), "cls": cls, "label": label or cls, "cx": round(float(cx), 3),
             "cz": round(float(cz), 3), "w": round(float(w), 3), "d": round(float(d), 3), "h": round(float(h), 3),
             "y0": float(y0), "yaw": int(yaw) % 360, "wall": wall, "group": group, "access": access}
        if run:
            p["run"] = True                                            # counter modules may touch each other
        if self.free(p, ignore):
            self.items.append(p)
            return p
        return None

    def against(self, cls: str, w: float, d: float, h: float, walls, s=None, tries: int = 60, **kw) -> dict | None:
        """Place a piece with its back against one of `walls` (seeded random position unless s given)."""
        walls = list(walls)
        for k in range(tries):
            wall = walls[k % len(walls)] if s is not None else str(self.rng.choice(walls))
            L = self.wall_len(wall)
            if L < w + 0.02:
                continue
            ss = s if s is not None else float(self.rng.uniform(w / 2 + 0.02, L - w / 2 - 0.02))
            if s is not None and k >= len(walls):                      # requested spot taken: jitter it
                ss = float(np.clip(s + self.rng.normal(0, 0.4), w / 2 + 0.02, L - w / 2 - 0.02))
            cx, cz, yaw = self.on_wall(wall, ss, w, d)
            p = self.add(cls, w, d, h, cx, cz, yaw, wall=wall, **kw)
            if p:
                return p
        return None

    def corner(self, cls: str, size: float, h: float, **kw) -> dict | None:
        """A small piece (plant, lamp) in a free corner, back to one of the two walls."""
        corners = [("S", 0.0), ("S", 1.0), ("N", 0.0), ("N", 1.0)]
        order = self.rng.permutation(len(corners))
        for i in order:
            wall, t = corners[i]
            L = self.wall_len(wall)
            s = size / 2 + 0.05 if t == 0 else L - size / 2 - 0.05
            cx, cz, yaw = self.on_wall(wall, s, size, size)
            p = self.add(cls, size, size, h, cx, cz, yaw, wall=wall, access=False, **kw)
            if p:
                return p
        return None

    def other_walls(self, *exclude) -> list[str]:
        return [w for w in ("N", "S", "E", "W") if w not in exclude]

    # camera --------------------------------------------------------------- #
    def occupancy(self, cell: float = 0.05) -> tuple[np.ndarray, float]:
        nx, nz = int(np.ceil(self.W / cell)), int(np.ceil(self.D / cell))
        occ = np.zeros((nz, nx), bool)
        for p in self.items:
            if p.get("y0", 0) > 1.6:
                continue
            x0, z0, x1, z1 = extent(p)
            occ[max(0, int(z0 / cell)):int(np.ceil(z1 / cell)), max(0, int(x0 / cell)):int(np.ceil(x1 / cell))] = True
        return occ, cell

    def clearance_map(self, cell: float = 0.05) -> tuple[np.ndarray, float]:
        """Distance (m) of each floor cell to the nearest furniture or wall."""
        import cv2
        occ, cell = self.occupancy(cell)
        pad = np.pad(~occ, 1, constant_values=False).astype(np.uint8)
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1] * cell
        return dist, cell

    def pano_camera(self, height: float = 1.5, clear: float = 0.5) -> list[float]:
        """Panorama camera at `height`: among floor points >= `clear` m from every piece of furniture and from the
        walls, the one that maximises min(wall distance, 1.5 m) + 0.3 min(furniture clearance, 1 m). (Pure maximum
        clearance put the camera next to a wall in crowded rooms; a near blank wall then dominated the depth
        condition and the model filled it with invented objects.) Fallback: maximum clearance."""
        import cv2
        occ, cell = self.occupancy()
        free = (~occ).astype(np.uint8)
        fclear = cv2.distanceTransform(np.pad(free, 1, constant_values=1), cv2.DIST_L2, 5)[1:-1, 1:-1] * cell
        zz, xx = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
        X, Z = (xx + 0.5) * cell, (zz + 0.5) * cell
        wdist = np.minimum.reduce([X, self.W - X, Z, self.D - Z])
        ok = (fclear >= clear + cell) & (wdist >= clear)          # + one cell: grid distances are cell-centre based
        if not ok.any():
            dist, _ = self.clearance_map()
            i = np.unravel_index(np.argmax(dist), dist.shape)
        else:
            score = np.minimum(wdist, 1.5) + 0.3 * np.minimum(fclear, 1.0) - 0.01 * np.hypot(X - self.W / 2, Z - self.D / 2)
            i = np.unravel_index(np.argmax(np.where(ok, score, -1e9)), score.shape)
        return [round((i[1] + 0.5) * cell, 3), height, round((i[0] + 0.5) * cell, 3)]

    def reachable(self) -> np.ndarray:
        """Floor cells reachable from the door by a person (path at least WALK wide)."""
        import cv2
        dist, cell = self.clearance_map()
        ok = (dist >= WALK / 2).astype(np.uint8)
        n, lab = cv2.connectedComponents(ok, connectivity=4)
        o = self.door
        cx, cz, _ = self.on_wall(o["wall"], o["c"], DOOR_W, 1.2)          # point 0.6 m inside the door
        i, j = min(int(cz / cell), lab.shape[0] - 1), min(int(cx / cell), lab.shape[1] - 1)
        return lab == lab[i, j] if lab[i, j] > 0 else np.zeros_like(ok, bool)

    def to_dict(self) -> dict:
        return {"version": 6, "room": self.room, "style": self.style, "seed": self.seed, "W": self.W, "D": self.D,
                "H": self.H, "items": self.items, "openings": self.openings, "camera": self.pano_camera()}


# --------------------------------------------------------------------------- #
# Builders per room type
# --------------------------------------------------------------------------- #
def _bedroom(L: Layout):
    r = L.rng
    door = L.door["wall"]
    L.add_window(str(r.choice(L.other_walls(door))))
    if r.random() < 0.4:
        L.add_window(str(r.choice(L.other_walls(door))))
    bw = 1.8 if min(L.W, L.D) > 4.0 and r.random() < 0.5 else 1.6 if min(L.W, L.D) > 3.5 else 1.4
    bed = None
    heads = L.other_walls(door)
    r.shuffle(heads)
    heads.sort(key=lambda w: len(L.spans(w, ("window",))))          # headboard preferably on a solid wall
    for wall in heads:
        Lw = L.wall_len(wall)
        s = Lw / 2 + float(r.uniform(-0.25, 0.25))
        bed = L.against("bed", bw, 2.1, 1.0, [wall], s=s, tries=6)
        if bed:
            break
    if bed:
        for side in (-1, 1):                                            # nightstands on both sides
            ns = 0.5
            off = bw / 2 + 0.07 + ns / 2
            s = (bed["cx"] if bed["wall"] in ("N", "S") else bed["cz"]) + side * off
            cx, cz, yaw = L.on_wall(bed["wall"], s, ns, 0.42)
            L.add("nightstand", ns, 0.42, 0.55, cx, cz, yaw, wall=bed["wall"], group=bed["id"], access=False)
    wd = float(r.choice([1.2, 1.6, 2.0]))
    L.against("wardrobe", wd, 0.6, 2.1, L.other_walls(bed["wall"] if bed else ""), group=None) or \
        L.against("wardrobe", 1.0, 0.6, 2.1, L.other_walls(bed["wall"] if bed else ""))
    if r.random() < 0.55:
        desk = L.against("desk", 1.2, 0.6, 0.76, L.other_walls())
        if desk:
            f = front_dir(desk["yaw"])
            L.add("chair", 0.48, 0.52, 0.95, desk["cx"] + f[0] * (0.3 + 0.32), desk["cz"] + f[1] * (0.3 + 0.32),
                  (desk["yaw"] + 180) % 360, group=desk["id"], access=False)
    elif r.random() < 0.7:
        L.against("armchair", 0.85, 0.85, 0.9, L.other_walls())
    if r.random() < 0.6:
        L.corner("plant", 0.45, 1.1)
    if r.random() < 0.4:
        L.against("shelf", 1.0, 0.35, 1.9, L.other_walls())


def _living(L: Layout):
    r = L.rng
    door = L.door["wall"]
    for w in r.permutation(L.other_walls(door))[: 1 + int(r.random() < 0.6)]:
        L.add_window(str(w))
    sw = 2.2 if min(L.W, L.D) > 4.6 else 2.0
    sofa = None
    walls = L.other_walls(door)
    r.shuffle(walls)
    walls.sort(key=lambda w: -L.wall_len(w))
    for wall in walls:
        s = L.wall_len(wall) / 2 + float(r.uniform(-0.3, 0.3))
        span = L.D if wall in ("N", "S") else L.W                       # distance sofa wall -> TV wall
        if span > 5.0:                                                 # big room: float the sofa 3.2 m from the TV
            off = span - 3.2 - 0.95
            cx, cz, yaw = L.on_wall(wall, s, sw, 0.95)
            f = front_dir(yaw)
            sofa = L.add("sofa", sw, 0.95, 0.85, cx + f[0] * off, cz + f[1] * off, yaw, wall=None)
        else:
            sofa = L.against("sofa", sw, 0.95, 0.85, [wall], s=s, tries=4)
        if sofa:
            break
    if not sofa:
        return
    axis = sofa["cx"] if sofa["yaw"] in (0, 180) else sofa["cz"]
    tvw = OPP[{0: "S", 180: "N", 90: "W", 270: "E"}[sofa["yaw"]]]
    tv = L.against("TV unit", 1.8, 0.45, 0.55, [tvw], s=axis, tries=8, group=sofa["id"])
    if tv:
        L.add("television", 1.2, 0.12, 0.72, tv["cx"], tv["cz"], tv["yaw"], wall=None, group=tv["id"], y0=0.56,
              access=False)
    f = front_dir(sofa["yaw"])
    ct = L.add("coffee table", 1.1, 0.6, 0.42, sofa["cx"] + f[0] * (0.475 + 0.45 + 0.3),
               sofa["cz"] + f[1] * (0.475 + 0.45 + 0.3), sofa["yaw"], group=sofa["id"], access=False)
    if ct:
        side = np.array([f[1], -f[0]])                                  # sofa's right-hand axis in the plan
        for sgn in r.permutation([-1, 1])[: 1 + int(r.random() < 0.5)]:
            c = np.array([ct["cx"], ct["cz"]]) + sgn * side * (0.55 + 0.35 + 0.45) + f * 0.45
            yaw = int(np.rad2deg(np.arctan2(-sgn * side[0], -sgn * side[1]))) % 360
            L.add("armchair", 0.85, 0.85, 0.9, c[0], c[1], yaw, group=sofa["id"], access=False)
    L.against("shelf", 1.0, 0.38, 1.9, L.other_walls(tvw))
    if r.random() < 0.6:
        L.corner("plant", 0.45, 1.2)
    if r.random() < 0.5:
        L.corner("floor lamp", 0.35, 1.6)


def _slot_free(L: Layout, wall: str, s: float, w: float, d: float, h: float) -> bool:
    cx, cz, yaw = L.on_wall(wall, s, w, d)
    return L.free({"cx": cx, "cz": cz, "w": w, "d": d, "h": h, "y0": 0.0, "yaw": yaw, "wall": wall, "access": True,
                   "run": True})


def _counter_run(L: Layout, wall: str, need_fridge: bool, need_sink: bool, need_stove: bool) -> None:
    """Counter run along `wall` in 0.6 m modules: fridge at one end (never under the window), sink under
    the window (or near the middle), stove at least two modules from the sink, counters elsewhere."""
    r = L.rng
    Lw = L.wall_len(wall)
    n = int((Lw - 0.04) // 0.6)
    slots = [0.02 + 0.3 + k * 0.6 for k in range(n)]
    ok = [_slot_free(L, wall, s, 0.6, 0.62, 0.92) for s in slots]
    use = {}
    if need_fridge:
        ends = [0, n - 1] if r.random() < 0.5 else [n - 1, 0]
        for k in ends:
            if 0 <= k < n and ok[k] and _slot_free(L, wall, slots[k], 0.6, 0.68, 1.85):
                use[k] = "fridge"
                break
    free_k = [k for k in range(n) if ok[k] and k not in use]
    win = L.spans(wall, ("window",))
    target = (win[0][0] + win[0][1]) / 2 if win else Lw / 2
    if need_sink and free_k:
        k = min(free_k, key=lambda k: abs(slots[k] - target))
        use[k] = "sink"
    sink_k = next((k for k, v in use.items() if v == "sink"), None)
    if need_stove:
        cand = [k for k in free_k if k not in use and (sink_k is None or abs(k - sink_k) >= 2)] or                [k for k in free_k if k not in use]
        if cand:
            use[int(r.choice(cand))] = "stove"
    for k in free_k:
        use.setdefault(k, "counter")
    for k in sorted(use):
        m = use[k]
        if m == "fridge":
            L.against("fridge", 0.6, 0.68, 1.85, [wall], s=slots[k], tries=1, access=True, run=True)
        else:
            L.against(m, 0.6, 0.62, 0.92, [wall], s=slots[k], tries=1, access=m != "counter", run=True)


def _kitchen(L: Layout):
    r = L.rng
    door = L.door["wall"]
    walls = L.other_walls(door)
    r.shuffle(walls)
    walls.sort(key=lambda w: -L.wall_len(w))
    w1 = walls[0]
    L.add_window(w1, width=float(r.uniform(1.0, 1.3)))
    _counter_run(L, w1, True, True, True)
    has = lambda c: any(i["cls"] == c for i in L.items)                # noqa: E731
    w2 = [w for w in walls[1:] if w != OPP[w1]]
    if w2 and (r.random() < 0.6 or not (has("fridge") and has("stove") and has("sink"))):
        _counter_run(L, w2[0], not has("fridge"), not has("sink"), not has("stove"))   # L-shaped kitchen
    if min(L.W, L.D) >= 3.6:
        yaw = 0 if L.W >= L.D else 90
        L.add("island", 1.6, 0.9, 0.92, L.W / 2, L.D / 2, yaw, access=False)
    elif r.random() < 0.7:
        t = L.against("dining table", 1.0, 0.7, 0.76, [OPP[w1]], access=False)
        if t:
            f = front_dir(t["yaw"])
            side = np.array([f[1], -f[0]])
            for sgn in (-1, 1):
                c = np.array([t["cx"], t["cz"]]) + sgn * side * (0.5 + 0.06 + 0.26)
                yaw = int(np.rad2deg(np.arctan2(-sgn * side[0], -sgn * side[1]))) % 360
                L.add("chair", 0.46, 0.52, 0.95, c[0], c[1], yaw, group=t["id"], access=False)
    if r.random() < 0.5:
        L.corner("plant", 0.4, 1.0)


def _table_chairs(L: Layout, t: dict, n_side: int, end_chairs: bool) -> None:
    """Chairs on both long sides of table t (and at both ends), facing it."""
    tl, tdp = t["w"], t["d"]
    f = front_dir(t["yaw"])                                             # table's "front" = one long side
    along = np.array([f[1], -f[0]])
    cw, cd = 0.46, 0.52
    for sgn in (-1, 1):
        for k in range(n_side):
            u = (k - (n_side - 1) / 2) * (tl / n_side)
            c = np.array([t["cx"], t["cz"]]) + along * u + sgn * f * (tdp / 2 + GAP + 0.015 + cd / 2)
            cyaw = int(np.rad2deg(np.arctan2(-sgn * f[0], -sgn * f[1]))) % 360
            L.add("chair", cw, cd, 0.95, c[0], c[1], cyaw, group=t["id"], access=False)
    if end_chairs:
        for sgn in (-1, 1):
            c = np.array([t["cx"], t["cz"]]) + sgn * along * (tl / 2 + GAP + 0.015 + cd / 2)
            cyaw = int(np.rad2deg(np.arctan2(-sgn * along[0], -sgn * along[1]))) % 360
            L.add("chair", cw, cd, 0.95, c[0], c[1], cyaw, group=t["id"], access=False)


def _dining(L: Layout):
    r = L.rng
    door = L.door["wall"]
    for w in r.permutation(L.other_walls(door))[: 1 + int(r.random() < 0.5)]:
        L.add_window(str(w))
    long_x = L.W >= L.D
    tl = float(np.clip(min(max(L.W, L.D) - 2.0, 2.0), 1.4, 2.0))
    tdp = 0.95 if tl >= 1.8 else 0.9
    yaw = 0 if long_x else 90                                           # long side along the longer room side
    n_side = 3 if tl >= 1.9 else 2
    end_chairs = r.random() < 0.6
    cands = [(L.W / 2 + dx, L.D / 2 + dz) for dx in (0, -0.3, 0.3, -0.5, 0.5) for dz in (0, -0.3, 0.3, -0.5, 0.5)]
    base = list(L.items)
    best = None
    for cx, cz in cands:                                               # most chairs, then closest to the centre
        L.items = list(base)
        t = L.add("dining table", tl, tdp, 0.76, cx, cz, yaw, access=False)
        if not t:
            continue
        _table_chairs(L, t, n_side, end_chairs)
        n = sum(i["cls"] == "chair" for i in L.items)
        key = (n, -np.hypot(cx - L.W / 2, cz - L.D / 2))
        if best is None or key > best[0]:
            best = (key, list(L.items))
    L.items = best[1] if best else base
    if not best:
        return
    walls = L.other_walls(door)
    walls.sort(key=lambda w: -L.wall_len(w))
    L.against("sideboard", 1.6, 0.48, 0.85, walls[:2] + walls[2:], access=True)
    if r.random() < 0.5:
        L.against("shelf", 1.0, 0.38, 1.9, L.other_walls())
    if r.random() < 0.7:
        L.corner("plant", 0.45, 1.2)


def _bathroom(L: Layout):
    r = L.rng
    door = L.door["wall"]
    walls = L.other_walls(door)
    r.shuffle(walls)
    short = sorted(["N", "S", "E", "W"], key=L.wall_len)[:2]
    tub_wall = next((w for w in short if w != door), walls[0])
    Lw = L.wall_len(tub_wall)
    tl = float(min(1.7, Lw - 0.04))
    L.against("bathtub", tl, 0.75, 0.58, [tub_wall], s=tl / 2 + 0.02 if r.random() < 0.5 else Lw - tl / 2 - 0.02,
              tries=1, access=True)
    if not L.items:
        L.against("bathtub", tl, 0.75, 0.58, [tub_wall], s=Lw / 2, tries=1)
    rest = [w for w in L.other_walls(tub_wall)]
    L.add_window(str(r.choice([w for w in rest if w != door] or rest)), width=float(r.uniform(0.7, 1.0)))
    L.against("vanity", 0.9, 0.5, 0.86, rest)
    L.against("toilet", 0.42, 0.68, 0.78, rest)
    if r.random() < 0.5:
        L.corner("plant", 0.35, 0.9)


BUILDERS = {"bedroom": _bedroom, "living room": _living, "kitchen": _kitchen, "dining room": _dining,
            "bathroom": _bathroom}
MAIN = {"bedroom": "bed", "living room": "sofa", "kitchen": "stove", "dining room": "dining table",
        "bathroom": "bathtub"}


def generate(prompt: str, seed: int) -> dict:
    """Seeded v6 layout of a prompt (deterministic). If a builder misses its main piece (rare tight room),
    the room is redrawn with a derived seed (at most 20 times)."""
    room, style = parse_prompt(prompt)
    best = None
    for k in range(20):
        L = Layout(room, style, seed * 1000 + k)
        BUILDERS[room](L)
        d = L.to_dict()
        score = 10 * violations(d)["total"] + missing(d)
        if best is None or score < best[0]:
            best = (score, d, k)
        if score == 0:
            break
    score, d, k = best
    d["seed"], d["prompt"], d["attempt"] = seed, prompt, k
    return d


REQUIRED = {"bedroom": {"bed": 1, "nightstand": 2, "wardrobe": 1}, "living room": {"sofa": 1, "TV unit": 1,
            "coffee table": 1}, "kitchen": {"stove": 1, "sink": 1, "fridge": 1}, "dining room": {"dining table": 1,
            "chair": 4}, "bathroom": {"bathtub": 1, "toilet": 1, "vanity": 1}}


def missing(lay: dict) -> int:
    """Number of required pieces of the room type that the builder could not place."""
    n = 0
    for cls, k in REQUIRED[lay["room"]].items():
        n += max(0, k - sum(i["cls"] == cls for i in lay["items"]))
    return n


# --------------------------------------------------------------------------- #
# Checks (used by the unit tests and the milestone metrics)
# --------------------------------------------------------------------------- #
def violations(lay: dict) -> dict:
    """Rule violations of a layout: overlaps, outside the room, wall-aligned pieces off their wall or
    facing the wrong way, chairs not facing their table, pieces in the door swing, tall pieces in front
    of openings, unreachable access strips."""
    W, D = lay["W"], lay["D"]
    items = lay["items"]
    v = {"overlap": 0, "outside": 0, "wall_align": 0, "facing": 0, "door": 0, "window": 0, "access": 0}
    for i, p in enumerate(items):
        e = extent(p)
        if e[0] < -1e-3 or e[1] < -1e-3 or e[2] > W + 1e-3 or e[3] > D + 1e-3:
            v["outside"] += 1
        for q in items[i + 1:]:
            stacked = q["y0"] >= p["y0"] + p["h"] or p["y0"] >= q["y0"] + q["h"]
            if not stacked and _overlap(e, extent(q), 0.0):
                v["overlap"] += 1
        if p["yaw"] % 90:
            v["wall_align"] += 1
        if p["wall"]:
            want = WALL_YAW[p["wall"]]
            gap = {"S": e[1], "N": D - e[3], "W": e[0], "E": W - e[2]}[p["wall"]]
            if p["yaw"] != want or gap > 0.02:
                v["wall_align"] += 1
        if p["cls"] == "chair" and p["group"] is not None:
            t = items[p["group"]]
            to = np.array([t["cx"] - p["cx"], t["cz"] - p["cz"]])
            if np.dot(front_dir(p["yaw"]), to) <= 0:
                v["facing"] += 1
    L = Layout.__new__(Layout)
    L.W, L.D, L.items, L.openings = W, D, items, lay["openings"]
    L.door = next(o for o in lay["openings"] if o["kind"] == "door")
    cx, cz, yaw = Layout.on_wall(L, L.door["wall"], L.door["c"], DOOR_W + 0.2, DOOR_SWING)
    swing = extent({"cx": cx, "cz": cz, "w": DOOR_W, "d": DOOR_SWING, "yaw": yaw})
    for p in items:
        if _overlap(extent(p), swing, 0.0):
            v["door"] += 1
        if p["wall"] and p["h"] > TALL:
            e = extent(p)
            s0, s1 = (e[0], e[2]) if p["wall"] in ("N", "S") else (e[1], e[3])
            if any(s1 > o["c"] - o["w"] / 2 and s0 < o["c"] + o["w"] / 2 for o in lay["openings"] if o["wall"] == p["wall"]):
                v["window"] += 1
    reach = Layout.reachable(L)
    cell = 0.05
    for p in items:
        if not p.get("access"):
            continue
        x0, z0, x1, z1 = front_strip(p, REACH)
        sub = reach[max(0, int(z0 / cell)):int(np.ceil(z1 / cell)), max(0, int(x0 / cell)):int(np.ceil(x1 / cell))]
        if sub.size == 0 or not sub.any():
            v["access"] += 1
    v["total"] = sum(v.values())
    return v


def save(lay: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lay, f, indent=1)
