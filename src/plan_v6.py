"""Floor plan of a v6 layout (PLAN_V6 milestone 4), drawn directly from the layout (no estimation).

    walls        thick dark outline (0.12 m), door as a gap + swing arc, windows as a thin double line
    furniture    one symbol per class, drawn in the piece's own frame (front = yaw direction):
                 bed (mattress + pillows + folded sheet), sofa / armchair (back + arms + seat), chair
                 (seat + backrest), tables (top, round if the model is round), counters / stove (burners) /
                 sink (basin), wardrobe (doors), shelf, plant (circle), toilet, bathtub, TV unit + screen
    labels       name inside the piece when it fits, else outside with a leader line; candidate positions
                 are tried in order and a label is only placed where it overlaps no other label, piece
                 symbol of another piece or wall (collision avoidance)
    cameras      every shot as a cone of the same style (filled wedge, labelled H, H2, A-D) at its true
                 position, yaw and field of view; the panorama position as a small circle "360"
    dimensions   overall width and depth only (outside the walls), scale bar 1 m or 2 m with 0.5 m ticks,
                 about a third of the room's width
Plan orientation: x right, z (plan y) up.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Polygon, Rectangle, Wedge  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from . import layout_v6 as LV  # noqa: E402

WALL_T = 0.12
INK = "#2b2b2b"
FILL = "#f4efe6"
FURN = "#ffffff"
CAM = "#d1495b"
LABEL_NAMES = {"TV unit": "TV unit", "television": None, "coffee table": "coffee table", "dining table": "dining table"}


def _frame(p: dict):
    """Local -> plan transform of a piece: local u across the front (right-hand), v from back (0) to
    front (d); returns a function (u, v) -> (x, z)."""
    f = LV.front_dir(p["yaw"])
    r = np.array([f[1], -f[0]])                     # right-hand axis when looking along the front
    back = np.array([p["cx"], p["cz"]]) - f * p["d"] / 2

    def T(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        P = back[None, :] + np.outer(u, r) + np.outer(v, f)
        return P
    return T


def _poly(ax, T, pts, **kw):
    pts = np.asarray(pts, float)
    ax.add_patch(Polygon(T(pts[:, 0], pts[:, 1]), closed=True, **kw))


def _rect(ax, T, u0, v0, u1, v1, **kw):
    _poly(ax, T, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], **kw)


def draw_piece(ax, p: dict, model: str = "") -> None:
    T = _frame(p)
    w, d = p["w"], p["d"]
    a, b = -w / 2, w / 2
    lw = 1.0
    base = dict(fc=FURN, ec=INK, lw=lw, zorder=3)
    line = dict(fc="none", ec=INK, lw=0.7, zorder=4)
    c = p["cls"]
    if c == "bed":
        _rect(ax, T, a, 0, b, d, **base)
        _rect(ax, T, a, 0, b, 0.08, fc="#ddd5c8", ec=INK, lw=lw, zorder=4)              # headboard
        n = 2 if w >= 1.3 else 1
        pw = (w - 0.2 - 0.1 * (n - 1)) / n
        for k in range(n):
            u0 = a + 0.1 + k * (pw + 0.1)
            _rect(ax, T, u0, 0.15, u0 + pw, 0.5, **line)                                 # pillows
        _poly(ax, T, [(a, 0.75), (b, 0.75)], fill=False, ec=INK, lw=0.7, zorder=4)
        _rect(ax, T, a, 0.75, b, 0.95, fc="#ece6db", ec=INK, lw=0.7, zorder=4)           # folded sheet
    elif c in ("sofa", "armchair"):
        arm = 0.16 if c == "sofa" else 0.15
        _rect(ax, T, a, 0, b, d, **base)
        _rect(ax, T, a, 0, b, 0.22, fc="#e9e2d6", ec=INK, lw=lw, zorder=4)               # back
        _rect(ax, T, a, 0.22, a + arm, d, fc="#e9e2d6", ec=INK, lw=lw, zorder=4)
        _rect(ax, T, b - arm, 0.22, b, d, fc="#e9e2d6", ec=INK, lw=lw, zorder=4)
        if c == "sofa":
            n = 3 if w > 1.9 else 2
            uu = np.linspace(a + arm, b - arm, n + 1)
            for k in range(1, n):
                _poly(ax, T, [(uu[k], 0.22), (uu[k], d)], fill=False, ec=INK, lw=0.6, zorder=4)
    elif c == "chair":
        _rect(ax, T, a + 0.02, 0.06, b - 0.02, d, **base)
        _rect(ax, T, a, 0, b, 0.08, fc="#e9e2d6", ec=INK, lw=lw, zorder=4)               # backrest
    elif c in ("dining table", "coffee table", "desk", "island", "nightstand"):
        if "round" in model:
            r = min(w, d) / 2
            ax.add_patch(Circle((p["cx"], p["cz"]), r, **base))
        else:
            _rect(ax, T, a, 0, b, d, **base)
            if c == "nightstand":
                _rect(ax, T, a + 0.05, d - 0.08, b - 0.05, d - 0.03, **line)
    elif c == "wardrobe":
        _rect(ax, T, a, 0, b, d, **base)
        n = max(2, int(round(w / 0.5)))
        for k in range(1, n):
            u = a + k * w / n
            _poly(ax, T, [(u, 0), (u, d)], fill=False, ec=INK, lw=0.6, zorder=4)
        _poly(ax, T, [(a, d - 0.04), (b, d - 0.04)], fill=False, ec=INK, lw=0.6, zorder=4)
    elif c in ("shelf",):
        _rect(ax, T, a, 0, b, d, **base)
        for k in range(1, 4):
            u = a + k * w / 4
            _poly(ax, T, [(u, 0.03), (u, d - 0.03)], fill=False, ec=INK, lw=0.5, zorder=4)
    elif c in ("sideboard", "TV unit", "counter", "fridge"):
        _rect(ax, T, a, 0, b, d, **base)
        _poly(ax, T, [(a, d - 0.05), (b, d - 0.05)], fill=False, ec=INK, lw=0.6, zorder=4)
        if c == "fridge":
            _poly(ax, T, [(a, 0), (b, d - 0.05)], fill=False, ec=INK, lw=0.5, zorder=4)
            _poly(ax, T, [(b, 0), (a, d - 0.05)], fill=False, ec=INK, lw=0.5, zorder=4)
    elif c == "television":
        _rect(ax, T, a, 0.02, b, 0.08, fc=INK, ec=INK, lw=lw, zorder=5)
    elif c == "stove":
        _rect(ax, T, a, 0, b, d, **base)
        for du in (-w / 4, w / 4):
            for dv in (d * 0.33, d * 0.7):
                ax.add_patch(Circle(T([du], [dv])[0], 0.09, fc="none", ec=INK, lw=0.7, zorder=4))
    elif c in ("sink", "vanity"):
        _rect(ax, T, a, 0, b, d, **base)
        ax.add_patch(Circle(T([0], [d * 0.55])[0], min(w, d) * 0.3, fc="#eef3f6", ec=INK, lw=0.7, zorder=4))
    elif c == "bathtub":
        _rect(ax, T, a, 0, b, d, **base)
        _rect(ax, T, a + 0.08, 0.08, b - 0.08, d - 0.08, fc="#eef3f6", ec=INK, lw=0.7, zorder=4)
    elif c == "toilet":
        _rect(ax, T, a + 0.02, 0, b - 0.02, 0.2, **base)
        from matplotlib.patches import Ellipse
        ctr = T([0], [0.2 + (d - 0.2) / 2])[0]
        ang = np.rad2deg(np.arctan2(*LV.front_dir(p["yaw"])[::-1]))
        ax.add_patch(Ellipse(ctr, d - 0.2, w * 0.85, angle=ang, fc=FURN, ec=INK, lw=lw, zorder=3))
    elif c == "plant":
        ax.add_patch(Circle((p["cx"], p["cz"]), min(w, d) / 2 * 0.9, fc="#dfe8d6", ec="#4f6d3a", lw=1, zorder=3))
        for k in range(6):
            t = k * np.pi / 3
            ax.plot([p["cx"], p["cx"] + 0.16 * np.cos(t)], [p["cz"], p["cz"] + 0.16 * np.sin(t)], c="#4f6d3a",
                    lw=0.8, zorder=4)
    elif c == "floor lamp":
        ax.add_patch(Circle((p["cx"], p["cz"]), min(w, d) / 2 * 0.8, fc="#fff6d8", ec=INK, lw=0.8, zorder=3))
        ax.plot(*np.array([[p["cx"] - 0.1, p["cz"] - 0.1], [p["cx"] + 0.1, p["cz"] + 0.1]]).T, c=INK, lw=0.6, zorder=4)
        ax.plot(*np.array([[p["cx"] - 0.1, p["cz"] + 0.1], [p["cx"] + 0.1, p["cz"] - 0.1]]).T, c=INK, lw=0.6, zorder=4)
    else:
        _rect(ax, T, a, 0, b, d, **base)


def _walls(ax, lay: dict) -> None:
    W, D = lay["W"], lay["D"]
    t = WALL_T
    segs = {"S": ((0, 0), (W, 0)), "N": ((0, D), (W, D)), "W": ((0, 0), (0, D)), "E": ((W, 0), (W, D))}
    for wall, (p0, p1) in segs.items():
        L = W if wall in ("N", "S") else D
        cuts = sorted((o["c"] - o["w"] / 2, o["c"] + o["w"] / 2, o) for o in lay["openings"] if o["wall"] == wall)
        s = -t
        pieces = []
        for c0, c1, o in cuts:
            pieces.append((s, c0))
            s = c1
        pieces.append((s, L + t))
        for s0, s1 in pieces:
            if wall in ("N", "S"):
                y = 0 if wall == "S" else D
                ax.add_patch(Rectangle((s0, y - (t if wall == "S" else 0)), s1 - s0, t, fc=INK, ec=INK, zorder=6))
            else:
                x = 0 if wall == "W" else W
                ax.add_patch(Rectangle((x - (t if wall == "W" else 0), s0), t, s1 - s0, fc=INK, ec=INK, zorder=6))
        for c0, c1, o in cuts:
            inward = {"S": (0, 1), "N": (0, -1), "W": (1, 0), "E": (-1, 0)}[wall]
            if wall in ("N", "S"):
                y = 0 if wall == "S" else D
                ys = y - t if wall == "S" else y
                if o["kind"] == "window":
                    ax.add_patch(Rectangle((c0, ys), c1 - c0, t, fc="#ffffff", ec=INK, lw=0.8, zorder=6))
                    ax.plot([c0, c1], [ys + t / 2] * 2, c=INK, lw=0.8, zorder=7)
                else:
                    hinge = (c0, y)
                    ax.plot([c0, c0], [y, y + inward[1] * (c1 - c0)], c=INK, lw=1.2, zorder=6)
                    a0 = 0 if inward[1] > 0 else 270
                    ax.add_patch(Arc(hinge, 2 * (c1 - c0), 2 * (c1 - c0), theta1=a0, theta2=a0 + 90, ec=INK, lw=0.7,
                                     ls="--", zorder=6))
            else:
                x = 0 if wall == "W" else W
                xs = x - t if wall == "W" else x
                if o["kind"] == "window":
                    ax.add_patch(Rectangle((xs, c0), t, c1 - c0, fc="#ffffff", ec=INK, lw=0.8, zorder=6))
                    ax.plot([xs + t / 2] * 2, [c0, c1], c=INK, lw=0.8, zorder=7)
                else:
                    hinge = (x, c0)
                    ax.plot([x, x + inward[0] * (c1 - c0)], [c0, c0], c=INK, lw=1.2, zorder=6)
                    a0 = 0 if inward[0] > 0 else 90
                    ax.add_patch(Arc(hinge, 2 * (c1 - c0), 2 * (c1 - c0), theta1=a0, theta2=a0 + 90, ec=INK, lw=0.7,
                                     ls="--", zorder=6))


def _dims(ax, W: float, D: float) -> None:
    off = 0.45
    kw = dict(arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
    ax.annotate("", (0, -off), (W, -off), **kw)
    ax.text(W / 2, -off - 0.05, f"{W:.2f} m", ha="center", va="top", fontsize=9, color=INK)
    ax.annotate("", (-off, 0), (-off, D), **kw)
    ax.text(-off - 0.06, D / 2, f"{D:.2f} m", ha="right", va="center", rotation=90, fontsize=9, color=INK)
    for x in (0, W):
        ax.plot([x, x], [-WALL_T - 0.05, -off - 0.08], c=INK, lw=0.5)
    for z in (0, D):
        ax.plot([-WALL_T - 0.05, -off - 0.08], [z, z], c=INK, lw=0.5)


def _scale_bar(ax, W: float, D: float) -> None:
    L = 2.0 if W >= 3.5 else 1.0                              # about half the room width
    x0, y = W - L, -0.75
    for k in range(int(L / 0.5)):
        ax.add_patch(Rectangle((x0 + k * 0.5, y), 0.5, 0.08, fc=INK if k % 2 == 0 else "white", ec=INK, lw=0.7))
    ax.text(x0, y - 0.05, "0", ha="center", va="top", fontsize=8)
    ax.text(x0 + L, y - 0.05, f"{L:g} m", ha="center", va="top", fontsize=8)


def _cones(ax, shots: list, pano=None) -> list:
    arts = []
    for s in shots:
        x, z = s["pos"][0], s["pos"][2]
        hf = s["hfov"]
        a = 90 - s["yaw"]                                   # plan angle (deg from +x towards +z)
        R = 0.9
        arts.append(ax.add_patch(Wedge((x, z), R, a - hf / 2, a + hf / 2, fc=CAM, alpha=0.18, ec=CAM, lw=1.0,
                                       zorder=8)))
        ax.add_patch(Circle((x, z), 0.09, fc=CAM, ec="white", lw=0.8, zorder=9))
        ax.text(x, z, s["name"], ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=10)
    if pano is not None:
        ax.add_patch(Circle((pano[0], pano[2]), 0.12, fc="white", ec=CAM, lw=1.2, ls="--", zorder=9))
        ax.text(pano[0], pano[2] - 0.2, "360", ha="center", va="top", fontsize=6.5, color=CAM, zorder=10)
    return arts


def _labels(ax, fig, lay: dict, models: dict, avoid_pts: list) -> None:
    """Place labels without overlaps (inside the piece when it fits, else around it with a leader)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    to_disp = ax.transData.transform
    furn_boxes = []
    for p in lay["items"]:
        x0, z0, x1, z1 = LV.extent(p)
        (a, b), (c, d) = to_disp([(x0, z0), (x1, z1)])
        furn_boxes.append((p["id"], Bbox([[a, b], [c, d]])))
    for (x, z) in avoid_pts:
        (a, b) = to_disp([(x, z)])[0]
        furn_boxes.append((-1, Bbox([[a - 14, b - 14], [a + 14, b + 14]])))
    order = sorted(lay["items"], key=lambda p: -p["w"] * p["d"])
    for p in order:
        name = LABEL_NAMES.get(p["cls"], p["cls"])
        if not name:
            continue
        x0, z0, x1, z1 = LV.extent(p)
        cx, cz = p["cx"], p["cz"]
        cands = [((cx, cz), None)]
        for r in (0.35, 0.6, 0.9):
            for ang in range(0, 360, 45):
                t = np.deg2rad(ang)
                ex, ez = (x1 - x0) / 2 + r, (z1 - z0) / 2 + r
                cands.append(((cx + ex * np.cos(t), cz + ez * np.sin(t)), (cx, cz)))
        for pos, leader in cands:
            if not (-0.2 < pos[0] < lay["W"] + 0.2 and -0.2 < pos[1] < lay["D"] + 0.2):
                continue
            txt = ax.text(*pos, name, ha="center", va="center", fontsize=7, color=INK, zorder=12,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            bb = txt.get_window_extent(renderer).expanded(1.05, 1.15)
            inside_own = leader is None
            if inside_own:
                (a, b), (c, d) = to_disp([(x0, z0), (x1, z1)])
                ok = bb.x0 >= a and bb.x1 <= c and bb.y0 >= b and bb.y1 <= d
            else:
                ok = True
            ok = ok and not any(bb.overlaps(q) for q in placed)
            ok = ok and not any(bb.overlaps(q) for pid, q in furn_boxes if pid != p["id"] and (leader is not None or pid == -1))
            if ok and leader is not None:
                ok = not any(bb.overlaps(q) for pid, q in furn_boxes if pid == p["id"])
            if ok:
                placed.append(bb)
                if leader is not None:
                    ax.plot([pos[0], cx], [pos[1], cz], c="#888888", lw=0.5, zorder=11)
                break
            txt.remove()


def draw(lay: dict, path=None, shots: list | None = None, models: dict | None = None, title: str | None = None,
         dpi: int = 150) -> np.ndarray:
    """Floor plan image (RGB uint8) of a v6 layout; `shots` = [{"name", "pos", "yaw", "hfov"}]."""
    W, D = lay["W"], lay["D"]
    models = models or {}
    fig_w = 7.0
    fig_h = fig_w * (D + 1.6) / (W + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, W + 0.6)
    ax.set_ylim(-1.05, D + 0.55)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, D, fc=FILL, ec="none", zorder=0))
    for p in sorted(lay["items"], key=lambda p: p["y0"]):
        draw_piece(ax, p, models.get(p["id"], ""))
    _walls(ax, lay)
    _dims(ax, W, D)
    _scale_bar(ax, W, D)
    shots = shots or []
    _cones(ax, shots, lay.get("camera"))
    _labels(ax, fig, lay, models, [(s["pos"][0], s["pos"][2]) for s in shots] + [(lay["camera"][0], lay["camera"][2])])
    ax.set_title(title or f"{lay.get('prompt', lay['room'])} - {W:.2f} x {D:.2f} m, ceiling {lay['H']:.2f} m "
                 f"({W * D:.1f} m²)", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if path:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return img
