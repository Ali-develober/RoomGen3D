"""v6 dollhouse and top-down view: the proxy scene, textured from the panorama (PLAN_V6 milestone 4).

Shell (floor and walls; the ceiling is left out so the room can be seen from above)
    each plane is rasterised at PX_PER_M texels per metre; a texel is SEEN if the ray from the panorama
    centre C0 to its 3D point hits the proxy at that point (z-buffer test, furniture in front -> unseen);
    seen texels take the panorama colour. Unseen texels:
      floor  image quilting (Efros & Freeman 2001, overlap-SSD patch choice + feathered overlaps) from
             fully seen floor patches of the same room, so the floor pattern continues under furniture
      walls  normalised convolution (Gaussian, sigma 0.6 m) of the seen texels of the same wall: the wall's
             own colour with its low-frequency shading, no copied decor
    then a 4-texel feather between seen and filled texels.
    Doors and windows are the layout's real openings (frames, door leaf and curtains are kept as geometry).
Furniture
    the placed asset meshes (subdivided to <= 6 cm edges); each face centre seen from C0 (facing it and
    not occluded) takes the panorama colour; hidden faces take the instance's dominant colour (median of
    its seen faces, else of its panorama pixels) modulated by the asset's own texture luminance, so the
    model's material detail stays.
Top-down
    orthographic ray casting of the same textured scene straight down, >= 1024 px on the long side,
    instance outlines, the same orientation as the floor plan (x right, z up).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import raycast_v6 as R
from . import scene_v6 as S
from . import views_v6 as V

PX_PER_M = 100


ADE_WALL, ADE_FLOOR, ADE_CEIL, ADE_WINDOW, ADE_DOOR, ADE_CURTAIN, ADE_RUG = 0, 3, 5, 8, 14, 18, 28


def _seen_colors(rd, pano, c0, P, seg=None, allowed=None, forbidden=None):
    """Panorama colour and SEEN flag of 3D points P (N, 3). With an ADE20K class map `seg` of the panorama,
    a point only counts as seen if the panorama shows an allowed class there (class consistency: the
    generated image need not match the proxy exactly)."""
    v = P - np.asarray(c0)[None]
    dist = np.linalg.norm(v, axis=1)
    u = v / np.maximum(dist[:, None], 1e-6)
    t = rd.tracer.cast(np.asarray(c0, float), u, colors=False)["t"]
    seen = np.isfinite(t) & (np.abs(t - dist) < 0.015 * dist + 0.02)
    col = V.sample_equirect(pano, u[None])[0]
    if seg is not None:
        cls = V.sample_equirect(seg.astype(np.float32)[..., None], u[None], cv2.INTER_NEAREST)
        cls = np.asarray(cls).reshape(-1).astype(int)
        if allowed is not None:
            seen &= np.isin(cls, allowed)
        if forbidden is not None:
            seen &= ~np.isin(cls, forbidden)
    return col, seen


def plane_grid(lay: dict, kind: str, wall: str | None = None, px: int = PX_PER_M):
    """Texel centres of a plane: (P (h, w, 3), (h, w)); floor: rows = z (top = z max), cols = x;
    walls: rows = y (top = ceiling), cols = along the wall."""
    if kind == "floor":
        w, h = int(round(lay["W"] * px)), int(round(lay["D"] * px))
        X, Z = np.meshgrid((np.arange(w) + 0.5) / px, lay["D"] - (np.arange(h) + 0.5) / px)
        P = np.stack([X, np.full_like(X, 0.002), Z], -1)
        return P, (h, w)
    o, a, n, L = S.wall_frame(lay, wall)
    w, h = int(round(L * px)), int(round(lay["H"] * px))
    s, y = np.meshgrid((np.arange(w) + 0.5) / px, lay["H"] - (np.arange(h) + 0.5) / px)
    P = o[None, None] + a[None, None] * s[..., None] + np.array([0, 1.0, 0])[None, None] * y[..., None] + n * 0.002
    return P, (h, w)


def quilt(tex: np.ndarray, seen: np.ndarray, patch: int = 48, overlap: int = 12, n_cand: int = 300,
          seed: int = 0) -> np.ndarray:
    """Fill the unseen texels of `tex` by image quilting from fully seen patches of the same texture."""
    rng = np.random.default_rng(seed)
    h, w = seen.shape
    ok = cv2.erode(seen.astype(np.uint8), np.ones((patch, patch), np.uint8), anchor=(0, 0)) > 0
    ys, xs = np.nonzero(ok[: h - patch + 1, : w - patch + 1]) if h > patch and w > patch else ([], [])
    if len(ys) < 5:
        return None
    idx = rng.choice(len(ys), min(n_cand, len(ys)), replace=False)
    cands = np.stack([tex[ys[i]:ys[i] + patch, xs[i]:xs[i] + patch] for i in idx]).astype(np.float32)
    out = tex.astype(np.float32).copy()
    filled = seen.copy()
    step = patch - overlap
    ramp = np.ones((patch, patch), np.float32)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(h, y0 + patch), min(w, x0 + patch)
            if seen[y0:y1, x0:x1].all():
                continue
            ph, pw = y1 - y0, x1 - x0
            cur, have = out[y0:y1, x0:x1], filled[y0:y1, x0:x1]
            if have.any():
                d = ((cands[:, :ph, :pw] - cur[None]) ** 2).sum(-1)
                cost = (d * have[None]).sum((1, 2)) / max(have.sum(), 1)
                pick = int(rng.choice(np.argsort(cost)[:5]))
            else:
                pick = int(rng.integers(len(cands)))
            P = cands[pick, :ph, :pw]
            # feathered blend over the already filled part of the overlap
            wgt = ramp[:ph, :pw].copy()
            if y0 > 0:
                k = min(overlap, ph)
                wgt[:k] *= np.linspace(0, 1, overlap)[:k, None]
            if x0 > 0:
                k = min(overlap, pw)
                wgt[:, :k] *= np.linspace(0, 1, overlap)[None, :k]
            wgt = np.where(have, wgt, 1.0)
            blended = cur * (1 - wgt[..., None]) + P * wgt[..., None]
            out[y0:y1, x0:x1] = np.where(seen[y0:y1, x0:x1, None], cur, blended)
            filled[y0:y1, x0:x1] = True
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_fill(tex: np.ndarray, seen: np.ndarray, sigma_px: float) -> np.ndarray:
    """Normalised convolution of the seen texels (low-frequency colour + shading)."""
    small = 4
    t = cv2.resize(tex.astype(np.float32), None, fx=1 / small, fy=1 / small, interpolation=cv2.INTER_AREA)
    m = cv2.resize(seen.astype(np.float32), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
    s = max(1.0, sigma_px / small)
    num = cv2.GaussianBlur(t * m[..., None], (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)[..., None]
    fill = num / np.maximum(den, 1e-4)
    if (den < 1e-3).any():                                          # nothing seen nearby: the wall's median
        med = np.median(tex[seen], 0) if seen.any() else np.array([225, 222, 215], np.float32)
        fill = np.where(den < 1e-3, med, fill)
    fill = cv2.resize(fill, (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(fill, 0, 255).astype(np.uint8)


def match_stats(fill: np.ndarray, ref: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Per-channel Lab mean / std of the filled texels matched to the seen texels of the same plane (the quilted
    patches come from the brighter, nearer floor and showed as pale halos)."""
    a = cv2.cvtColor(fill, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = ~seen
    if m.sum() < 50 or seen.sum() < 50:
        return fill
    for c in range(3):
        ma, sa = a[..., c][m].mean(), a[..., c][m].std() + 1e-3
        mb, sb = b[..., c][seen].mean(), b[..., c][seen].std() + 1e-3
        a[..., c] = (a[..., c] - ma) / sa * min(sb, 1.5 * sa) + mb
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _feather(seen_tex, fill, seen, r=4):
    a = cv2.GaussianBlur(seen.astype(np.float32), (0, 0), r)
    a = np.where(seen, np.maximum(a, 0.5), a * 0.8)
    a = np.where(seen, 1.0, a)
    return (seen_tex * a[..., None] + fill * (1 - a[..., None])).astype(np.uint8)


def shell_textures(rd, lay, pano, seed: int = 0, seg=None) -> dict:
    """{"floor": img, "S"/"N"/"W"/"E": img, "seen": {...share}}."""
    c0 = lay["camera"]
    out, share = {}, {}
    for kind, wall in [("floor", None), ("wall", "S"), ("wall", "N"), ("wall", "W"), ("wall", "E")]:
        P, (h, w) = plane_grid(lay, kind, wall)
        allowed = [ADE_FLOOR, ADE_RUG] if kind == "floor" else [ADE_WALL]
        col, seen = _seen_colors(rd, pano, c0, P.reshape(-1, 3), seg, allowed if seg is not None else None)
        col, seen = col.reshape(h, w, 3), seen.reshape(h, w)
        if seg is not None:                                         # drop thin slivers at class borders
            seen = cv2.erode(seen.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if kind == "floor":                                         # the panorama's nadir is distorted (swirls,
            Pf = P.reshape(h, w, 3)                                 # light streaks): use floor seen <= 55 deg
            hd = np.hypot(Pf[..., 0] - c0[0], Pf[..., 2] - c0[2])   # below the horizon only, quilt the rest
            seen &= hd >= c0[1] / np.tan(np.deg2rad(55))
            if seen.sum() > 100:                                    # specular highlights / sun patches stretched
                lum = col.astype(np.float32).mean(-1)               # along the view rays: treat as unseen,
                hi = seen & (lum > 1.2 * np.median(lum[seen]) + 12) # with their bright fringe (8 texels)
                hi = cv2.dilate(hi.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
                seen &= ~hi
        name = "floor" if kind == "floor" else wall
        share[name] = float(seen.mean())
        if kind == "floor":
            fill = quilt(col, seen, seed=seed)
            if fill is None:
                fill = smooth_fill(col, seen, 0.6 * PX_PER_M)
            else:
                fill = match_stats(fill, col, seen)
        else:
            fill = smooth_fill(col, seen, 0.6 * PX_PER_M)
        out[name] = _feather(col.astype(np.float32), fill.astype(np.float32), seen, r=6)
    out["seen"] = share
    return out


def color_furniture(rd, scene: dict, pano: np.ndarray, max_edge: float = 0.04, seg=None) -> list:
    """[(trimesh with per-face colours, inst)] of the furniture coloured from the panorama."""
    import trimesh
    lay = scene["layout"]
    c0 = np.asarray(lay["camera"], float)
    out = []
    eq_inst = None
    for mesh, inst in scene["parts"]:
        if inst < S.FURN0:
            continue
        m = mesh.copy()
        fc0 = np.asarray(m.visual.face_colors)[:, :3].astype(np.float32)
        try:
            v, f, idx = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge, max_iter=6, return_index=True)
            fc0 = fc0[idx]
            m = trimesh.Trimesh(v, f, process=False)
        except Exception:  # noqa: BLE001 (keep the original mesh)
            pass
        cen = m.triangles_center
        nrm = m.face_normals
        to_c = c0[None] - cen
        dist = np.linalg.norm(to_c, axis=1)
        facing = np.sum(nrm * to_c, 1) > 0
        col, seen = _seen_colors(rd, pano, c0, cen + nrm * 0.01, seg, None,
                                 [ADE_WALL, ADE_FLOOR, ADE_CEIL, ADE_WINDOW, ADE_DOOR] if seg is not None else None)
        seen &= facing | (np.abs(np.sum(nrm * to_c, 1)) / np.maximum(dist, 1e-6) < 0.05)
        lum = fc0.mean(1)
        if seen.sum() >= 20:
            dom = np.median(col[seen], 0)
        else:
            if eq_inst is None:
                eq_inst = rd.equirect(c0, 1024, 512)["inst"]
                eq_img = cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA)
            pix = eq_img[eq_inst == inst]
            dom = np.median(pix, 0) if len(pix) > 20 else fc0.mean(0)
        hidden = np.clip(dom[None] * (lum / max(lum.mean(), 1.0))[:, None] ** 0.6, 0, 255)
        fcol = np.where(seen[:, None], col, hidden).astype(np.float32)
        # per-vertex colours = mean of the adjacent faces (smooth, no triangle mosaic)
        vsum = np.zeros((len(m.vertices), 3), np.float32)
        vcnt = np.zeros(len(m.vertices), np.float32)
        for k in range(3):
            np.add.at(vsum, m.faces[:, k], fcol)
            np.add.at(vcnt, m.faces[:, k], 1)
        vcol = vsum / np.maximum(vcnt, 1)[:, None]
        m._vcolors = vcol
        fsm = vcol[m.faces].mean(1).astype(np.uint8)
        m.visual.face_colors = np.concatenate([fsm, np.full((len(fsm), 1), 255, np.uint8)], 1)
        out.append((m, inst))
    return out


def textured_parts(rd, scene, pano, seed: int = 0, seg=None) -> dict:
    """Everything needed for the dollhouse export and the top-down render. seg: ADE20K class map of the
    panorama (same size), enables the class-consistency test."""
    lay = scene["layout"]
    tex = shell_textures(rd, lay, pano, seed, seg)
    furn = color_furniture(rd, scene, pano, seg=seg)
    struct = [(m, i) for m, i in scene["parts"] if i in (S.DOOR, S.CURTAIN_ID, S.LAMP)
              or (i == S.WINDOW and not _is_backdrop(m, lay))]
    return {"tex": tex, "furniture": furn, "struct": struct, "layout": lay}


def _is_backdrop(m, lay) -> bool:
    b = m.bounds
    return b[0][0] < -0.5 or b[0][2] < -0.5 or b[1][0] > lay["W"] + 0.5 or b[1][2] > lay["D"] + 0.5


def _textured_quad_grid(lay, kind, wall, img):
    """Plane mesh with UVs (walls with holes at the openings) + its texture image."""
    import trimesh
    from PIL import Image
    if kind == "floor":
        W, D = lay["W"], lay["D"]
        v = np.array([[0, 0, 0], [W, 0, 0], [W, 0, D], [0, 0, D]], float)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        f = np.array([[0, 2, 1], [0, 3, 2]])
    else:
        o, a, n, L = S.wall_frame(lay, wall)
        Hh = lay["H"]
        holes = [(x["c"] - x["w"] / 2, x["c"] + x["w"] / 2, x["y0"], x["y1"]) for x in lay["openings"] if x["wall"] == wall]
        ss = sorted({0.0, L, *[h[0] for h in holes], *[h[1] for h in holes]})
        ys = sorted({0.0, Hh, *[h[2] for h in holes], *[h[3] for h in holes]})
        v, uv, f = [], [], []
        for i in range(len(ss) - 1):
            for j in range(len(ys) - 1):
                sc, yc = (ss[i] + ss[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
                if any(h[0] < sc < h[1] and h[2] < yc < h[3] for h in holes):
                    continue
                k = len(v)
                for s_, y_ in ((ss[i], ys[j]), (ss[i + 1], ys[j]), (ss[i + 1], ys[j + 1]), (ss[i], ys[j + 1])):
                    v.append(o + a * s_ + np.array([0, y_, 0]))
                    uv.append([s_ / L, y_ / Hh])
                tri = [[k, k + 1, k + 2], [k, k + 2, k + 3]]
                if np.dot(np.cross(v[k + 1] - v[k], v[k + 2] - v[k]), n) < 0:   # face into the room
                    tri = [[k, k + 2, k + 1], [k, k + 3, k + 2]]
                f += tri
        v, uv, f = np.array(v), np.array(uv), np.array(f)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(img), metallicFactor=0.0,
                                              roughnessFactor=1.0, doubleSided=False)
    m = trimesh.Trimesh(v, f, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return m


def export_glb(tp: dict, path) -> None:
    """Dollhouse .glb (y up; z flipped back to the right-handed glTF frame, so the model is not mirrored)."""
    import trimesh
    lay = tp["layout"]
    sc = trimesh.Scene()
    flip = np.diag([1.0, 1.0, -1.0, 1.0])

    def add(m, name):
        m = m.copy()
        m.apply_transform(flip)            # trimesh flips the face winding itself for a mirroring transform
        sc.add_geometry(m, geom_name=name)
    add(_textured_quad_grid(lay, "floor", None, np.ascontiguousarray(tp["tex"]["floor"])), "floor")
    for w in ("S", "N", "W", "E"):
        add(_textured_quad_grid(lay, "wall", w, np.ascontiguousarray(tp["tex"][w])), f"wall_{w}")
    for k, (m, i) in enumerate(tp["struct"] + tp["furniture"]):
        vc = getattr(m, "_vcolors", None)
        if vc is not None:
            mm = trimesh.Trimesh(m.vertices, m.faces, process=False)
            mm.visual.vertex_colors = np.concatenate([np.clip(vc, 0, 255).astype(np.uint8),
                                                      np.full((len(vc), 1), 255, np.uint8)], 1)
        else:
            fc = np.asarray(m.visual.face_colors).copy()
            mm = trimesh.Trimesh(m.vertices[m.faces].reshape(-1, 3), np.arange(len(m.faces) * 3).reshape(-1, 3),
                                 process=False)
            mm.visual.vertex_colors = np.repeat(fc, 3, 0)
        add(mm, f"part_{k}_{i}")
    sc.export(str(path))


def topdown(tp: dict, scene: dict, px_per_m: float = 220.0, margin: float = 0.25) -> np.ndarray:
    """Orthographic top view of the textured dollhouse (x right, z up), with instance outlines."""
    lay = tp["layout"]
    px_per_m = max(px_per_m, 1040.0 / (min(lay["W"], lay["D"]) + 2 * margin))   # >= 1024 px on both sides
    parts = [(m, i) for m, i in tp["furniture"]] + [(m, i) for m, i in tp["struct"] if i != S.LAMP]
    walls = []
    t = 0.12
    W, D = lay["W"], lay["D"]
    for (x0, z0, x1, z1) in [(-t, -t, W + t, 0), (-t, D, W + t, D + t), (-t, 0, 0, D), (W, 0, W + t, D)]:
        walls.append((S._box(x0, 0, z0, x1, lay["H"], z1, color=(60, 60, 60)), S.WALL))
    tr = R.Tracer(parts + walls)
    O, Dd = R.ortho_down(-margin, W + margin, -margin, D + margin, px_per_m, lay["H"] + 1)
    res = tr.cast(O, Dd)
    img = res["color"].astype(np.float32)
    # floor texture where the ray reaches the floor
    miss = ~np.isfinite(res["t"])
    fl = tp["tex"]["floor"]
    X, Z = O[..., 0], O[..., 2]
    inside = (X >= 0) & (X < W) & (Z >= 0) & (Z < D)
    fx = np.clip((X / W * fl.shape[1]).astype(int), 0, fl.shape[1] - 1)
    fz = np.clip(((D - Z) / D * fl.shape[0]).astype(int), 0, fl.shape[0] - 1)
    floor_hit = miss & inside
    img[floor_hit] = fl[fz[floor_hit], fx[floor_hit]]
    img[miss & ~inside] = 255
    # shading by height (higher = lighter) for readability
    hgt = np.where(np.isfinite(res["t"]), lay["H"] + 1 - res["t"], 0)
    img = np.where((res["inst"] >= S.FURN0)[..., None], img * (0.8 + 0.25 * np.clip(hgt / 1.2, 0, 1))[..., None], img)
    # outlines of instances
    inst = np.where(np.isfinite(res["t"]), res["inst"], -1)
    edge = np.zeros(inst.shape, bool)
    edge[:, 1:] |= inst[:, 1:] != inst[:, :-1]
    edge[1:, :] |= inst[1:, :] != inst[:-1, :]
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    img[edge & ((inst >= S.FURN0) | np.roll(inst >= S.FURN0, 1, 0) | np.roll(inst >= S.FURN0, 1, 1))] = (40, 40, 40)
    return np.clip(img, 0, 255).astype(np.uint8)
