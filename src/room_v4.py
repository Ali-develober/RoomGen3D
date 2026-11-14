"""v4 3D room: a sharp, clean 3D view (PLAN_V4 milestone 3).

Same geometry idea as v3 (src/pipeline_v3.build: box shell + observed surfaces in front of it for the inside
view; shell without ceiling + furniture solids for the dollhouse), with the losses of the v3 texture path removed:

    texture     the 4096 x 2048 refined panorama (milestone 2) instead of the 2048 x 1024 texture
    shell       plane textures at TEXEL_M per texel (v3: 1.5 cm, capped at 1024 px per side) and a SILHOUETTE
                BAND: texels just behind a furniture silhouette keep the projected panorama colour instead of
                being inpainted, so from the centre no inpainted wall shows through the stair-stepped edge of
                the furniture surface (rb.plane_texture(band_m=...))
    furniture   observed surfaces (inside view): grid STRIDE in panorama pixels; optional depth smoothing that
                cannot cross instance borders or depth edges (bilateral-style normalised convolution per
                instance) or radial Taubin smoothing of the mesh with its borders fixed. Both only move
                vertices ALONG their camera ray, so the projective texture stays exact from the centre
    atlas       parts packed in shelves (max ATLAS_W wide) instead of stacked in one tall column
    export      mipmapped sampler (LINEAR_MIPMAP_LINEAR), JPEG quality JPEG_Q; size reported
Frames as in v3: vertices in frame A (gravity aligned, camera at the origin).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import equirect as eq
from . import furniture as fu
from . import pipeline_v3 as p3v
from . import room_box as rb
from . import room_mesh as rm
from . import textured_mesh as tm

DEFAULTS = {"texel_m": 0.005, "max_tex": 4096, "band_m": 0.06, "stride": 1, "smooth": "full", "taubin_iters": 30,
            "atlas_w": 8192, "jpeg_q": 92, "mipmaps": True, "denoise": "none"}


def denoise_texture(img: np.ndarray, method: str) -> np.ndarray:
    """Edge-preserving denoising of a plane texture in its own texture space."""
    if method == "bilateral":
        return cv2.bilateralFilter(img, 7, 12, 3)
    if method == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 5, 15)
    return img
ATLAS_MAX = 16384


# --------------------------------------------------------------------------- #
# Furniture surfaces
# --------------------------------------------------------------------------- #
def smooth_depth_instances(dist: np.ndarray, keep: np.ndarray, labels: np.ndarray, sigma: float = 1.5,
                           rel: float = 0.05) -> np.ndarray:
    """Edge-preserving smoothing of the distance inside each label region (normalised Gaussian convolution
    restricted to the region, wrap-aware); a pixel keeps its value if the smoothed one differs by more than
    rel (a depth edge inside the region)."""
    out = dist.astype(np.float32).copy()
    P = int(4 * sigma) + 2
    for L in np.unique(labels[keep]):
        m = (keep & (labels == L)).astype(np.float32)
        if m.sum() < 9:
            continue
        num = cv2.GaussianBlur(cv2.copyMakeBorder(dist.astype(np.float32) * m, 0, 0, P, P, cv2.BORDER_WRAP), (0, 0), sigma)
        den = cv2.GaussianBlur(cv2.copyMakeBorder(m, 0, 0, P, P, cv2.BORDER_WRAP), (0, 0), sigma)
        sm = (num / np.maximum(den, 1e-6))[:, P:-P]
        ok = (m > 0) & (np.abs(sm - dist) <= rel * dist)
        out[ok] = sm[ok]
    return out


def radial_taubin(part: dict, iters: int, lam: float = 0.5, mu: float = -0.53) -> dict:
    """Taubin smoothing of the vertex DISTANCES (vertices move only along their ray from the camera), border
    vertices (on edges used by one face) fixed."""
    from scipy import sparse
    V, F = part["vertices"].astype(np.float64), np.asarray(part["faces"], np.int64)
    if iters <= 0 or len(F) == 0:
        return part
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    es = np.sort(e, 1)
    uniq, cnt = np.unique(es, axis=0, return_counts=True)
    border = np.zeros(n, bool)
    border[uniq[cnt == 1].ravel()] = True
    A = sparse.coo_matrix((np.ones(len(uniq) * 2), (np.r_[uniq[:, 0], uniq[:, 1]], np.r_[uniq[:, 1], uniq[:, 0]])),
                          shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    r = np.linalg.norm(V, axis=1)
    d = V / np.maximum(r[:, None], 1e-9)
    free = (~border) & (deg > 0)
    for _ in range(iters):
        for f in (lam, mu):
            avg = (A @ r) / np.maximum(deg, 1)
            r = np.where(free, r + f * (avg - r), r)
    out = dict(part)
    out["vertices"] = (d * r[:, None]).astype(np.float32)
    return out


def full_taubin(part: dict, R: np.ndarray, W: int, H: int, iters: int, lam: float = 0.5, mu: float = -0.53) -> dict:
    """Taubin smoothing of the 3D vertex positions: interior vertices average over all mesh neighbours, border
    vertices only along their border chain (so silhouettes become smooth curves instead of pixel staircases
    and do not shrink inwards). The UVs are then recomputed from each vertex's direction (projective texture:
    seen from the panorama centre every vertex still shows the panorama pixel on its own ray)."""
    from scipy import sparse
    V, F = part["vertices"].astype(np.float64), np.asarray(part["faces"], np.int64)
    if iters <= 0 or len(F) == 0:
        return part
    n = len(V)
    es = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    uniq, cnt = np.unique(es, axis=0, return_counts=True)
    bedge = uniq[cnt == 1]
    border = np.zeros(n, bool)
    border[bedge.ravel()] = True
    # interior vertices: all their edges; border vertices: only border edges
    i, j = np.r_[uniq[:, 0], uniq[:, 1]], np.r_[uniq[:, 1], uniq[:, 0]]
    sel = ~border[i]
    i = np.r_[i[sel], bedge[:, 0], bedge[:, 1]]
    j = np.r_[j[sel], bedge[:, 1], bedge[:, 0]]
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    P = V.copy()
    for _ in range(iters):
        for f in (lam, mu):
            avg = (A @ P) / np.maximum(deg, 1)[:, None]
            P = np.where(deg[:, None] > 0, P + f * (avg - P), P)
    out = dict(part)
    out["vertices"] = P.astype(np.float32)
    dc = P @ R                                                         # frame A -> panorama camera frame
    u, v = eq.dirs_to_uv(dc / np.linalg.norm(dc, axis=1, keepdims=True), W, H)
    nu, nv = (u + 0.5) / W, (v + 0.5) / H
    ou = part["uvs"][:, 0]
    nu = ou + ((nu - ou + 0.5) % 1.0 - 0.5)                             # keep each vertex on its side of the seam
    out["uvs"] = np.stack([nu, nv], 1).astype(np.float32)
    return out


def observed_part(dist: np.ndarray, texture: np.ndarray, R: np.ndarray, keep: np.ndarray, inst_map: np.ndarray,
                  stride: int = 1, smooth: str = "none", taubin_iters: int = 0) -> dict:
    """smooth: "none" | "instances" (depth smoothing inside instances) | "radial" (Taubin along the rays,
    borders fixed) | "full" (3D Taubin, borders along their chain, UVs re-projected)."""
    d = dist
    if smooth == "instances":
        lab = np.where(inst_map >= 0, inst_map, -2)
        d = smooth_depth_instances(dist, keep, lab)
    part = rb.observed_part(d, texture, R, keep, stride=stride)
    if smooth == "radial":
        return radial_taubin(part, taubin_iters)
    if smooth == "full":
        return full_taubin(part, R, dist.shape[1], dist.shape[0], taubin_iters)
    return part


# --------------------------------------------------------------------------- #
# Atlas packing
# --------------------------------------------------------------------------- #
def combine_packed(parts: list[dict], atlas_w: int = 8192, pad: int = 4) -> dict:
    """Like room_box.combine, but the part textures are packed in shelves (tallest first) into an atlas at most
    atlas_w wide; textures wider than atlas_w are downscaled to fit."""
    items = []
    for i, p in enumerate(parts):
        t = p["texture"]
        if t.shape[1] > atlas_w:
            s = atlas_w / t.shape[1]
            t = cv2.resize(t, (atlas_w, max(1, int(round(t.shape[0] * s)))), interpolation=cv2.INTER_AREA)
        items.append((i, t))
    order = sorted(items, key=lambda it: -it[1].shape[0])
    pos, x, y, shelf_h = {}, 0, 0, 0
    for i, t in order:
        h, w = t.shape[:2]
        if x + w > atlas_w:
            x, y, shelf_h = 0, y + shelf_h + pad, 0
        pos[i] = (x, y)
        x += w + pad
        shelf_h = max(shelf_h, h)
    Ha = y + shelf_h
    Wa = atlas_w if len(parts) > 1 else items[0][1].shape[1]
    Wa = min(Wa, max(pos[i][0] + t.shape[1] for i, t in items))
    atlas = np.full((Ha, Wa, 3), 127, np.uint8)
    verts, uvs, faces, kinds = [], [], [], []
    base = 0
    for i, t in items:
        p = parts[i]
        h, w = t.shape[:2]
        x0, y0 = pos[i]
        atlas[y0:y0 + h, x0:x0 + w] = t
        if pad:                                         # replicate the border into the padding (no bleeding)
            atlas[y0:y0 + h, x0 + w:min(x0 + w + pad, Wa)] = t[:, -1:]
            atlas[y0 + h:min(y0 + h + pad, Ha), x0:x0 + w] = t[-1:]
        uv = p["uvs"] * np.array([w / Wa, h / Ha]) + np.array([x0 / Wa, y0 / Ha])
        verts.append(p["vertices"])
        uvs.append(uv)
        faces.append(np.asarray(p["faces"]) + base)
        kinds += [p["kind"]] * len(p["faces"])
        base += len(p["vertices"])
    return {"vertices": np.concatenate(verts).astype(np.float32), "uvs": np.concatenate(uvs).astype(np.float32),
            "faces": np.concatenate(faces).astype(np.uint32), "texture": atlas, "face_kind": np.array(kinds)}


# --------------------------------------------------------------------------- #
# Whole room
# --------------------------------------------------------------------------- #
def build_meshes(dist: np.ndarray, hires: np.ndarray, res_v3: dict, ana: dict, id2label: dict,
                 opts: dict | None = None, block_solids: list | None = None, relief: np.ndarray | None = None) -> dict:
    """Inside and dollhouse meshes of the v4 room from the v3 analysis (box, solids) and the 4096 panorama.
    Milestone 4: block_solids = the solids that stay blocks (default: all), relief = mask of uncertain instances
    shown as their observed surface in the dollhouse."""
    o = {**DEFAULTS, **(opts or {})}
    box = res_v3["box"]
    shell_in = rb.build_shell(box, hires, dist, texel_m=o["texel_m"], max_tex=o["max_tex"], band_m=o["band_m"])
    if o["denoise"] != "none":
        for p in shell_in["parts"]:
            p["texture"] = denoise_texture(p["texture"], o["denoise"])
    keep = p3v.in_front_of_shell(dist, box)
    seen = observed_part(dist, hires, box["R"], keep, ana["inst_map"], o["stride"], o["smooth"], o["taubin_iters"])
    inside = combine_packed(shell_in["parts"] + ([seen] if len(seen["faces"]) else []), o["atlas_w"])
    shell_d = rb.build_shell(box, hires, dist, texel_m=o["texel_m"], max_tex=o["max_tex"], band_m=0.0) \
        if o["band_m"] > 0 else shell_in
    if o["denoise"] != "none" and shell_d is not shell_in:
        for p in shell_d["parts"]:
            p["texture"] = denoise_texture(p["texture"], o["denoise"])
    inst_map = ana["inst_map"]
    if fu.DEFAULT_PARAMS.get("leftovers"):
        inst_map, _ = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)
    blocks = res_v3["solids"] if block_solids is None else block_solids
    furn = fu.solids_part(blocks, box, dist, inst_map, hires) if blocks else {"faces": []}
    parts = [p for p in shell_d["parts"] if p["kind"] != "ceiling"] + ([furn] if len(furn["faces"]) else [])
    if relief is not None and relief.any():
        rp = rb.observed_part(dist, hires, box["R"], relief, stride=2)
        if len(rp["faces"]):
            rp["kind"] = "relief"
            parts.append(rp)
    doll = combine_packed(parts, o["atlas_w"])
    for m in (inside, doll):
        m.update({"box": box, "camera_height": box["camera_height"]})
    return {"inside": inside, "dollhouse": doll, "opts": o}


def export(meshes: dict, out_dir, tag: str = "v4") -> dict:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = meshes["opts"]
    files = {}
    for k in ("inside", "dollhouse"):
        p = out / f"{'room' if k == 'inside' else 'dollhouse'}_{tag}.glb"
        tm.export_glb(meshes[k], p, double_sided=False, mipmaps=o["mipmaps"], jpeg_quality=o["jpeg_q"])
        files[k] = str(p)
        files[f"{k}_mb"] = p.stat().st_size / 2 ** 20
        files[f"{k}_atlas"] = meshes[k]["texture"].shape[:2]
        files[f"{k}_triangles"] = int(len(meshes[k]["faces"]))
    return files


# --------------------------------------------------------------------------- #
# Offline renderer (for the metrics): exact ray casting, bilinear texture lookup, supersampling
# --------------------------------------------------------------------------- #
def render(mesh: dict, box: dict, yaw_B: float, hfov: float, w: int = 1024, h: int = 768, origin_A=None,
           ss: int = 2, bg=(0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """Level pinhole view (frame-B yaw, like design_shots) of a frame-A mesh from origin_A (default: the panorama
    centre). Returns (RGB uint8, hit mask)."""
    from .design_shots import view_rays, view_to_B
    r, _ = view_rays(w * ss, h * ss, hfov)
    dA = (r @ view_to_B(yaw_B).T @ box["Q"].T).reshape(-1, 3)
    dA /= np.linalg.norm(dA, axis=1, keepdims=True)
    o = np.zeros(3) if origin_A is None else np.asarray(origin_A, float)
    rc = rm.Raycaster(mesh)
    rays = np.concatenate([np.broadcast_to(o, dA.shape), dA], 1).astype(np.float32)
    ans = rc.scene.cast_rays(rc.o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy()
    hit = np.isfinite(t)
    prim, b = ans["primitive_ids"].numpy(), ans["primitive_uvs"].numpy()
    img = np.zeros((len(t), 3), np.float32) + np.array(bg, np.float32)
    f = mesh["faces"][prim[hit]]
    wgt = np.stack([1 - b[hit, 0] - b[hit, 1], b[hit, 0], b[hit, 1]], 1)
    uv = (mesh["uvs"][f] * wgt[:, :, None]).sum(1)
    tex = mesh["texture"]
    img[hit] = bilinear(tex, uv[:, 0] * tex.shape[1] - 0.5, uv[:, 1] * tex.shape[0] - 0.5)
    img = img.reshape(h * ss, w * ss, 3)
    hm = hit.reshape(h * ss, w * ss).astype(np.float32)
    if ss > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255).astype(np.uint8), hm > 0.5


def bilinear(tex: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup of an H x W x 3 image at continuous pixel coordinates (border clamped)."""
    H, W = tex.shape[:2]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    t = tex.astype(np.float32)
    return ((t[y0, x0] * (1 - fx) + t[y0, x1] * fx) * (1 - fy) + (t[y1, x0] * (1 - fx) + t[y1, x1] * fx) * fy)


def kind_mask(mesh: dict, box: dict, yaw_B: float, hfov: float, kind: str = "observed", w: int = 1024, h: int = 768,
              origin_A=None) -> np.ndarray:
    """Pixels whose first hit is a face of the given kind (silhouette of the furniture surfaces)."""
    from .design_shots import view_rays, view_to_B
    r, _ = view_rays(w, h, hfov)
    dA = (r @ view_to_B(yaw_B).T @ box["Q"].T).reshape(-1, 3)
    dA /= np.linalg.norm(dA, axis=1, keepdims=True)
    o = np.zeros(3) if origin_A is None else np.asarray(origin_A, float)
    rc = rm.Raycaster(mesh)
    ans = rc.scene.cast_rays(rc.o3d.core.Tensor(np.concatenate([np.broadcast_to(o, dA.shape), dA], 1).astype(np.float32)))
    prim = ans["primitive_ids"].numpy()
    hit = np.isfinite(ans["t_hit"].numpy())
    k = np.zeros(len(prim), bool)
    k[hit] = mesh["face_kind"][prim[hit]] == kind
    return k.reshape(h, w)


def noise_sigma(img: np.ndarray, mask: np.ndarray, flat: float = 12.0) -> float:
    """Grain: Immerkaer's noise estimate (mean |Laplacian-difference kernel| * sqrt(pi/2) / 6) over the mask
    pixels whose Sobel gradient is below `flat` (texture-free areas), grey levels."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, N))
    gm = np.hypot(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F, 0, 1))
    m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    m &= gm < flat
    return float(np.sqrt(np.pi / 2) / 6 * r[m].mean()) if m.sum() > 500 else float("nan")


def grain_band(img: np.ndarray, mask: np.ndarray) -> float:
    """Mid-band texture energy (std of a difference of Gaussians, sigma 1 - 4 px) on the mask pixels: the blotchy
    'grain' scale. Real fine texture (wallpaper, wood) also counts, so compare variants of the SAME image."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 4.0)
    m = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float(b[m].std()) if m.sum() > 500 else float("nan")


def jaggedness(mask: np.ndarray, sigma: float = 3.0) -> float:
    """Stair-step measure of silhouettes: contour length / length of the Gaussian-smoothed contour (1 = smooth;
    a staircase is ~sqrt(2) = 1.41 times longer than the diagonal it approximates)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    L, Ls = 0.0, 0.0
    for c in cs:
        c = c[:, 0].astype(np.float64)
        if len(c) < 30:
            continue
        k = int(3 * sigma)
        ext = np.concatenate([c[-k:], c, c[:k]])
        g = cv2.getGaussianKernel(2 * k + 1, sigma)[:, 0]
        sm = np.stack([np.convolve(ext[:, j], g, mode="valid") for j in range(2)], 1)
        L += np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1).sum()
        Ls += np.linalg.norm(np.diff(np.vstack([sm, sm[:1]]), axis=0), axis=1).sum()
    return float(L / Ls) if Ls > 0 else float("nan")
