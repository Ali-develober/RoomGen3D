"""Real furniture shapes: object-level image-to-3D (PLAN_V4 milestone 4).

Model: TripoSR (stabilityai/TripoSR, MIT; code github.com/VAST-AI-Research/TripoSR in checkpoints/triposr_code,
weights in checkpoints/triposr). Its only compiled dependency, torchmcubes, is replaced by scikit-image's
marching cubes (same conventions, see _install_mcubes). Stable Fast 3D (gated + compiled C++ extensions, no
compiler here) and Hunyuan3D-2mini (23.5 GB repository, compiled rasterizer for textures) were not usable:
README, milestone 4.

Per furniture instance (v3 solid with its panoptic instance mask on the panorama grid):
    view       the panorama has ONE viewpoint, so the "best view" of an instance is the perspective crop centred on
               its mask (mean direction), FOV = 1.3 x its angular extent (30-110 deg), cut from the 4096 panorama;
               occlusion = share of the instance's hull covered by CLOSER other instances
    input      mask cleaned (open / close, largest component), occluded parts of the hull inpainted (Telea) and
               added to the mask, edge feathered, composited on grey, object scaled to 85 % of a 512 square
    generate   TripoSR -> density grid 256^3 -> marching cubes -> mesh with vertex colours (object frame:
               front = +x towards the input camera, image right = +y, up = +z)
    orient     the object is mapped into room frame B facing the panorama camera, then render-and-compare: for
               candidate yaws (every 15 deg + the four wall-aligned directions) it is scaled into the v3 solid's
               box (footprint u x v, height bottom..top: rests on the floor / hangs from the ceiling exactly like
               the solid, which v3 already made collision-free), ray-cast from the panorama centre, and the
               silhouette IoU against the instance mask is computed (pixels of closer instances ignored);
               the best yaw wins (+WALL_BONUS for wall-aligned fronts)
    colour     vertices visible from the panorama centre take the panorama colour (projective), the rest keep
               TripoSR's colour
    fallback   IoU < IOU_MIN (chosen on the development rooms) or a class in BLOCK_CLASSES -> the v3 block
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from . import equirect as eq

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "checkpoints" / "triposr_code"
WEIGHTS = ROOT / "checkpoints" / "triposr"
MC_RES = 256
IN_SIZE = 512
FG_RATIO = 0.85
YAW_STEP = 15.0
WALL_BONUS = 0.02
IOU_MIN = 0.5                 # replaced by the development choice (results/tables/objects_selection.json)
BLOCK_CLASSES: set = set()    # classes kept as blocks (development choice)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _install_mcubes():
    """A drop-in `torchmcubes` module backed by skimage.measure.marching_cubes. torchmcubes returns vertices in
    (dim2, dim1, dim0) order (TripoSR swaps them back with [..., [2, 1, 0]]), so skimage's (dim0, dim1, dim2)
    vertices are reversed here."""
    if "torchmcubes" in sys.modules:
        return
    from skimage.measure import marching_cubes as sk_mc

    def marching_cubes(vol: torch.Tensor, thresh: float):
        v = vol.detach().float().cpu().numpy()
        if not (v.min() < thresh < v.max()):
            return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long)
        verts, faces, _, _ = sk_mc(v, thresh)
        return (torch.from_numpy(np.ascontiguousarray(verts[:, ::-1])).float(),
                torch.from_numpy(faces.astype(np.int64)))

    m = types.ModuleType("torchmcubes")
    m.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = m


def _remap_vit_keys(ckpt: dict, target: dict) -> dict:
    """transformers 5 renamed the ViT (DINO) layers: encoder.layer.N.attention.attention.query -> layers.N.attention.
    q_proj, ... The weights are identical; only the names in TripoSR's checkpoint are mapped (if needed)."""
    if all(k in target for k in ckpt):
        return ckpt
    ren = [(".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
           (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj."),
           (".intermediate.dense.", ".mlp.fc1."), (".output.dense.", ".mlp.fc2.")]
    out = {}
    for k, v in ckpt.items():
        nk = k.replace("image_tokenizer.model.encoder.layer.", "image_tokenizer.model.layers.")
        if nk != k:
            for a, b in ren:
                if a in nk:
                    nk = nk.replace(a, b)
                    break
        out[nk] = v
    missing = [k for k in target if k not in out]
    extra = [k for k in out if k not in target]
    if missing or extra:
        raise RuntimeError(f"TripoSR key remap incomplete: missing {missing[:5]}, unexpected {extra[:5]}")
    return out


def load_triposr(device, chunk: int = 8192):
    _install_mcubes()
    if "rembg" not in sys.modules:          # imported by tsr.utils but only needed for background removal;
        sys.modules["rembg"] = types.ModuleType("rembg")       # we cut objects out with our own masks
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from omegaconf import OmegaConf
    from tsr.system import TSR
    cfg = OmegaConf.load(str(WEIGHTS / "config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(WEIGHTS / "model.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(_remap_vit_keys(ckpt, model.state_dict()))
    model.renderer.set_chunk_size(chunk)
    return model.to(device).eval()


@torch.inference_mode()
def generate(model, img: np.ndarray, device, resolution: int = MC_RES) -> dict:
    """RGB uint8 square image (object on grey) -> {"vertices", "faces", "colors" (uint8)} in TripoSR's frame."""
    from PIL import Image
    codes = model([Image.fromarray(img)], device=device)
    mesh = model.extract_mesh(codes, True, resolution=resolution)[0]
    col = np.asarray(mesh.visual.vertex_colors)[:, :3] if len(mesh.vertices) else np.zeros((0, 3), np.uint8)
    return {"vertices": np.asarray(mesh.vertices, np.float64), "faces": np.asarray(mesh.faces, np.int64),
            "colors": col.astype(np.uint8)}


# --------------------------------------------------------------------------- #
# Input view
# --------------------------------------------------------------------------- #
def instance_view(hires: np.ndarray, inst_map: np.ndarray, dist: np.ndarray, iid: int, size: int = IN_SIZE) -> dict:
    """Perspective crop centred on instance iid (see module docstring). Returns crop, mask, occluder mask,
    yaw / pitch / fov (pano camera frame), angular size, occlusion share."""
    H, W = inst_map.shape
    m = inst_map == iid
    d = eq.pixel_dirs(W, H)[m]
    c = d.mean(0)
    c /= np.linalg.norm(c)
    ang = np.degrees(np.arccos(np.clip(d @ c, -1, 1)))
    ext = float(np.percentile(ang, 99))
    fov = float(np.clip(2.6 * ext, 30, 110))
    yaw = float(np.degrees(np.arctan2(c[0], c[2])))
    pitch = float(np.degrees(np.arcsin(np.clip(-c[1], -1, 1))))
    crop = eq.e2p(hires, fov, yaw, pitch, size)
    lab = eq.e2p(inst_map.astype(np.float32), fov, yaw, pitch, size, cv2.INTER_NEAREST).astype(np.int32)
    dv = eq.e2p(dist.astype(np.float32), fov, yaw, pitch, size, cv2.INTER_NEAREST)
    mask = lab == iid
    own_d = float(np.median(dist[m]))
    occ = (lab >= 0) & (lab != iid) & (dv < own_d * 0.97)
    hull = _hull(mask)
    occl_share = float((occ & hull).sum() / max(hull.sum(), 1))
    return {"crop": crop, "mask": mask, "occ": occ, "hull": hull, "yaw": yaw, "pitch": pitch, "fov": fov,
            "extent_deg": ext, "occlusion": occl_share, "pixels": int(m.sum())}


def _hull(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    if len(xs) < 3:
        return mask.copy()
    h = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    cv2.fillConvexPoly(out.view(np.uint8), h, 1)
    return out.astype(bool)


def grabcut_refine(img: np.ndarray, m: np.ndarray, occ: np.ndarray, iters: int = 3) -> np.ndarray:
    """The panoptic mask comes from the 512 x 1024 panorama grid (blocky at crop resolution): GrabCut at crop
    resolution, seeded with sure foreground = mask eroded 6 px, probable = mask and a 12 px ring around it,
    sure background = everything else and the closer occluders."""
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    ring = cv2.dilate(m, np.ones((25, 25), np.uint8)) > 0
    gc[ring] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, np.ones((13, 13), np.uint8)) > 0] = cv2.GC_FGD
    gc[occ & (m == 0)] = cv2.GC_BGD
    try:
        cv2.grabCut(img[..., ::-1].copy(), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out)
    if n > 1:
        out = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return out if out.sum() > 0.5 * m.sum() else m


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill holes that are not connected to the border (gaps between cushions etc.)."""
    inv = (m == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = m.copy()
    for k in range(1, n):
        if k not in border:
            out[lab == k] = 1
    return out


def prepare_input(v: dict, size: int = IN_SIZE, ratio: float = FG_RATIO, inpaint: bool = True,
                  refine: bool = True) -> tuple[np.ndarray, dict]:
    """Clean mask (GrabCut refinement at crop resolution, holes filled), inpaint occluded hull parts, feather,
    grey background, re-centre + scale to `ratio`."""
    m = v["mask"].astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if refine and m.sum() > 200:
        m = grabcut_refine(v["crop"], m, v["occ"])
    m = fill_holes(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    img = v["crop"].copy()
    fill = (v["occ"] & v["hull"]).astype(np.uint8)
    if inpaint and fill.any():
        img = cv2.inpaint(img, cv2.dilate(fill, k), 5, cv2.INPAINT_TELEA)
        m = np.maximum(m, fill)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
    comp = img.astype(np.float32) * alpha[..., None] + 127.5 * (1 - alpha[..., None])
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.full((size, size, 3), 127, np.uint8), {"ok": False}
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    obj = comp[y0:y1, x0:x1]
    s = ratio * size / max(x1 - x0, y1 - y0)
    obj = cv2.resize(obj, (max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))), interpolation=cv2.INTER_CUBIC)
    out = np.full((size, size, 3), 127.5, np.float32)
    oy, ox = (size - obj.shape[0]) // 2, (size - obj.shape[1]) // 2
    out[oy:oy + obj.shape[0], ox:ox + obj.shape[1]] = obj
    return np.clip(out, 0, 255).astype(np.uint8), {"ok": True, "inpainted_share": float(fill.sum() / max(m.sum(), 1))}


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def _view_dir_B(v: dict, box: dict) -> np.ndarray:
    y, p = np.radians(v["yaw"]), np.radians(v["pitch"])
    d_cam = np.array([np.cos(p) * np.sin(y), -np.sin(p), np.cos(p) * np.cos(y)])
    return d_cam @ box["R"].T @ box["Q"]


def place(obj: dict, solid: dict, box: dict, front_B: np.ndarray) -> np.ndarray:
    """Object vertices -> frame B: front (+x obj) along front_B (horizontal unit), up (+z obj) = up (-Y in B),
    then scaled anisotropically into the solid's box (u, v footprint axes, bottom..top height)."""
    f = np.array([front_B[0], 0.0, front_B[2]])
    f /= np.linalg.norm(f)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(up, f)          # object +y = image right as seen from the camera that looks along -f
    right = -right if np.dot(np.cross(f, right), up) < 0 else right
    M = np.stack([f, right, up], 1)   # columns: images of obj x, y, z
    P = obj["vertices"] @ M.T
    u = np.array([solid["u"][0], 0.0, solid["u"][1]])
    vv = np.array([solid["v"][0], 0.0, solid["v"][1]])
    a, b, h = P @ u, P @ vv, -P[:, 1]
    def fit(x, lo, hi):
        span = max(x.max() - x.min(), 1e-6)
        return lo + (x - x.min()) / span * (hi - lo)
    a = fit(a, -solid["e"][0] / 2, solid["e"][0] / 2)
    b = fit(b, -solid["e"][1] / 2, solid["e"][1] / 2)
    h = fit(h, solid["bottom"], solid["top"])
    c = np.array([solid["c"][0], 0.0, solid["c"][1]])
    return c + a[:, None] * u + b[:, None] * vv + (box["floor_y"] - h)[:, None] * np.array([0.0, 1.0, 0.0])


def _silhouette(VB: np.ndarray, F: np.ndarray, box: dict, rays_B: np.ndarray) -> np.ndarray:
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(VB.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    rays = np.concatenate([np.zeros_like(rays_B), rays_B], 1).astype(np.float32)
    return np.isfinite(sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())


def orient(obj: dict, solid: dict, box: dict, v: dict, inst_map: np.ndarray, dist: np.ndarray, iid: int) -> dict:
    """Render-and-compare yaw search (module docstring). Returns {"VB", "yaw_front", "iou", "iou_block",
    "front_B"}."""
    H, W = inst_map.shape
    m = inst_map == iid
    ys, xs = np.nonzero(m)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    xs_r = np.unique(np.concatenate([(xs + k) % W for k in range(-pad, pad + 1)]))
    sub_i = inst_map[y0:y1][:, xs_r]
    sub_d = dist[y0:y1][:, xs_r]
    dirs = eq.pixel_dirs(W, H)[y0:y1][:, xs_r]
    rays_B = (dirs @ box["R"].T @ box["Q"]).reshape(-1, 3)
    tgt = (sub_i == iid).ravel()
    own_d = float(np.median(dist[m]))
    care = ~(((sub_i >= 0) & (sub_i != iid) & (sub_d < own_d * 0.97)).ravel())   # closer instances: ignore
    def iou(sil):
        a, b = sil & care, tgt & care
        return float((a & b).sum() / max((a | b).sum(), 1))
    # block baseline: the v3 solid as a box
    from .furniture import _corners
    cr = _corners(solid)
    fy, top, bot = box["floor_y"], solid["top"], solid["bottom"]
    BV = np.array([[x, fy - hh, z] for hh in (bot, top) for x, z in cr])
    BF = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                   [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    iou_block = iou(_silhouette(BV, BF, box, rays_B))
    to_cam = -_view_dir_B(v, box)
    base = np.degrees(np.arctan2(to_cam[0], to_cam[2]))
    cands = [(base + k * YAW_STEP) % 360 for k in range(int(360 / YAW_STEP))] + [0.0, 90.0, 180.0, 270.0]
    best = None
    for yaw in cands:
        fB = np.array([np.sin(np.radians(yaw)), 0.0, np.cos(np.radians(yaw))])
        VB = place(obj, solid, box, fB)
        s = iou(_silhouette(VB, obj["faces"], box, rays_B))
        wall = min(abs((yaw % 90)), 90 - abs(yaw % 90)) < 1e-6
        sc = s + (WALL_BONUS if wall else 0.0)
        if best is None or sc > best["score"]:
            best = {"score": sc, "iou": s, "yaw_front": yaw, "VB": VB, "front_B": fB}
    best["iou_block"] = iou_block
    return best


def colour_from_pano(VB: np.ndarray, F: np.ndarray, colors: np.ndarray, box: dict, hires: np.ndarray,
                     dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Vertices visible from the panorama centre (ray from the centre hits this object first and the observed
    distance agrees) take the panorama colour. Returns (colours, visible share)."""
    import open3d as o3d
    Pc = VB @ box["Q"].T @ box["R"]
    r = np.linalg.norm(Pc, axis=1)
    d = Pc / r[:, None]
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(VB.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    dB = d @ box["R"].T @ box["Q"]
    t = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dB), dB], 1).astype(np.float32)))["t_hit"].numpy()
    H, W = dist.shape
    u, v = eq.dirs_to_uv(d, W, H)
    obs = dist[np.clip(np.round(v).astype(int), 0, H - 1), np.round(u).astype(int) % W]
    vis = (t > r - 0.03) & (np.abs(obs - r) < np.maximum(0.15, 0.1 * r))
    Hh, Wh = hires.shape[:2]
    uh, vh = eq.dirs_to_uv(d, Wh, Hh)
    pc = hires[np.clip(np.round(vh).astype(int), 0, Hh - 1), np.round(uh).astype(int) % Wh]
    out = colors.copy()
    out[vis] = pc[vis]
    return out, float(vis.mean())


def simplify(obj: dict, target: int = 8000) -> dict:
    """Quadric decimation (Open3D) to keep the GLB small; vertex colours carried by nearest original vertex."""
    import open3d as o3d
    if len(obj["faces"]) <= target:
        return obj
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj["vertices"]), o3d.utility.Vector3iVector(obj["faces"]))
    m.vertex_colors = o3d.utility.Vector3dVector(obj["colors"] / 255.0)
    s = m.simplify_quadric_decimation(target)
    return {"vertices": np.asarray(s.vertices), "faces": np.asarray(s.triangles, np.int64),
            "colors": (np.asarray(s.vertex_colors) * 255).clip(0, 255).astype(np.uint8)}


def build_objects(model, device, hires: np.ndarray, dist: np.ndarray, inst_map: np.ndarray, solids: list[dict],
                  box: dict, max_objects: int = 99, iou_min: float | None = None, block_classes=None,
                  cache_dir: Path | None = None, log=None) -> dict:
    """All solids, largest footprint first: generated object if allowed and good enough, else the v3 block.
    Returns {"objects": [{"solid", "VB" (frame B), "faces", "colors", "iou", "iou_block", ...}], "blocks": [solid],
    "rows": per-instance log, "seconds"}."""
    iou_min = IOU_MIN if iou_min is None else iou_min
    block_classes = BLOCK_CLASSES if block_classes is None else set(block_classes)
    t0 = time.perf_counter()
    order = sorted(range(len(solids)), key=lambda i: -solids[i]["e"][0] * solids[i]["e"][1])
    objs, blocks, rows = [], [], []
    for rank, i in enumerate(order):
        s = solids[i]
        row = {"label": s["label"], "instance": int(s.get("instance", -1)), "rank": rank}
        iid = int(s.get("instance", -1))
        if rank >= max_objects or iid < 0 or not (inst_map == iid).any():
            row["kind"] = "block"
            row["reason"] = "limit" if rank >= max_objects else "no mask"
            blocks.append(s)
            rows.append(row)
            continue
        ts = time.perf_counter()
        v = instance_view(hires, inst_map, dist, iid)
        img, info = prepare_input(v)
        ck = cache_dir / f"obj_{iid}.npz" if cache_dir else None
        if ck is not None and ck.exists():
            z = np.load(ck)
            obj = {"vertices": z["v"], "faces": z["f"], "colors": z["c"]}
        else:
            obj = generate(model, img, device)
            if ck is not None:
                ck.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(ck, v=obj["vertices"], f=obj["faces"], c=obj["colors"])
        t_gen = time.perf_counter() - ts
        row.update({"occlusion": v["occlusion"], "inpainted_share": info.get("inpainted_share", 0.0),
                    "extent_deg": v["extent_deg"], "gen_s": t_gen})
        if len(obj["faces"]) < 20:
            row.update({"kind": "block", "reason": "empty mesh", "iou": 0.0})
            blocks.append(s)
            rows.append(row)
            continue
        obj = simplify(obj)
        o = orient(obj, s, box, v, inst_map, dist, iid)
        cols, vis = colour_from_pano(o["VB"], obj["faces"], obj["colors"], box, hires, dist)
        row.update({"iou": o["iou"], "iou_block": o["iou_block"], "yaw_front": o["yaw_front"], "visible_share": vis,
                    "faces": int(len(obj["faces"])), "total_s": time.perf_counter() - ts})
        if s["label"] in block_classes or o["iou"] < iou_min:
            row["kind"] = "block"
            row["reason"] = "class" if s["label"] in block_classes else "iou"
            blocks.append(s)
        else:
            row["kind"] = "object"
            objs.append({"solid": s, "VB": o["VB"], "faces": obj["faces"], "colors": cols, "iou": o["iou"],
                         "iou_block": o["iou_block"], "front_B": o["front_B"], "input": img})
        rows.append(row)
        if log:
            log(row)
    return {"objects": objs, "blocks": blocks, "rows": rows, "seconds": time.perf_counter() - t0}
