"""PLAN_V5 milestone 5: a clean dollhouse.

v4 projected the panorama onto the room box as it is, so every piece of furniture standing against a wall or on the
floor was also painted, stretched, onto that wall / floor (and the furniture blocks carried stretched texture on the
faces the camera never saw). v5:

    shell textures   for every plane (walls, floor; rectified texture space, src/room_box.plane_texture):
                       mask = texels whose panorama pixel is furniture / an object (class map + instances), plus
                              texels hidden behind something, dilated by DILATE_M
                       fill = inpainting in the plane's own texture space:
                              "telea"  cv2 Telea (fast, smooth)
                              "lama"   LaMa (big-lama TorchScript, checkpoints/lama/lama.pt), at <= LAMA_MAX px
                              "sd"     SD 1.5 inpainting with a plain "empty floor / bare wall" prompt (<= 768 px)
                       clean = edge-preserving denoising of the whole plane texture ("bilateral" / "nlm" / "none")
    blocks           v3 footprints / heights / orientation; faces the camera saw keep the projected texture (only
                     triangles whose three corners are visible); every other triangle gets the piece's median colour
                     (flat, sides a little darker than the top) instead of stretched texture
The choice of fill / clean is made on the development rooms (scripts/eval_v5_doll.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from . import equirect as eq
from . import furniture as fu
from . import room_box as rb

ROOT = Path(__file__).resolve().parents[1]
LAMA = ROOT / "checkpoints" / "lama" / "lama.pt"
DILATE_M = 0.05
LAMA_MAX = 1024
OBJECT_CLASSES = {"lamp", "plant", "flower", "pot", "flowerpot", "vase", "television receiver", "computer", "box",
                  "basket", "bag", "bottle", "book", "clock", "fan", "toy", "pillow", "cushion", "towel", "trash can",
                  "ashcan", "screen", "monitor", "crt screen", "apparel", "radiator", "kitchen island", "stool",
                  "bucket", "tray", "bowl", "glass", "food", "fruit", "sculpture", "chandelier", "light"}
DEFAULT = {"fill": "lama", "clean": "bilateral"}


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def object_mask(class_map: np.ndarray, inst_map: np.ndarray | None, id2label: dict) -> np.ndarray:
    """Panorama pixels that are furniture or a loose object (not wall / floor / ceiling / window / door)."""
    from .floorplan import FURNITURE, class_ids
    names = set(FURNITURE) | OBJECT_CLASSES
    m = np.isin(class_map, class_ids(id2label, names))
    if inst_map is not None:
        m |= inst_map >= 0
    return m


def plane_mask(pl: dict, box: dict, mask_pano: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Texels of a shell plane (texture size `shape` = (th, tw)) whose panorama pixel is in mask_pano."""
    th, tw = shape
    s = (np.arange(tw) + 0.5) / tw
    t = (np.arange(th) + 0.5) / th
    PB = pl["o"] + s[None, :, None] * pl["a"] + t[:, None, None] * pl["b"]
    Pcam = (PB @ box["Q"].T) @ box["R"]
    return rb._sample_pano(mask_pano.astype(np.uint8) * 255, Pcam, cv2.INTER_NEAREST) > 127


# --------------------------------------------------------------------------- #
# Fills
# --------------------------------------------------------------------------- #
_LAMA = {}


def load_lama(device):
    if "m" not in _LAMA:
        m = torch.jit.load(str(LAMA), map_location="cpu").eval()
        _LAMA["m"] = m.to(device)
    return _LAMA["m"]


def free_lama():
    _LAMA.clear()


@torch.inference_mode()
def lama_fill(img: np.ndarray, mask: np.ndarray, device, max_side: int = LAMA_MAX) -> np.ndarray:
    """LaMa on the texture downscaled to <= max_side (multiple of 8); only masked texels are replaced, and they keep
    the fill at full resolution (upscaled) - the rest of the texture stays untouched."""
    h, w = mask.shape
    s = min(1.0, max_side / max(h, w))
    sw, sh = max(8, int(round(w * s / 8)) * 8), max(8, int(round(h * s / 8)) * 8)
    im = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    mk = cv2.dilate(mk, np.ones((3, 3), np.uint8))
    m = load_lama(device)
    x = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(device)
    y = torch.from_numpy(mk)[None, None].float().to(device)
    out = m(x, y)[0].permute(1, 2, 0).float().cpu().numpy()
    if out.max() <= 1.5:
        out = out * 255
    out = cv2.resize(np.clip(out, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    res = img.copy()
    res[mask] = out[mask]
    return res


_SD = {}


def load_sd_inpaint(device):
    if "p" not in _SD:
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
        from .generate_2d import BASE_MODEL
        p = StableDiffusionInpaintPipeline.from_pretrained(BASE_MODEL, dtype=torch.float16, variant="fp16",
                                                           safety_checker=None, requires_safety_checker=False)
        p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        _SD["p"] = p.to(device)
    return _SD["p"]


def free_sd():
    _SD.clear()


@torch.inference_mode()
def sd_fill(img: np.ndarray, mask: np.ndarray, kind: str, device, max_side: int = 768, seed: int = 0) -> np.ndarray:
    from PIL import Image
    pipe = load_sd_inpaint(device)
    h, w = mask.shape
    s = max_side / max(h, w)
    sw, sh = max(64, int(round(w * s / 8)) * 8), max(64, int(round(h * s / 8)) * 8)
    im = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    mk = cv2.dilate(cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST),
                    np.ones((5, 5), np.uint8))
    what = "empty floor, plain floor texture seen from above" if kind == "floor" else "bare empty wall, plain wall texture"
    out = pipe(prompt=f"{what}, photo, uniform, clean", negative_prompt="furniture, object, chair, table, bed, sofa, "
               "lamp, plant, person, text", image=Image.fromarray(im), mask_image=Image.fromarray(mk), height=sh,
               width=sw, num_inference_steps=30, guidance_scale=6.0, strength=1.0,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    out = cv2.resize(np.asarray(out.convert("RGB")), (w, h), interpolation=cv2.INTER_CUBIC)
    res = img.copy()
    res[mask] = out[mask]
    return res


def fill(img: np.ndarray, mask: np.ndarray, method: str, kind: str, device=None) -> np.ndarray:
    if not mask.any() or method == "none":
        return img
    if method == "telea":
        return cv2.inpaint(img, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    if method == "lama":
        return lama_fill(img, mask, device)
    if method == "sd":
        return sd_fill(img, mask, kind, device)
    return rb.inpaint(img, mask, method)


def clean(img: np.ndarray, method: str) -> np.ndarray:
    """Edge-preserving denoising of a plane texture."""
    if method == "bilateral":
        return cv2.bilateralFilter(img, 9, 18, 5)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 5, 15)
    if method == "guided":
        return cv2.edgePreservingFilter(img, flags=cv2.RECURS_FILTER, sigma_s=20, sigma_r=0.12)
    return img


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
def clean_shell(box: dict, texture: np.ndarray, dist: np.ndarray, obj_pano: np.ndarray, method: str = "lama",
                denoise: str = "bilateral", device=None, texel_m: float = 0.005, max_tex: int = 4096,
                with_ceiling: bool = False) -> dict:
    """Shell parts (frame A) with furniture removed from wall / floor textures. Returns parts + per-plane masks."""
    out = rb.build_shell(box, texture, dist, "none", texel_m=texel_m, max_tex=max_tex)
    planes = {p["kind"]: p for p in rb.shell_planes(box)}
    stats = {}
    for part in out["parts"]:
        k = part["kind"]
        if k == "ceiling" and not with_ceiling:
            continue
        tex = part["texture"]
        pm = plane_mask(planes[k], box, obj_pano, tex.shape[:2])
        _, hidden, _ = rb.plane_texture(planes[k], box, cv2.resize(texture, (256, 128)), dist, "none",
                                        texel_m=texel_m, max_tex=max_tex)
        if hidden.shape != pm.shape:
            hidden = cv2.resize(hidden.astype(np.uint8), pm.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        texel = max(texel_m, max(np.linalg.norm(planes[k]["a"]), np.linalg.norm(planes[k]["b"])) / max_tex)
        r = max(1, int(round(DILATE_M / texel)))
        mask = cv2.dilate((pm | hidden).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        part["texture"] = clean(fill(tex, mask, method, "floor" if k == "floor" else "wall", device), denoise)
        part["mask"] = mask
        stats[k] = {"masked_share": float(mask.mean()), "object_share": float(pm.mean())}
    out["stats"] = stats
    return out


# --------------------------------------------------------------------------- #
# Blocks with flat colour on unseen faces
# --------------------------------------------------------------------------- #
def blocks_part(solids: list[dict], box: dict, dist: np.ndarray, inst_map: np.ndarray, texture: np.ndarray,
                cell: float = fu.CELL) -> dict:
    """Furniture blocks (frame A): triangles whose 3 corners the camera saw keep the panorama projection, all others
    the piece's median colour (sides x SIDE_SHADE). The flat colours live in swatches appended under the texture."""
    H, W = dist.shape
    th, tw = texture.shape[:2]
    ext_w = tw + tw // 4
    n = max(len(solids), 1)
    sw = 16
    swatch_rows = 2 * sw
    tex = np.zeros((th + swatch_rows, ext_w, 3), np.uint8)
    tex[:th] = np.concatenate([texture, texture[:, :tw // 4]], 1)
    hires_inst = cv2.resize(inst_map.astype(np.int32), (tw, th), interpolation=cv2.INTER_NEAREST)
    verts, uvs, faces = [], [], []
    base = 0
    for si, s in enumerate(solids):
        m = hires_inst == s.get("instance", -99)
        col = np.median(texture[m], 0) if m.sum() > 20 else np.array([150, 140, 130], float)
        x0 = (si * sw) % (ext_w - sw)
        tex[th:th + sw, x0:x0 + sw] = col.astype(np.uint8)                          # top colour
        tex[th + sw:th + 2 * sw, x0:x0 + sw] = (col * 0.82).astype(np.uint8)        # side colour (flat shading)
        top_uv = ((x0 + sw / 2) / ext_w, (th + sw / 2) / tex.shape[0])
        side_uv = ((x0 + sw / 2) / ext_w, (th + 1.5 * sw) / tex.shape[0])
        F = fu._faces_of(s, box)
        centre = np.array([s["c"][0], box["floor_y"] - (s["top"] + s["bottom"]) / 2, s["c"][1]])
        for f in F:
            na = max(int(np.ceil(np.linalg.norm(f["a"]) / cell)), 1)
            nb = max(int(np.ceil(np.linalg.norm(f["b"]) / cell)), 1)
            sa, sb = np.linspace(0, 1, na + 1), np.linspace(0, 1, nb + 1)
            PB = f["o"] + sa[None, :, None] * f["a"] + sb[:, None, None] * f["b"]
            Pc = (PB @ box["Q"].T) @ box["R"]
            r = np.linalg.norm(Pc, axis=-1)
            u, v = eq.dirs_to_uv(Pc / r[..., None], W, H)
            ui = np.clip(np.round(u).astype(int) % W, 0, W - 1)
            vi = np.clip(np.round(v).astype(int), 0, H - 1)
            nrm = np.cross(f["a"], f["b"])
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
            if nrm @ (centre - (f["o"] + 0.5 * f["a"] + 0.5 * f["b"])) > 0:
                nrm = -nrm
            facing = (nrm @ box["Q"].T @ box["R"] @ (-Pc.reshape(-1, 3)).T).reshape(r.shape) > 0
            vis = facing & (np.abs(dist[vi, ui] - r) < np.maximum(0.12, 0.08 * r)) & (inst_map[vi, ui] == s["instance"])
            PA = (PB @ box["Q"].T).reshape(-1, 3)
            idx = np.arange((na + 1) * (nb + 1)).reshape(nb + 1, na + 1)
            q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
            tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            p0, p1, p2 = PA[tri[0, 0]], PA[tri[0, 1]], PA[tri[0, 2]]
            if np.cross(p1 - p0, p2 - p0) @ (nrm @ box["Q"].T) < 0:
                tri = tri[:, ::-1]
            vv_ = vis.reshape(-1)[tri].all(1)
            tu = (u.reshape(-1)[tri] + 0.5) / W
            span = tu.max(1) - tu.min(1)
            wrap = span > 0.5
            tu[wrap] = np.where(tu[wrap] < 0.5, tu[wrap] + 1.0, tu[wrap])
            tvv = (v.reshape(-1)[tri] + 0.5) / H
            U = tu * tw / ext_w
            V = tvv * th / tex.shape[0]
            flat = top_uv if f["kind"] == "top" else side_uv
            U[~vv_] = flat[0]
            V[~vv_] = flat[1]
            tv = PA[tri].reshape(-1, 3)
            verts.append(tv)
            uvs.append(np.stack([U.reshape(-1), V.reshape(-1)], 1))
            faces.append(np.arange(len(tv)).reshape(-1, 3) + base)
            base += len(tv)
    if not verts:
        return {"vertices": np.zeros((0, 3)), "uvs": np.zeros((0, 2)), "faces": np.zeros((0, 3), int),
                "texture": tex, "kind": "furniture"}
    return {"vertices": np.concatenate(verts), "uvs": np.concatenate(uvs), "faces": np.concatenate(faces),
            "texture": tex, "kind": "furniture"}


def build_dollhouse(dist, hires, res, ana, id2label, device=None, fill_method: str = DEFAULT["fill"],
                    denoise: str = DEFAULT["clean"], relief: np.ndarray | None = None) -> dict:
    """v5 dollhouse mesh (frame A): clean shell without ceiling + flat-coloured blocks (+ relief surfaces)."""
    from . import room_v4 as r4
    box = res["box"]
    inst_map = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)[0]
    obj = object_mask(ana["class_map"], inst_map, id2label)
    shell = clean_shell(box, hires, dist, obj, fill_method, denoise, device)
    parts = [p for p in shell["parts"] if p["kind"] != "ceiling"]
    blocks = blocks_part(res["solids"], box, dist, inst_map, hires) if res["solids"] else {"faces": []}
    if len(blocks["faces"]):
        parts.append(blocks)
    if relief is not None and relief.any():
        rp = rb.observed_part(dist, hires, box["R"], relief, stride=2)
        if len(rp["faces"]):
            rp["kind"] = "relief"
            parts.append(rp)
    doll = r4.combine_packed(parts, 8192)
    doll.update({"box": box, "camera_height": box["camera_height"], "shell_stats": shell["stats"],
                 "plane_textures": {p["kind"]: p["texture"] for p in shell["parts"] if p["kind"] != "ceiling"},
                 "plane_masks": {p["kind"]: p["mask"] for p in shell["parts"] if "mask" in p}})
    return doll
