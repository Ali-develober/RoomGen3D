"""Instance segmentation of a 360 degree panorama (PLAN_V3 milestone 1).

v2 split furniture into instances by connected components of a semantic class map (SegFormer on 14
views, labels projected back): touching objects of one class became one instance, one object
split by an occluder became two, and an object crossing the panorama seam (lon = +-180 deg) was
always cut in two because connected components do not wrap.

v3:
    panorama
      -> 13 perspective views (100 deg FOV): 8 around the horizon (pitch -15 deg, every 45 deg),
         4 looking down (pitch -55 deg), 1 straight up; neighbouring views overlap by about half
      -> Mask2Former (Swin-L, ADE20K panoptic, same 150 class ids as SegFormer) on every view:
         one mask + class + score per object ("thing") or region ("stuff")
      -> every view's segment map is projected back onto the equirectangular grid (nearest
         neighbour). A panorama pixel takes the segment of the view that sees it most centrally.
      -> cross-view matching: for every pair of views, inside the area BOTH views see, the IoU of
         every pair of furniture segments is computed from one contingency table; segments of the
         same furniture family with IoU >= MERGE_IOU are joined (union-find), so one physical
         object = one instance even if it is cut by view borders or crosses the seam
      -> instance class = pixel-weighted vote of its members; small instances are dropped
Outputs: class map (ADE20K ids, 255 = unknown), instance map (-1 = none), instance list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from . import equirect as eq

ROOT = Path(__file__).resolve().parents[1]
PANOPTIC_MODEL = ROOT / "checkpoints" / "mask2former_ade_panoptic"   # facebook/mask2former-swin-large-ade-panoptic
VIEWS = [(float(y), -15.0) for y in range(0, 360, 45)] + [(float(y), -55.0) for y in (0, 90, 180, 270)] + [(0.0, 90.0)]
FOV = 100.0
SIZE = 640
SCORE_MIN = 0.5            # panoptic segments below this confidence are discarded
MERGE_IOU = 0.5            # cross-view IoU (inside the shared area) to join two segments
MIN_SHARE = 0.0012         # instance must cover 0.12% of the panorama (the v2 threshold)
UNKNOWN = 255

# Furniture families: segments are only merged within a family, and evaluation counts a class as
# correct if the family agrees (ADE20K often confuses e.g. armchair / chair, cabinet / chest of drawers).
FAMILIES = {
    "bed": ["bed", "cradle"],
    "sofa": ["sofa"],
    "chair": ["chair", "swivel chair", "armchair", "seat", "stool", "bench", "ottoman"],
    "table": ["table", "coffee table", "desk", "pool table"],
    "storage": ["cabinet", "wardrobe", "chest of drawers", "shelf", "bookcase", "buffet"],
    "counter": ["counter", "countertop", "kitchen island", "sink", "stove", "oven", "dishwasher"],
    "bath": ["bathtub", "shower", "toilet"],
    "appliance": ["refrigerator", "washer"],
    "tv": ["television receiver"],
    "lamp": ["lamp"],
    "plant": ["plant", "pot", "flower"],
    "fireplace": ["fireplace"],
}


def family_of(id2label: dict) -> dict[int, str]:
    """ADE20K id -> furniture family name (ids of non-furniture classes are absent)."""
    name2fam = {n: f for f, names in FAMILIES.items() for n in names}
    out = {}
    for i, lab in id2label.items():
        for part in str(lab).split(";"):
            part = part.strip()
            if part in name2fam:
                out[int(i)] = name2fam[part]
                break
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_panoptic(device, path: Path = PANOPTIC_MODEL):
    """Mask2Former Swin-L ADE20K panoptic, fp16 (~1.2 GB VRAM)."""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(path), dtype=torch.float16).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(str(path))


@torch.inference_mode()
def panoptic_view(model, proc, img: np.ndarray, device, score_min: float = 0.3) -> tuple[np.ndarray, list[dict]]:
    """Panoptic segmentation of one RGB view: (segment id map, -1 = none; [{id, label_id, score}])."""
    inputs = proc(images=img, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    out = model(**inputs)
    res = proc.post_process_panoptic_segmentation(out, threshold=score_min, target_sizes=[img.shape[:2]],
                                              label_ids_to_fuse=set())[0]
    seg = res["segmentation"].cpu().numpy().astype(np.int32)
    info = [{"id": int(s["id"]), "label_id": int(s["label_id"]), "score": float(s["score"])} for s in res["segments_info"]]
    valid = np.isin(seg, [s["id"] for s in info])
    return np.where(valid, seg, -1), info


# --------------------------------------------------------------------------- #
# Geometry: panorama pixel -> view pixel
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def view_maps(w: int, h: int, views: tuple = tuple(VIEWS), fov: float = FOV, size: int = SIZE):
    """For every view: (u, v) float32 maps of the panorama grid into the view, and a centrality weight
    (cosine to the view axis; -1 where the view does not see the pixel)."""
    dirs = eq.pixel_dirs(w, h)
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    c = (size - 1) / 2
    out = []
    for yaw, pitch in views:
        dc = dirs @ eq.rotation(yaw, pitch)
        z = dc[..., 2]
        ok = z > 1e-3
        u = np.where(ok, dc[..., 0] / np.maximum(z, 1e-3) * f + c, -1)
        v = np.where(ok, dc[..., 1] / np.maximum(z, 1e-3) * f + c, -1)
        inside = ok & (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
        out.append((u.astype(np.float32), v.astype(np.float32), np.where(inside, z, -1.0).astype(np.float32)))
    return out


def run_views(model, proc, pano: np.ndarray, device, views=VIEWS, fov: float = FOV,
              size: int = SIZE) -> list[tuple[np.ndarray, list[dict]]]:
    """Panoptic segmentation of every perspective view of the panorama (raw, low score threshold)."""
    return [panoptic_view(model, proc, eq.e2p(pano, fov, y, p, size), device) for y, p in views]


# --------------------------------------------------------------------------- #
# Merging onto the panorama
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def merge_views(view_out: list, id2label: dict, w: int = 1024, h: int = 512, views=VIEWS,
                score_min: float = SCORE_MIN, merge_iou: float = MERGE_IOU, min_share: float = MIN_SHARE,
                fov: float = FOV, size: int = SIZE) -> dict:
    """Per-view panoptic results -> panorama class map, instance map and instance list."""
    fam = family_of(id2label)
    maps = view_maps(w, h, tuple(views), fov, size)
    segs = []                                       # global segment table
    proj = np.full((len(views), h, w), -1, np.int32)
    weight = np.stack([m[2] for m in maps])
    for k, ((seg, info), (u, v, wt)) in enumerate(zip(view_out, maps)):
        lut = {}
        for s in info:
            if s["score"] < score_min:
                continue
            lut[s["id"]] = len(segs)
            segs.append({"view": k, "label_id": s["label_id"], "score": s["score"],
                         "family": fam.get(s["label_id"])})
        g = np.full(seg.max() + 2, -1, np.int32)
        for sid, gi in lut.items():
            g[sid] = gi
        samp = cv2.remap(seg.astype(np.float32), u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=-1).astype(np.int32)
        proj[k] = np.where(wt > 0, g[np.clip(samp, -1, len(g) - 2)], -1)
        proj[k][samp < 0] = -1
    best = np.argmax(weight, 0)
    chosen = np.take_along_axis(proj, best[None], 0)[0]            # segment of the most central view
    labels = np.array([s["label_id"] for s in segs] + [UNKNOWN], np.int32)
    class_map = labels[chosen]                                      # chosen = -1 -> UNKNOWN

    # cross-view matching of furniture segments (half resolution is enough)
    uf = _UF(len(segs))
    sub = proj[:, ::2, ::2]
    is_furn = np.array([s["family"] is not None for s in segs] + [False])
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            both = (weight[a, ::2, ::2] > 0) & (weight[b, ::2, ::2] > 0)
            if both.sum() < 200:
                continue
            sa, sb = sub[a][both], sub[b][both]
            fa, fb = is_furn[sa], is_furn[sb]
            n = len(segs) + 1
            ia, ib = np.where(sa < 0, n - 1, sa), np.where(sb < 0, n - 1, sb)
            area_a = np.bincount(ia[fa], minlength=n)
            area_b = np.bincount(ib[fb], minlength=n)
            both_f = fa & fb
            pairs, cnt = np.unique(ia[both_f].astype(np.int64) * n + ib[both_f], return_counts=True)
            for code, inter in zip(pairs, cnt):
                i, j = int(code // n), int(code % n)
                if segs[i]["family"] != segs[j]["family"]:
                    continue
                iou = inter / max(area_a[i] + area_b[j] - inter, 1)
                if iou >= merge_iou:
                    uf.union(i, j)

    root = np.array([uf.find(i) for i in range(len(segs))] + [-1], np.int64)
    inst_raw = np.where(is_furn[chosen], root[chosen], -1)
    min_px = int(min_share * w * h)
    inst_map = np.full((h, w), -1, np.int32)
    out = []
    for r in np.unique(inst_raw[inst_raw >= 0]):
        m = inst_raw == r
        if m.sum() < min_px:
            continue
        members = [i for i in range(len(segs)) if root[i] == r]
        votes = np.bincount(chosen[m], minlength=len(segs) + 1)[:len(segs)]
        lab = segs[int(np.argmax(votes))]["label_id"]
        idx = len(out)
        inst_map[m] = idx
        class_map[m] = lab
        out.append({"id": idx, "label_id": lab, "label": str(id2label[lab]).split(";")[0].strip(), "family": fam[lab],
                    "score": float(max(segs[i]["score"] for i in members)), "views": len({segs[i]["view"] for i in members}),
                    "pixels": int(m.sum())})
    # furniture pixels of dropped (tiny) instances keep their class but get no instance
    return {"class_map": class_map, "inst_map": inst_map, "instances": out}


def fill_unknown(class_map: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Pixels without a segment: take the fallback class map if given, else the nearest known class."""
    unk = class_map == UNKNOWN
    if not unk.any():
        return class_map
    if fallback is not None:
        return np.where(unk, fallback, class_map)
    _, idx = cv2.distanceTransformWithLabels(unk.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~unk)
    lut = np.zeros(idx.max() + 1, np.int32)
    lut[idx[~unk]] = class_map[ys, xs]
    return np.where(unk, lut[idx], class_map)


def segment_instances(model, proc, pano: np.ndarray, device, id2label: dict, w: int = 1024, h: int = 512, **kw) -> dict:
    """Whole milestone-1 pipeline on one panorama (resized to w x h for the maps)."""
    res = merge_views(run_views(model, proc, pano, device), id2label, w, h, **kw)
    res["class_map"] = fill_unknown(res["class_map"])
    return res


def hybrid_class_map(stuff_map: np.ndarray, v3: dict) -> np.ndarray:
    """v3 class map used downstream: room surfaces and openings from the SegFormer map (better on
    ceiling / floor: Mask2Former calls the ceiling of the upward view "floor"), furniture classes from
    the panoptic instances (milestone 1 evaluation, results/tables/inst_summary.csv)."""
    out = stuff_map.copy()
    m = v3["inst_map"] >= 0
    out[m] = v3["class_map"][m]
    return out


# --------------------------------------------------------------------------- #
# v2 baseline (for the comparison): connected components of the semantic map
# --------------------------------------------------------------------------- #
def v2_instances(class_map: np.ndarray, id2label: dict, min_share: float = MIN_SHARE) -> dict:
    """The v2 instance split (floorplan.extract_furniture): per furniture class, a 5x5 opening, then
    8-connected components of at least 0.12% of the image. No wrap-around at the seam."""
    from .floorplan import FURNITURE, class_ids, instances
    fam = family_of(id2label)
    inst_map = np.full(class_map.shape, -1, np.int32)
    out = []
    min_px = int(min_share * class_map.size)
    for name in FURNITURE:
        ids = class_ids(id2label, [name])
        if not ids:
            continue
        mask = np.isin(class_map, ids)
        if mask.sum() < min_px:
            continue
        for m in instances(mask, min_px):
            idx = len(out)
            inst_map[m & (inst_map < 0)] = idx
            out.append({"id": idx, "label_id": ids[0], "label": name, "family": fam.get(ids[0], name),
                        "pixels": int(m.sum())})
    return {"class_map": class_map, "inst_map": inst_map, "instances": out}
