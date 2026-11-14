"""PLAN_V4 milestone 4: real furniture shapes (TripoSR) against the v3 blocks.

Rooms: the 40-room bench (dev = layout seeds 1-4, held-out = 5-8) + the six demo rooms. Per room (resumable,
outputs/objects_eval/<room>/): v3 room (box, solids, instances incl. leftovers), the 4096 panorama (demo: the
milestone-2 selection; bench: Real-ESRGAN views, cached), and for EVERY solid a TripoSR object (cached
obj_<instance>.npz) oriented by render-and-compare, with its silhouette IoU and the v3 block's IoU.
Choices on the DEVELOPMENT rooms only:
    IOU_MIN        threshold t in {0, 0.3, ..., 0.7}: the object is used if its IoU >= t, else the block;
                   t maximises the mean silhouette IoU of the final mix (ties: the higher t)
    BLOCK_CLASSES  classes with >= 3 dev instances whose object IoU is on average > 0.05 below the block's
Reported (held-out bench + demo rooms):
    silhouette IoU from the panorama centre       v3 (all blocks) vs v4 (mix)
    full-room re-render from the centre vs the panorama (1024 x 512): PSNR, SSIM, LPIPS - same v4 shell for both,
        so only the furniture representation differs; v4 also shows uncertain instances as RELIEF (their
        observed surface) instead of nothing
    footprint IoU / size / orientation vs the layout boxes (unchanged by construction: objects are scaled into
        the v3 solid's box) and FACING: for layout boxes within 30 cm of a wall the true front faces away from
        it; v3's facing = the plan rule (back = side nearest a wall), v4's = the generated object's front
    share of instances as object / block / relief; recall@0.25 with the relief tier
    plan check: the plan symbol's facing vs the placed object's front (share within 45 deg)
    seconds and peak VRAM per room
Outputs: results/tables/objects_per_instance.csv, objects_per_room.csv, objects_dev_threshold.csv,
objects_classes_dev.csv, objects_selection.json, objects_summary.csv; figures objects_examples.png,
objects_dollhouse.png
Usage: python scripts/eval_objects.py [--limit N] [--stage gen|report|all]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import equirect as eq  # noqa: E402
from src import furniture as fu  # noqa: E402
from src import generate_pano as gp  # noqa: E402
from src import objects3d as o3  # noqa: E402
from src import pano_refine as pr  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import pipeline_v3 as p3v  # noqa: E402
from src import room_box as rb  # noqa: E402
from src import room_v4 as r4  # noqa: E402
import eval_furniture as ef  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402

OUT = ROOT / "outputs" / "objects_eval"
TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
DEVICE = torch.device("cuda")
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]


def room_list(limit=None) -> pd.DataFrame:
    rows = []
    for r in mb.rooms().itertuples():
        d = mb.OUT / r.id
        rows.append({"id": r.id, "set": "dev" if r.split == "dev" else "test", "d": d, "ana": d, "dist": d / "dist.npy",
                     "tex": d / "texture.png", "hires": None, "bench": True})
    for prompt, seed in pl.DEMO_PROMPTS:
        d = ROOT / "outputs/room360" / pl.slug(prompt, seed)
        rows.append({"id": "demo_" + pl.slug(prompt, seed), "set": "demo", "d": d, "ana": d / "v3",
                     "dist": d / "distance.npy", "tex": d / "texture.png", "hires": d / "v4/pano_4096_detail.png",
                     "bench": False})
    df = pd.DataFrame(rows)
    return df.groupby("set").head(limit) if limit else df


def load_room(r, id2label, net):
    dist = np.load(r.dist)
    ana = {"class_map": cv2.imread(str(r.ana / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32),
           "inst_map": cv2.imread(str(r.ana / "inst_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1,
           "instances": json.loads((r.ana / "inst_v3.json").read_text())}
    tex = mb.rgb(r.tex)
    od = OUT / r.id
    od.mkdir(parents=True, exist_ok=True)
    if r.hires is not None:
        hires = mb.rgb(r.hires)
    else:
        hp = od / "pano_4096.jpg"
        if not hp.exists():
            pano, _ = pr.refine(tex, net)
            cv2.imwrite(str(hp), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        hires = mb.rgb(hp)
    res = p3v.build(dist, tex, ana, id2label)
    inst_map, insts = fu.add_leftovers(ana["inst_map"], ana["instances"], ana["class_map"], id2label)
    return {"dist": dist, "ana": ana, "hires": hires, "res": res, "inst_map": inst_map, "insts": insts, "od": od}


# --------------------------------------------------------------------------- #
# Facing
# --------------------------------------------------------------------------- #
def facing_rule_B(s: dict, box: dict) -> np.ndarray:
    """v3 / plan rule: the back of the symbol is the side nearest a wall -> front = into the room (frame B xz)."""
    c = fu._corners(s)
    (x0, x1), (z0, z1) = box["x"], box["z"]
    best = None
    for p, q in ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])):
        mid = (p + q) / 2
        n = np.array([-(q - p)[1], (q - p)[0]])
        n /= max(np.linalg.norm(n), 1e-9)
        if n @ (mid - s["c"]) < 0:
            n = -n                                                     # outward normal of this side
        for wn, val, ax in (((1, 0), x1, 0), ((-1, 0), x0, 0), ((0, 1), z1, 1), ((0, -1), z0, 1)):
            if n @ np.array(wn) > 0.9:
                d = abs(mid[ax] - val)
                if best is None or d < best[0]:
                    best = (d, -n)
    return best[1] if best else np.array([0.0, 1.0])


def b_to_cam_xz(v_xz: np.ndarray, box: dict) -> np.ndarray:
    v = np.array([v_xz[0], 0.0, v_xz[1]]) @ box["Q"].T @ box["R"]
    return v[[0, 2]] / max(np.linalg.norm(v[[0, 2]]), 1e-9)


def gt_facing(b: dict, L: dict, near: float = 0.3):
    """Front of a layout box that stands within `near` of a wall: away from the nearest wall (camera frame xz)."""
    gaps = [(b["x0"], (1.0, 0.0)), (L["W"] - b["x1"], (-1.0, 0.0)), (b["z0"], (0.0, 1.0)), (L["D"] - b["z1"], (0.0, -1.0))]
    g, d = min(gaps, key=lambda t: t[0])
    return np.array(d) if g < near else None


def angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1, 1))))


# --------------------------------------------------------------------------- #
# Full-room render from the centre (shell + textured parts + vertex-coloured objects)
# --------------------------------------------------------------------------- #
def render_equirect(tex_mesh: dict, objs: list[dict], box: dict, W: int = 1024, H: int = 512) -> np.ndarray:
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    g0 = sc.add_triangles(o3d.core.Tensor(tex_mesh["vertices"].astype(np.float32)),
                          o3d.core.Tensor(tex_mesh["faces"].astype(np.uint32)))
    if objs:
        V = np.concatenate([o["VB"] @ box["Q"].T for o in objs]).astype(np.float32)
        off = np.cumsum([0] + [len(o["VB"]) for o in objs])[:-1]
        F = np.concatenate([o["faces"] + k for o, k in zip(objs, off)]).astype(np.uint32)
        C = np.concatenate([o["colors"] for o in objs]).astype(np.float32)
        sc.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(F))
    d = (eq.pixel_dirs(W, H) @ box["R"].T).reshape(-1, 3).astype(np.float32)          # camera -> frame A
    ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(d), d], 1)))
    hit = np.isfinite(ans["t_hit"].numpy())
    gid, prim, bc = ans["geometry_ids"].numpy(), ans["primitive_ids"].numpy(), ans["primitive_uvs"].numpy()
    img = np.zeros((len(d), 3), np.float32)
    w = np.stack([1 - bc[:, 0] - bc[:, 1], bc[:, 0], bc[:, 1]], 1)
    t = hit & (gid == g0)
    f = tex_mesh["faces"][prim[t]]
    uv = (tex_mesh["uvs"][f] * w[t][:, :, None]).sum(1)
    tx = tex_mesh["texture"]
    img[t] = r4.bilinear(tx, uv[:, 0] * tx.shape[1] - 0.5, uv[:, 1] * tx.shape[0] - 0.5)
    if objs:
        o = hit & (gid != g0)
        img[o] = (C[F[prim[o]]] * w[o][:, :, None]).sum(1)
    return img.reshape(H, W, 3).clip(0, 255).astype(np.uint8)


def relief_mask(R: dict, used_instances: set, min_px: int = 150) -> np.ndarray:
    """Furniture instances that became neither a solid nor an object (rejected / uncertain): kept as RELIEF."""
    inst = R["inst_map"]
    keep = np.zeros(inst.shape, bool)
    ids, cnt = np.unique(inst[inst >= 0], return_counts=True)
    for i, n in zip(ids, cnt):
        if int(i) not in used_instances and n >= min_px:
            keep |= inst == i
    return keep & p3v.in_front_of_shell(R["dist"], R["res"]["box"])


def models_of(R: dict, objs: list[dict], blocks: list[dict], with_relief: bool, with_ceiling: bool = True):
    box, dist, hires = R["res"]["box"], R["dist"], R["hires"]
    shell = rb.build_shell(box, hires, dist, texel_m=0.01, max_tex=2048)
    parts = [p for p in shell["parts"] if with_ceiling or p["kind"] != "ceiling"]
    if blocks:
        bp = fu.solids_part(blocks, box, dist, R["inst_map"], hires)
        if len(bp["faces"]):
            parts.append(bp)
    n_relief = 0
    if with_relief:
        used = {int(s.get("instance", -1)) for s in R["res"]["solids"]}
        km = relief_mask(R, used)
        if km.any():
            rp = rb.observed_part(dist, hires, box["R"], km, stride=2)
            if len(rp["faces"]):
                rp["kind"] = "relief"
                parts.append(rp)
                n_relief = len(np.unique(R["inst_map"][km]))
    return r4.combine_packed(parts, 8192), n_relief


# --------------------------------------------------------------------------- #
def stage_gen(df: pd.DataFrame, id2label):
    """Objects for every solid of every room (cached) + per-instance rows."""
    net = gp.load_esrgan(DEVICE)
    model = None
    for r in tqdm(list(df.itertuples()), desc="rooms (objects)"):
        od = OUT / r.id
        if (od / "instances.csv").exists():
            continue
        R = load_room(r, id2label, net)
        if model is None:
            model = o3.load_triposr(DEVICE)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = o3.build_objects(model, DEVICE, R["hires"], R["dist"], R["inst_map"], R["res"]["solids"], R["res"]["box"],
                               iou_min=0.0, block_classes=set(), cache_dir=od)
        meta = {"seconds_objects": time.perf_counter() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                "solids": len(R["res"]["solids"])}
        rows = pd.DataFrame(out["rows"], columns=None if out["rows"] else
                            ["label", "instance", "rank", "kind", "iou", "iou_block"])
        rows.insert(0, "id", r.id)
        rows.insert(1, "set", r.set)
        # facing of each generated object (frame B) + plan-rule facing
        fr = {int(o["solid"]["instance"]): o["front_B"] for o in out["objects"]}
        rows["front_x"] = [fr[i][0] if i in fr else np.nan for i in rows["instance"]]
        rows["front_z"] = [fr[i][2] if i in fr else np.nan for i in rows["instance"]]
        rows.to_csv(od / "instances.csv", index=False)
        (od / "meta.json").write_text(json.dumps(meta))
    del model
    torch.cuda.empty_cache()


def choose(per: pd.DataFrame) -> dict:
    dev = per[(per["set"] == "dev") & per["iou"].notna()]
    th = []
    for t in THRESHOLDS:
        use = dev["iou"] >= t
        th.append({"threshold": t, "mean_final_iou": float(np.where(use, dev["iou"], dev["iou_block"]).mean()),
                   "object_share": float(use.mean())})
    th = pd.DataFrame(th)
    th.to_csv(TAB / "objects_dev_threshold.csv", index=False)
    best_t = float(th.sort_values(["mean_final_iou", "threshold"], ascending=False).iloc[0]["threshold"])
    cl = dev.assign(diff=dev["iou"] - dev["iou_block"]).groupby("label").agg(
        n=("diff", "size"), obj_iou=("iou", "mean"), block_iou=("iou_block", "mean"), diff=("diff", "mean")).reset_index()
    cl.to_csv(TAB / "objects_classes_dev.csv", index=False)
    block_classes = sorted(cl[(cl["n"] >= 3) & (cl["diff"] < -0.05)]["label"])
    return {"iou_min": best_t, "block_classes": block_classes,
            "rule": "threshold maximising the dev mean final silhouette IoU (ties: higher); block classes: >= 3 dev "
                    "instances and object IoU > 0.05 below the block on average"}


def stage_report(df: pd.DataFrame, id2label, sel: dict):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
    lpips = lambda x, y: float(lp(*[torch.from_numpy(i).permute(2, 0, 1)[None].float().div(255) for i in (x, y)]))  # noqa: E731
    rows, figs = [], []
    for r in tqdm([r for r in df.itertuples() if r.set != "dev"], desc="rooms (report)"):
        od = OUT / r.id
        R = load_room(r, id2label, None)
        box = R["res"]["box"]
        inst = pd.read_csv(od / "instances.csv")
        meta = json.loads((od / "meta.json").read_text())
        # objects from cache, re-oriented exactly as in generation (deterministic)
        solids = {int(s.get("instance", -1)): s for s in R["res"]["solids"]}
        objs, blocks = [], []
        for _, row in inst.iterrows():
            s = solids.get(int(row["instance"]))
            if s is None:
                continue
            ok = row["kind"] == "object" and row["label"] not in sel["block_classes"] and row["iou"] >= sel["iou_min"]
            if ok:
                z = np.load(od / f"obj_{int(row['instance'])}.npz")
                obj = o3.simplify({"vertices": z["v"], "faces": z["f"], "colors": z["c"]})
                fB = np.array([row["front_x"], 0.0, row["front_z"]])
                VB = o3.place(obj, s, box, fB)
                cols, _ = o3.colour_from_pano(VB, obj["faces"], obj["colors"], box, R["hires"], R["dist"])
                objs.append({"solid": s, "VB": VB, "faces": obj["faces"], "colors": cols, "front_B": fB})
            else:
                blocks.append(s)
        pano_ref = cv2.resize(R["hires"], (1024, 512), interpolation=cv2.INTER_AREA)
        m3, _ = models_of(R, [], list(solids.values()), with_relief=False)
        m4, n_relief = models_of(R, objs, blocks, with_relief=True)
        m4n, _ = models_of(R, objs, blocks, with_relief=False)
        i3, i4, i4n = render_equirect(m3, [], box), render_equirect(m4, objs, box), render_equirect(m4n, objs, box)
        g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  # noqa: E731
        row = {"id": r.id, "set": r.set, "n_solids": len(solids), "n_object": len(objs), "n_block": len(blocks),
               "n_relief": n_relief, "psnr_v3": cv2.PSNR(i3, pano_ref), "psnr_v4": cv2.PSNR(i4, pano_ref),
               "ssim_v3": ssim(g(i3), g(pano_ref)), "ssim_v4": ssim(g(i4), g(pano_ref)),
               "lpips_v3": lpips(i3, pano_ref), "lpips_v4": lpips(i4, pano_ref),
               "psnr_v4_no_relief": cv2.PSNR(i4n, pano_ref), "ssim_v4_no_relief": ssim(g(i4n), g(pano_ref)),
               "lpips_v4_no_relief": lpips(i4n, pano_ref),
               "sil_iou_v3": float(inst["iou_block"].mean()),
               "sil_iou_v4": float(np.mean([row_["iou"] if (row_["kind"] == "object" and row_["label"] not in sel["block_classes"]
                                                            and row_["iou"] >= sel["iou_min"]) else row_["iou_block"]
                                            for _, row_ in inst.iterrows() if not np.isnan(row_.get("iou_block", np.nan))])),
               "seconds_objects": meta["seconds_objects"], "peak_vram_gb": meta["peak_vram_gb"]}
        # plan agreement: symbol facing (rule) vs object front
        agree = [angle(facing_rule_B(o["solid"], box), o["front_B"][[0, 2]]) <= 45 for o in objs]
        row["plan_object_agree"] = float(np.mean(agree)) if agree else np.nan
        if r.set == "test":
            gt = mb.load_gt(r.d)
            gtr, gboxes = ef.gt_rects(gt)
            sol_l = list(solids.values())
            pred = fu.footprints_cam(sol_l, box)
            o_, mrows = ef.match(gtr, pred)
            row.update({"n_gt": o_["n_gt"], "tp25": o_["tp25"], "footprint_iou": np.mean([m["iou"] for m in mrows]) if mrows else np.nan})
            # relief footprints count for recall
            used = {int(s.get("instance", -1)) for s in sol_l}
            P = (eq.pixel_dirs(R["dist"].shape[1], R["dist"].shape[0]) * R["dist"][..., None])     # camera frame
            rel = []
            for i in np.unique(R["inst_map"][R["inst_map"] >= 0]):
                if int(i) in used or (R["inst_map"] == i).sum() < 150:
                    continue
                xz = P[R["inst_map"] == i][:, [0, 2]].astype(np.float32)
                rel.append(cv2.boxPoints(cv2.minAreaRect(xz)))
            o2, _ = ef.match(gtr, pred + rel)
            row["tp25_with_relief"] = o2["tp25"]
            # facing vs GT (matched pairs, wall-adjacent GT boxes)
            from scipy.optimize import linear_sum_assignment
            if gtr and pred:
                M = np.array([[ef.iou(a, b) for b in pred] for a in gtr])
                ri, ci = linear_sum_assignment(-M)
                fv3, fv4 = [], []
                front = {id(o["solid"]): o["front_B"][[0, 2]] for o in objs}
                for i, j in zip(ri, ci):
                    if M[i, j] < 0.1:
                        continue
                    gf = gt_facing(gboxes[i], gt["layout"])
                    if gf is None:
                        continue
                    s = sol_l[j]
                    fv3.append(angle(b_to_cam_xz(facing_rule_B(s, box), box), gf) <= 45)
                    if id(s) in front:
                        fv4.append((angle(b_to_cam_xz(front[id(s)], box), gf) <= 45,
                                    angle(b_to_cam_xz(facing_rule_B(s, box), box), gf) <= 45))
                row.update({"facing_n_v3": len(fv3), "facing_acc_v3": float(np.mean(fv3)) if fv3 else np.nan,
                            "facing_n_obj": len(fv4),
                            "facing_acc_obj": float(np.mean([a for a, _ in fv4])) if fv4 else np.nan,
                            "facing_acc_rule_same": float(np.mean([b for _, b in fv4])) if fv4 else np.nan})
        rows.append(row)
        if r.set == "demo" or len(figs) < 3:
            d4, _ = models_of(R, objs, blocks, with_relief=True, with_ceiling=False)
            d3, _ = models_of(R, [], list(solids.values()), with_relief=False, with_ceiling=False)
            figs.append((r.id, dollhouse_render(d3, [], box), dollhouse_render(d4, objs, box),
                         [o3.prepare_input(o3.instance_view(R["hires"], R["inst_map"], R["dist"], int(o["solid"]["instance"])))[0]
                          for o in objs[:3]]))
        pd.DataFrame(rows).to_csv(TAB / "objects_per_room.csv", index=False)
    return pd.DataFrame(rows), figs


def dollhouse_render(mesh: dict, objs: list[dict], box: dict, size=(640, 480)) -> np.ndarray:
    """Perspective view from outside and above (like the dollhouse viewer), ray cast."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    # back-face culling of the shell: drop shell faces whose normal faces the camera's opposite side
    (x0, x1), (z0, z1) = box["x"], box["z"]
    ctr = np.array([(x0 + x1) / 2, box["floor_y"] - 0.9, (z0 + z1) / 2])
    span = max(x1 - x0, z1 - z0)
    eye = ctr + np.array([0.55, -1.0, -0.75]) / np.linalg.norm([0.55, -1.0, -0.75]) * span * 1.35
    eyeA = eye @ box["Q"].T
    V = mesh["vertices"]
    F = mesh["faces"]
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    keep = np.einsum("ij,ij->i", n, eyeA - V[F[:, 0]]) > 0
    kind = mesh["face_kind"]
    keep |= ~np.isin(kind, ["wall+x", "wall-x", "wall+z", "wall-z", "floor", "ceiling"])
    g0 = sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F[keep].astype(np.uint32)))
    Fk = F[keep]
    if objs:
        OV = np.concatenate([o["VB"] @ box["Q"].T for o in objs]).astype(np.float32)
        off = np.cumsum([0] + [len(o["VB"]) for o in objs])[:-1]
        OF = np.concatenate([o["faces"] + k for o, k in zip(objs, off)]).astype(np.uint32)
        OC = np.concatenate([o["colors"] for o in objs]).astype(np.float32)
        sc.add_triangles(o3d.core.Tensor(OV), o3d.core.Tensor(OF))
    w, h = size
    fwd = (ctr - eye) / np.linalg.norm(ctr - eye)
    right = np.cross(fwd, [0, -1.0, 0])           # frame B: Y points down, so world up is -Y
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 0.5 * w / np.tan(np.radians(25))
    u, v = np.meshgrid(np.arange(w) - w / 2, np.arange(h) - h / 2)
    dB = fwd[None, None] * f + u[..., None] * right[None, None] - v[..., None] * up[None, None]
    dA = (dB / np.linalg.norm(dB, axis=-1, keepdims=True)).reshape(-1, 3) @ box["Q"].T
    ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eyeA, dA.shape), dA], 1).astype(np.float32)))
    hit = np.isfinite(ans["t_hit"].numpy())
    gid, prim, bc = ans["geometry_ids"].numpy(), ans["primitive_ids"].numpy(), ans["primitive_uvs"].numpy()
    img = np.full((len(dA), 3), 233, np.float32)
    wg = np.stack([1 - bc[:, 0] - bc[:, 1], bc[:, 0], bc[:, 1]], 1)
    t = hit & (gid == g0)
    fa = Fk[prim[t]]
    uv = (mesh["uvs"][fa] * wg[t][:, :, None]).sum(1)
    tx = mesh["texture"]
    img[t] = r4.bilinear(tx, uv[:, 0] * tx.shape[1] - 0.5, uv[:, 1] * tx.shape[0] - 0.5)
    if objs:
        o = hit & (gid != g0)
        img[o] = (OC[OF[prim[o]]] * wg[o][:, :, None]).sum(1)
    return img.reshape(h, w, 3).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    id2label = erb.id2label_ade()
    df = room_list(a.limit)
    if a.stage in ("gen", "all"):
        stage_gen(df, id2label)
    per = pd.concat([x for x in (pd.read_csv(OUT / i / "instances.csv") for i in df["id"]
                                 if (OUT / i / "instances.csv").exists()) if len(x)])
    per.to_csv(TAB / "objects_per_instance.csv", index=False)
    sel = choose(per)
    (TAB / "objects_selection.json").write_text(json.dumps(sel, indent=1))
    print(sel)
    if a.stage in ("report", "all"):
        rep, figs = stage_report(df, id2label, sel)
        cols = [c for c in rep.columns if c not in ("id", "set")]
        summ = rep.groupby("set")[cols].mean().reset_index()
        tot = rep.groupby("set")[["n_solids", "n_object", "n_block", "n_relief"]].sum().add_prefix("total_").reset_index()
        summ = summ.merge(tot, on="set")
        summ.to_csv(TAB / "objects_summary.csv", index=False)
        print(summ.round(3).T.to_string())
        k = min(len(figs), 6)
        fig, axes = plt.subplots(k, 5, figsize=(26, 4.4 * k), dpi=60, squeeze=False)
        for i, (rid, a3, a4, ins) in enumerate(figs[:k]):
            axes[i, 0].imshow(a3); axes[i, 0].set_title(f"v3 dollhouse (blocks) - {rid}", fontsize=11)
            axes[i, 1].imshow(a4); axes[i, 1].set_title("v4 dollhouse (generated objects, blocks, relief)", fontsize=11)
            for j in range(3):
                if j < len(ins):
                    axes[i, 2 + j].imshow(ins[j])
                    axes[i, 2 + j].set_title(f"TripoSR input {j + 1}", fontsize=11)
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG / "objects_dollhouse.png", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
