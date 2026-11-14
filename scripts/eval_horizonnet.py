"""PLAN_V3 milestone 2, step 2: our box fit vs. a pretrained panorama layout estimator (HorizonNet).

HorizonNet (Sun et al., CVPR 2019; code github.com/sunset1995/HorizonNet, weights
huggingface.co/sunset1995/HorizonNet, resnet50_rnn__mp3d.pth = trained on Matterport3D homes) predicts the
floor / ceiling boundary and the wall-wall corners of a panorama; with force_cuboid it returns a 4-corner room.
Its layout is defined up to scale (camera height). For a fair comparison it is scaled with the SAME camera
height that our pipeline estimates from DAP (the gravity frame of room_box.fit_box); a second, shape-only
score uses the true camera height (1.4 m) for both.

Metrics on all 40 bench rooms (dev and held-out reported separately; nothing is tuned here): area error,
side error, corner error (as in scripts/eval_room_box.py).
Outputs: results/tables/box_horizonnet.csv, box_horizonnet_per_room.csv
Usage: python scripts/eval_horizonnet.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
HN = ROOT / "checkpoints" / "horizonnet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import room_box as rb  # noqa: E402
import eval_room_box as erb  # noqa: E402
import make_bench as mb  # noqa: E402


def _sklearn_shim():
    """HorizonNet's post-processing imports sklearn only for a 1-component PCA (direction of a wall's
    floor-boundary points); this numpy version (first right-singular vector) avoids the dependency."""
    import types

    class PCA:
        def __init__(self, n_components=1):
            self.n = n_components

        def fit(self, X):
            X = np.asarray(X, float)
            self.components_ = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:self.n]
            return self

    from importlib.machinery import ModuleSpec
    for name in ("sklearn", "sklearn.decomposition"):
        mod = types.ModuleType(name)
        mod.__spec__ = ModuleSpec(name, None)
        sys.modules[name] = mod
    sys.modules["sklearn.decomposition"].PCA = PCA


def load_horizonnet(device):
    _sklearn_shim()
    sys.path.insert(0, str(HN))
    from model import HorizonNet
    state = torch.load(HN / "ckpt" / "resnet50_rnn__mp3d.pth", map_location="cpu", weights_only=False)
    net = HorizonNet(**state["kwargs"])
    net.load_state_dict(state["state_dict"])
    return net.to(device).eval()


@torch.inference_mode()
def floor_corners(net, pano: np.ndarray, device, cam_h: float) -> np.ndarray:
    """4 floor corners (x right, z forward; camera frame) of HorizonNet's cuboid layout."""
    import inference as hn_inf
    x = torch.from_numpy(cv2.resize(pano, (1024, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None]
    cor_id, _, _, _ = hn_inf.inference(net, x, device, flip=False, rotate=[], force_cuboid=True)
    fl = cor_id[1::2]                                            # (x, y) normalised, floor corners
    lon = 2 * np.pi * fl[:, 0] - np.pi
    lat = np.pi / 2 - np.pi * fl[:, 1]                           # negative below the horizon
    r = cam_h / np.tan(np.maximum(-lat, 1e-3))
    c = np.stack([r * np.sin(lon), r * np.cos(lon)], 1)
    ang = np.arctan2(c[:, 1], c[:, 0])
    return c[np.argsort(ang)]


def main():
    device = torch.device("cuda")
    id2label = erb.id2label_ade()
    net = load_horizonnet(device)
    rows = []
    for r in tqdm(list(mb.rooms().itertuples()), desc="HorizonNet", unit="room"):
        d = mb.OUT / r.id
        g = mb.load_gt(d)
        L, G = g["layout"], erb.gt_box(g["layout"])
        dist = np.load(d / "dist.npy")
        cm3 = cv2.imread(str(d / "class_v3.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        box = rb.fit_box(dist, cm3, id2label)
        cam_h = box["camera_height"]
        pano = mb.rgb(d / "pano.png")
        ours = (np.c_[rb.box_corners_A(box)[:, 0], np.zeros(4), rb.box_corners_A(box)[:, 1]] @ box["R"])[:, [0, 2]]
        hn = floor_corners(net, pano, device, cam_h)
        for name, c, s in (("v3 box fit", ours, 1.4 / cam_h), ("HorizonNet (MP3D)", hn, 1.4 / cam_h)):
            m = erb.box_metrics(c, 0, G, L)
            ms = erb.box_metrics(c * s, 0, G, L)
            rows.append({"id": r.id, "split": r.split, "method": name, "area_rel_err": m["area_rel_err"],
                         "side_rel_err": m["side_rel_err"], "corner_err_m": m["corner_err_m"],
                         "area_rel_err_true_cam": ms["area_rel_err"], "corner_err_m_true_cam": ms["corner_err_m"]})
    df = pd.DataFrame(rows)
    summ = df.groupby(["split", "method"]).agg(
        rooms=("id", "size"), area_abs_rel_err=("area_rel_err", lambda s: s.abs().mean()),
        area_rel_err_mean=("area_rel_err", "mean"), side_rel_err=("side_rel_err", "mean"),
        corner_err_m=("corner_err_m", "mean"), corner_err_median_m=("corner_err_m", "median"),
        area_abs_rel_err_true_cam=("area_rel_err_true_cam", lambda s: s.abs().mean()),
        corner_err_m_true_cam=("corner_err_m_true_cam", "mean")).reset_index()
    T = ROOT / "results/tables"
    df.to_csv(T / "box_horizonnet_per_room.csv", index=False, float_format="%.4f")
    summ.to_csv(T / "box_horizonnet.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
