"""Depth estimation (Depth Anything, MiDaS baseline, metric Depth Anything V2) and depth-map refinement.

Two kinds of model:
  relative ("depth_anything", "midas"): return RELATIVE inverse depth ("disparity": large = near),
      correct only up to an unknown scale and shift. Evaluation aligns it to ground truth.
  metric ("depth_anything_metric"): returns depth in METRES directly. Used by the 3D stage, where
      an unknown shift would bend the geometry (see reconstruct_3d.disparity_to_depth).

Refinement (refine_depth) works on the predicted map itself, in four steps:
    1. median filter         - removes isolated spikes (salt-and-pepper outliers)
    2. bilateral filter      - smooths depth noise on surfaces, keeps depth steps
    3. edge-aware smoothing  - extra smoothing only AWAY from Canny edges of the RGB image,
                               so walls/floors get flat while object boundaries stay sharp
    4. hole filling          - morphological closing fills small pits / invalid pixels
"""

import cv2
import numpy as np
import torch

from .analysis import canny_edges
from .preprocessing import clahe_lab, denoise, preprocess_for_depth, preprocess_gated

DEPTH_MODELS = {
    "depth_anything": "LiheYoung/depth-anything-base-hf",   # main model
    "midas": "Intel/dpt-hybrid-midas",                      # baseline
    "depth_anything_metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",   # metres, 3D stage
}
METRIC_MODELS = {"depth_anything_metric"}     # these return depth in metres, the others disparity
DEFAULT_INPUT = "gated"           # preprocessing used by the pipeline: noise-gated denoising (see prepare_input)
DEPTH_MODEL_LABELS = {"depth_anything": "Depth Anything (ViT-B)", "midas": "MiDaS (DPT-Hybrid)",
                      "depth_anything_metric": "Depth Anything V2 Metric Indoor (ViT-B)"}

# Refinement parameters (the depth map is normalised to [0, 1] first, so they are scale-free).
REF_MEDIAN = 5            # median window (px)
REF_BILATERAL = (9, 0.03, 5.0)   # diameter, range sigma (3% of the depth range), spatial sigma (px)
REF_EDGE_RADIUS = 8       # px: within this distance of a Canny edge the depth is left sharp
REF_HOLE_KERNEL = 7       # px: closing kernel = largest hole that gets filled
REF_HOLE_THRESH = 0.03    # a pit must be deeper than 3% of the depth range to count as a hole


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_depth_model(name: str, device):
    """Load "depth_anything" or "midas" in fp16. Returns (model, processor).

    Only one depth model should be on the GPU at a time: `del model` and call
    evaluate.release_memory() before loading the next one.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    repo = DEPTH_MODELS[name]
    model = AutoModelForDepthEstimation.from_pretrained(repo, dtype=torch.float16).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(repo)


@torch.inference_mode()
def predict_disparity(model, processor, img: np.ndarray, device) -> np.ndarray:
    """Raw network output resized to the input (float32 HxW; input = RGB uint8).

    Relative models: inverse depth (disparity). Metric models: depth in metres (see predict_depth).
    """
    inp = processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.float16)
    pred = model(pixel_values=inp).predicted_depth              # 1 x h x w at the model resolution
    pred = torch.nn.functional.interpolate(pred[:, None].float(), size=img.shape[:2], mode="bilinear",
                                           align_corners=False)
    return pred[0, 0].cpu().numpy()


def predict_depth(model, processor, img: np.ndarray, device) -> np.ndarray:
    """Depth in metres from a METRIC model (float32 HxW)."""
    return np.maximum(predict_disparity(model, processor, img, device), 1e-3)


def prepare_input(img: np.ndarray, variant: str) -> np.ndarray:
    """Input preprocessing variants used in the ablation.

    "raw" | "denoise" | "clahe" | "full" (noise-aware pipeline, noise_aware=True) |
    "full_plain" (plain denoise -> CLAHE -> unsharp, noise_aware=False) |
    "gated" (noise-gated denoising, preprocess_gated = the pipeline default, DEFAULT_INPUT).
    """
    if variant == "raw":
        return img
    if variant == "denoise":
        return denoise(img)
    if variant == "clahe":
        return clahe_lab(img)
    if variant == "full":
        return preprocess_for_depth(img, noise_aware=True)
    if variant == "full_plain":
        return preprocess_for_depth(img, noise_aware=False)
    if variant == "gated":
        return preprocess_gated(img)
    raise ValueError(variant)


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #
def median_bilateral(d: np.ndarray) -> np.ndarray:
    """Steps 1-2: median (spike removal) then bilateral (edge-preserving smoothing) on a [0,1] map."""
    d = cv2.medianBlur(d.astype(np.float32), REF_MEDIAN)
    dia, s_range, s_space = REF_BILATERAL
    return cv2.bilateralFilter(d, dia, s_range, s_space)


def edge_aware_smooth(d: np.ndarray, edges: np.ndarray, radius: int = REF_EDGE_RADIUS) -> np.ndarray:
    """Step 3: smooth only away from image edges.

    dist = distance of each pixel to the nearest Canny edge (distance transform).
    w    = min(dist / radius, 1): 0 on an edge, 1 in the middle of a surface.
    out  = w * GaussianBlur(d) + (1 - w) * d
    The blur sigma is radius / 2, so wherever w = 1 the blur window does not reach an edge:
    surfaces are flattened, boundaries keep the original sharp depth step.
    """
    dist = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    w = np.clip(dist / radius, 0, 1).astype(np.float32)
    smooth = cv2.GaussianBlur(d, (0, 0), sigmaX=radius / 2)
    return w * smooth + (1 - w) * d


def fill_holes(d: np.ndarray, invalid: np.ndarray | None = None, ksize: int = REF_HOLE_KERNEL,
               thresh: float = REF_HOLE_THRESH) -> np.ndarray:
    """Step 4: morphological hole filling on a [0,1] map.

    Grey-level closing (dilate -> erode) raises every dark pit narrower than the kernel to the
    level of its surroundings. (closing - d) is the "black top-hat": it is large exactly on such
    pits, so only pixels where it exceeds `thresh` are replaced and the rest stays untouched.
    Pixels flagged `invalid` (NaN / zero depth) are filled the same way by growing valid
    neighbours into them.
    """
    d = d.astype(np.float32).copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if invalid is not None and invalid.any():
        d[invalid] = 0
        todo = invalid.copy()
        while todo.any():                       # dilation spreads valid depth into the hole, ring by ring
            grown = cv2.dilate(d, k)
            newly = todo & (grown > 0)
            if not newly.any():
                break
            d[newly] = grown[newly]
            todo &= ~newly
    closed = cv2.morphologyEx(d, cv2.MORPH_CLOSE, k)
    holes = (closed - d) > thresh
    d[holes] = closed[holes]
    return d


def refine_depth(disp: np.ndarray, rgb: np.ndarray, return_steps: bool = False):
    """Full refinement of a predicted depth/disparity map, guided by the RGB image's Canny edges.

    The map is normalised to [0, 1], refined, and mapped back to its original range.
    With return_steps=True also returns the intermediate maps (for figures).
    """
    invalid = ~np.isfinite(disp)
    lo, hi = float(np.nanmin(disp)), float(np.nanmax(disp))
    scale = max(hi - lo, 1e-8)
    d0 = np.nan_to_num((disp - lo) / scale, nan=0.0).astype(np.float32)

    edges = canny_edges(rgb)
    if edges.shape != d0.shape:
        edges = cv2.resize(edges.astype(np.uint8), d0.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    d1 = median_bilateral(d0)
    d2 = edge_aware_smooth(d1, edges)
    d3 = fill_holes(d2, invalid if invalid.any() else None)
    out = (d3 * scale + lo).astype(np.float32)
    if return_steps:
        return out, {"input": d0, "median_bilateral": d1, "edge_aware": d2, "hole_filled": d3, "edges": edges}
    return out


def refine_metric_depth(depth: np.ndarray, rgb: np.ndarray, return_steps: bool = False):
    """refine_depth for a metric depth map (metres in, metres out).

    The filters run on INVERSE depth, the same quantity they were evaluated on with the relative
    models: there near objects get the fine resolution and far walls are not over-weighted.
    """
    inv = 1.0 / np.maximum(depth, 1e-3)
    if return_steps:
        out, steps = refine_depth(inv, rgb, return_steps=True)
        return (1.0 / np.maximum(out, 1e-6)).astype(np.float32), steps
    return (1.0 / np.maximum(refine_depth(inv, rgb), 1e-6)).astype(np.float32)
