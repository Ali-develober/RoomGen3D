"""Image analysis: Canny edge detection and room segmentation (walls / floor / ceiling / furniture).

Canny gives the object and wall boundaries used later to keep depth edges sharp.
The segmenter (SegFormer trained on ADE20K, 150 classes) labels every pixel; the classes are
merged into four room groups and each mask is cleaned with morphological opening + closing.
"""

import cv2
import numpy as np
import torch

SEG_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K class names (as in the model config) merged into room groups.
ROOM_GROUPS = {
    "wall": {"wall"},
    "floor": {"floor", "rug", "carpet"},
    "ceiling": {"ceiling"},
    "furniture": {"bed", "cabinet", "sofa", "table", "chair", "armchair", "seat", "desk", "wardrobe",
                  "shelf", "bookcase", "chest of drawers", "counter", "countertop", "coffee table",
                  "stool", "bench", "ottoman", "swivel chair", "kitchen island", "buffet", "cradle",
                  "pool table", "cushion", "pillow", "lamp", "television receiver", "sink", "bathtub",
                  "toilet", "refrigerator", "stove", "oven", "dishwasher", "microwave", "washer"},
}
GROUP_COLORS = {"wall": (66, 135, 245), "floor": (240, 160, 40), "ceiling": (120, 200, 120),
                "furniture": (225, 70, 120)}
MORPH_KERNEL = 7        # px, elliptical structuring element for mask cleaning (at 480x640)


# --------------------------------------------------------------------------- #
# Canny edges
# --------------------------------------------------------------------------- #
def canny_edges(img: np.ndarray, blur_sigma: float = 1.4, low: float | None = None,
                high: float | None = None) -> np.ndarray:
    """Canny edge map (bool HxW) of an RGB uint8 image.

    Steps inside Canny: Gaussian smoothing -> Sobel gradient -> non-maximum suppression
    (thin edges to 1 px) -> hysteresis (keep weak edges only if connected to strong ones).
    If no thresholds are given they are set from the image itself: 0.66x and 1.33x the
    median grey level, so dark and bright images both get a sensible number of edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    if low is None or high is None:
        med = float(np.median(gray))
        low, high = max(0.0, 0.66 * med), min(255.0, 1.33 * med)
    return cv2.Canny(gray, low, high) > 0


# --------------------------------------------------------------------------- #
# Room segmentation
# --------------------------------------------------------------------------- #
def load_segmenter(device, model_name: str = SEG_MODEL):
    """SegFormer (ADE20K) in fp16. Returns (model, processor). Free it before loading another model."""
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(model_name)


@torch.inference_mode()
def segment_classes(model, processor, img: np.ndarray, device) -> np.ndarray:
    """ADE20K class id for every pixel (HxW int array at the input resolution)."""
    inp = processor(images=img, return_tensors="pt")["pixel_values"].to(device, torch.float16)
    logits = model(pixel_values=inp).logits                      # 1 x 150 x h/4 x w/4
    logits = torch.nn.functional.interpolate(logits.float(), size=img.shape[:2], mode="bilinear",
                                             align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.int32)


def group_ids(id2label: dict) -> dict:
    """Room group -> list of ADE20K class ids (matched by class name)."""
    out = {g: [] for g in ROOM_GROUPS}
    for i, label in id2label.items():
        names = {n.strip().lower() for n in str(label).split(",")}
        for g, wanted in ROOM_GROUPS.items():
            if names & wanted:
                out[g].append(int(i))
    return out


def clean_mask(mask: np.ndarray, ksize: int = MORPH_KERNEL) -> np.ndarray:
    """Morphological opening then closing of a binary mask.

    Opening (erode -> dilate) deletes specks smaller than the kernel;
    closing (dilate -> erode) fills pinholes and small gaps. Large shapes keep their outline.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m > 0


def room_masks(class_map: np.ndarray, id2label: dict, clean: bool = True) -> dict:
    """Boolean masks {"wall", "floor", "ceiling", "furniture"} from the class map."""
    masks = {}
    for g, ids in group_ids(id2label).items():
        m = np.isin(class_map, ids)
        masks[g] = clean_mask(m) if clean else m
    return masks


def segment_room(model, processor, img: np.ndarray, device, clean: bool = True) -> dict:
    """Convenience: image -> cleaned room masks."""
    return room_masks(segment_classes(model, processor, img, device), model.config.id2label, clean)


# --------------------------------------------------------------------------- #
# Overlays (for figures)
# --------------------------------------------------------------------------- #
def overlay_edges(img: np.ndarray, edges: np.ndarray, color=(255, 40, 40)) -> np.ndarray:
    """Draw the edge map on top of a dimmed copy of the image."""
    out = (img.astype(np.float32) * 0.55).astype(np.uint8)
    out[cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = color
    return out


def overlay_masks(img: np.ndarray, masks: dict, alpha: float = 0.55) -> np.ndarray:
    """Blend each room mask in its group colour over the image."""
    out = img.astype(np.float32)
    for g, m in masks.items():
        out[m] = (1 - alpha) * out[m] + alpha * np.array(GROUP_COLORS[g], np.float32)
    return out.astype(np.uint8)
