"""Preprocessing: resize/normalize, Gaussian/bilateral denoising, CLAHE, unsharp masking.

Training images (LoRA) only get: bicubic 256 -> 512 upsampling + mild unsharp masking.
CLAHE is deliberately NOT applied to training images: it would change the lighting
style the model learns. CLAHE belongs to the depth-estimation stage.
"""

from pathlib import Path

import cv2
import numpy as np

# Mild sharpening: radius ~ 1 px at 512x512, adds back 50% of the high-frequency detail.
USM_SIGMA, USM_AMOUNT = 1.0, 0.5


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8 (works with non-ASCII Windows paths)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot decode {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def upsample_bicubic(img: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize to size x size with bicubic interpolation (4x4 neighbourhood, smoother than bilinear).

    Upsampling cannot invent detail, so the result is softer than a native 512 image.
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def unsharp_mask(img: np.ndarray, sigma: float = USM_SIGMA, amount: float = USM_AMOUNT) -> np.ndarray:
    """Unsharp masking: out = img + amount * (img - GaussianBlur(img, sigma)).

    (img - blur) is the high-frequency detail (edges, texture); adding a fraction
    of it back boosts edge contrast lost by upsampling. Small sigma/amount = mild.
    """
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    return np.clip(f + amount * (f - blur), 0, 255).astype(np.uint8)


def preprocess_for_training(img: np.ndarray, size: int = 512) -> np.ndarray:
    """LoRA training preprocessing: bicubic upsample, then mild unsharp masking. RGB uint8 in/out."""
    return unsharp_mask(upsample_bicubic(img, size))


def to_model_tensor(img: np.ndarray):
    """RGB uint8 HxWx3 -> float tensor 3xHxW in [-1, 1] (the range SD's VAE expects)."""
    import torch
    return torch.from_numpy(img).permute(2, 0, 1).float().div(127.5).sub(1.0)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian of the grayscale image (higher = more edge detail)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
