"""Preprocessing: resize/normalize, Gaussian/bilateral denoising, CLAHE, unsharp masking.

Training images (LoRA) only get: bicubic 256 -> 512 upsampling + mild unsharp masking.
CLAHE is deliberately NOT applied to training images: it would change the lighting
style the model learns. CLAHE belongs to the depth-estimation stage.

Depth-stage preprocessing (preprocess_for_depth), in this order:
    1. denoise   - noise would be amplified by the next two steps, so it goes first
    2. CLAHE     - local contrast on the lightness channel only (colours unchanged)
    3. unsharp   - mild sharpening of the edges softened by step 1
An optional noise-aware variant (noise_aware=True) denoises again after CLAHE and skips step 3 for
noisy inputs. It is NOT the default: on NYU it gave no better depth than the plain three steps.
"""

from pathlib import Path

import cv2
import numpy as np

# Mild sharpening: radius ~ 1 px at 512x512, adds back 50% of the high-frequency detail.
USM_SIGMA, USM_AMOUNT = 1.0, 0.5
CLAHE_CLIP, CLAHE_GRID = 2.0, 8     # standard mild CLAHE: clip limit 2, 8x8 tiles
NOISY_INPUT = 5.0                   # estimated noise std above which an image counts as noisy


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


# --------------------------------------------------------------------------- #
# Depth-stage preprocessing: denoising, CLAHE, full pipeline
# --------------------------------------------------------------------------- #
def estimate_noise_sigma(img: np.ndarray) -> float:
    """Fast estimate of the Gaussian noise std in grey levels (Immerkaer, 1996).

    The 3x3 kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] is a difference of two Laplacians: it cancels
    smooth image structure and keeps mostly noise. sigma = sqrt(pi/2) * mean(|response|) / 6.
    Measured on the grey image, whose noise is smaller than the per-channel noise by
    sqrt(0.299^2 + 0.587^2 + 0.114^2) = 0.669 (weighted average of R, G, B); we divide by that
    factor to report the per-channel std. Fine texture raises the estimate slightly.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    resp = cv2.filter2D(gray, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(resp).mean() / 6 / 0.669)


def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian low-pass filter. Removes noise but blurs edges equally (linear, not edge-preserving)."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def denoise_bilateral(img: np.ndarray, sigma_color: float = 40.0, sigma_space: float = 5.0,
                      d: int = 9) -> np.ndarray:
    """Bilateral filter: average of neighbours weighted by distance AND by colour difference.

    Pixels across an edge differ a lot in colour, get a ~0 weight and are not mixed in,
    so flat regions are smoothed while edges stay sharp. sigma_color = how large a colour
    difference still counts as "the same surface" (should be ~2-3x the noise std).
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise(img: np.ndarray, method: str = "auto") -> np.ndarray:
    """Denoise an RGB uint8 image. method: "auto" | "bilateral" | "gaussian".

    "auto" adapts the strength to the measured noise level: sigma_color = 3 x the estimated
    noise std (clipped to 10..90). A clean image is then almost untouched and a noisy one is
    filtered hard. On NYU test images with added Gaussian noise this raises PSNR from
    25.1 to 32.5 dB (std 15) and from 20.9 to 28.9 dB (std 25).
    """
    if method == "gaussian":
        return denoise_gaussian(img)
    if method == "bilateral":
        return denoise_bilateral(img)
    return denoise_bilateral(img, sigma_color=float(np.clip(3.0 * estimate_noise_sigma(img), 10, 90)))


def clahe_lab(img: np.ndarray, clip_limit: float = CLAHE_CLIP, tile_grid: int = CLAHE_GRID) -> np.ndarray:
    """CLAHE on the L (lightness) channel of LAB; a and b (colour) are left untouched.

    CLAHE = histogram equalisation done per tile (tile_grid x tile_grid tiles) with the
    histogram clipped at clip_limit so noise in flat areas is not over-amplified; tiles are
    blended bilinearly. It lifts detail in dark corners and tames bright windows, which
    plain global equalisation cannot do for unevenly lit rooms.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_for_depth(img: np.ndarray, do_denoise: bool = True, do_clahe: bool = True,
                         do_sharpen: bool = True, noise_aware: bool = False) -> np.ndarray:
    """Full depth-stage preprocessing: denoise -> CLAHE -> mild unsharp mask. RGB uint8 in/out.

    Default (noise_aware=False): always the plain three steps.

    noise_aware=True: CLAHE and sharpening both amplify whatever noise the first pass left
    behind (on a std-25 image the estimate went 24 -> 6 after denoising but back to 21 after
    CLAHE + unsharp). So if the INPUT is noisy (estimated std > NOISY_INPUT) the image is
    denoised once more after CLAHE and the sharpening step is skipped; clean images take the
    plain path. The result looks cleaner, but on the 654 NYU test images it did not give better
    depth: equal on clean / blurred input, about the same for Depth Anything on noise, and worse
    for MiDaS on every noisy condition and for both models in severe low light (README section 8).
    That is why the plain version is the default.
    """
    noisy = noise_aware and estimate_noise_sigma(img) > NOISY_INPUT
    if do_denoise:
        img = denoise(img)
    if do_clahe:
        img = clahe_lab(img)
        if do_denoise and noisy:
            img = denoise(img)
    if do_sharpen and not (do_denoise and noisy):
        img = unsharp_mask(img)
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between two uint8 images (higher = closer)."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity (Wang et al., 2004) of two RGB uint8 images, on the grey level.

    Compares local mean (luminance), local variance (contrast) and local covariance (structure)
    in an 11x11 Gaussian window (sigma 1.5). 1 = identical; unlike PSNR it rewards kept
    structure, so a blurry result scores lower even when its pixel error is small.
    """
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(z):
        return cv2.GaussianBlur(z, (11, 11), 1.5)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s[5:-5, 5:-5].mean())          # drop the border where the window is incomplete
