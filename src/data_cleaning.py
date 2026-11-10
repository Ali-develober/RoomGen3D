"""Training-data cleaning: Laplacian-variance blur detection and histogram-based
exposure filtering (README section 4).

Per image we compute three kinds of scores on the grayscale image:
  * sharpness  = variance of the Laplacian (low -> few edges -> blurry)
  * brightness = mean gray level in [0, 255] (very low -> under-exposed, very high -> over-exposed)
  * clipping   = % of pixels at the ends of the histogram (<= 5 is crushed black, >= 250 is blown white)

Thresholds are not hand-picked: they are Tukey fences (Q1 - k*IQR, Q3 + k*IQR)
computed on the score distribution of the dataset itself, so only statistical
outliers are rejected. Clipping uses a variant of the fence (see clipping_threshold)
because most images clip almost nothing.

Near-duplicates are found with a perceptual hash (pHash, DCT-based) and the
Hamming distance between hashes; the distance threshold is Otsu's split of the
nearest-neighbour distance histogram.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DARK_LEVEL, BRIGHT_LEVEL = 5, 250


# --------------------------------------------------------------------------- #
# Per-image scores
# --------------------------------------------------------------------------- #
def load_gray(path: str | Path) -> np.ndarray:
    """Read an image as 8-bit grayscale (cv2.imdecode handles non-ASCII Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot decode {path}")
    return img


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. The Laplacian responds to edges; a blurry image
    has weak edges, so the spread (variance) of the response is small."""
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())


def exposure_stats(gray: np.ndarray) -> dict:
    """Histogram-based exposure measures: mean brightness and % of clipped pixels."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    n = hist.sum()
    levels = np.arange(256)
    return {
        "brightness": float((hist * levels).sum() / n),
        "pct_dark": float(hist[: DARK_LEVEL + 1].sum() / n * 100),
        "pct_bright": float(hist[BRIGHT_LEVEL:].sum() / n * 100),
    }


def phash(gray: np.ndarray) -> str:
    """64-bit perceptual hash (pHash) as a 16-char hex string.

    1. shrink to 32x32 (drops fine detail, keeps overall structure)
    2. 2-D DCT; keep the 8x8 lowest-frequency coefficients
    3. bit = 1 where a coefficient is above the median of the 64
    Near-identical images (resized, recompressed, slightly edited) get hashes
    that differ in only a few bits (small Hamming distance).
    """
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).ravel()
    return np.packbits(bits).tobytes().hex()


def score_image(path: str | Path) -> dict:
    """All cleaning scores for one image."""
    gray = load_gray(path)
    return {"lap_var": blur_score(gray), **exposure_stats(gray), "phash": phash(gray)}


def score_images(paths, progress: bool = True) -> pd.DataFrame:
    """Score a list of image paths; unreadable files get NaN scores and are rejected later."""
    from tqdm import tqdm
    rows = []
    for p in tqdm(paths, disable=not progress, unit="img"):
        try:
            rows.append(score_image(p))
        except IOError:
            rows.append({"lap_var": np.nan, "brightness": np.nan, "pct_dark": np.nan, "pct_bright": np.nan, "phash": None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data-driven thresholds
# --------------------------------------------------------------------------- #
def tukey_fences(x, k: float = 1.5) -> tuple[float, float]:
    """(Q1 - k*IQR, Q3 + k*IQR): the standard box-plot outlier rule."""
    q1, q3 = np.nanpercentile(x, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def choose_thresholds(scores: pd.DataFrame, k: float = 1.5) -> dict:
    """Pick cleaning thresholds from the score distributions.

    * blur: lower fence of log10(Laplacian variance). Sharpness is right-skewed
      over orders of magnitude, so the fence is computed in log space.
    * brightness: lower and upper fences of mean brightness.
    * clipping: upper fence of log10(% clipped), computed only over the images
      that clip more than the median image (see clipping_threshold).
    """
    blur_lo, _ = tukey_fences(np.log10(scores["lap_var"]), k)
    bri_lo, bri_hi = tukey_fences(scores["brightness"], k)
    return {
        "lap_var_min": float(10 ** blur_lo),
        "brightness_min": float(max(bri_lo, 0)),
        "brightness_max": float(min(bri_hi, 255)),
        "pct_dark_max": clipping_threshold(scores["pct_dark"], k),
        "pct_bright_max": clipping_threshold(scores["pct_bright"], k),
        "k": k,
    }


def clipping_threshold(pct: pd.Series, k: float = 1.5) -> float:
    """Upper Tukey fence for a clipped-pixel percentage.

    Clipping is zero-inflated: half the images have almost no clipped pixels, so a
    fence over all images has a tiny IQR and rejects any dark or bright room
    (e.g. a moody Art Deco interior at 3% black pixels). Instead we take the images
    above the median clipping and fence their log-percentage, which only
    flags the extreme tail (black letterbox bars, white catalogue padding).
    """
    tail = pct[pct > pct.median()]
    _, hi = tukey_fences(np.log10(tail), k)
    return float(10 ** hi)


def apply_thresholds(scores: pd.DataFrame, th: dict) -> pd.Series:
    """Return the rejection reason per image ('' = keep). Several reasons are joined with ';'."""
    checks = {
        "unreadable": scores["lap_var"].isna(),
        "blurry": scores["lap_var"] < th["lap_var_min"],
        "underexposed": scores["brightness"] < th["brightness_min"],
        "overexposed": scores["brightness"] > th["brightness_max"],
        "clipped_dark": scores["pct_dark"] > th["pct_dark_max"],
        "clipped_bright": scores["pct_bright"] > th["pct_bright_max"],
    }
    reasons = pd.DataFrame(checks)
    return reasons.apply(lambda r: ";".join(c for c in reasons.columns if r[c]), axis=1)


# --------------------------------------------------------------------------- #
# Near-duplicate removal (pHash + Hamming distance)
# --------------------------------------------------------------------------- #
def hashes_to_uint64(hex_hashes) -> np.ndarray:
    """Hex pHash strings -> uint64 array (for fast XOR + popcount)."""
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def hamming_to_all(h: np.ndarray, rows: slice) -> np.ndarray:
    """Hamming distances between hashes h[rows] and every hash (number of differing bits)."""
    return np.bitwise_count(h[rows, None] ^ h[None, :]).astype(np.int16)


def nearest_neighbor_distance(h: np.ndarray, block: int = 1000) -> np.ndarray:
    """For every image, the Hamming distance to its closest other image."""
    nn = np.empty(len(h), dtype=np.int16)
    for i in range(0, len(h), block):
        d = hamming_to_all(h, slice(i, i + block))
        d[np.arange(d.shape[0]), np.arange(i, i + d.shape[0])] = 64 + 1  # ignore self
        nn[i : i + block] = d.min(1)
    return nn


def choose_duplicate_threshold(nn: np.ndarray) -> int:
    """Otsu's threshold on the nearest-neighbour distance histogram.

    The histogram is bimodal: a spike near 0 (duplicates) and a broad mode at
    12-20 bits (different images). Otsu picks the split that maximises the
    between-class variance; distances <= threshold count as duplicates.
    """
    from skimage.filters import threshold_otsu
    return int(np.floor(threshold_otsu(nn.astype(float))))


def duplicate_pairs(h: np.ndarray, thr: int, block: int = 1000) -> list[tuple[int, int, int]]:
    """All pairs (i, j, distance) with i < j and distance <= thr."""
    out = []
    for i in range(0, len(h), block):
        d = hamming_to_all(h, slice(i, i + block))
        for a, b in zip(*np.nonzero(d <= thr)):
            if i + a < b:
                out.append((int(i + a), int(b), int(d[a, b])))
    return out


def duplicate_groups(n: int, pairs) -> list[list[int]]:
    """Union-find: connect duplicate pairs into groups (only groups of 2+ returned)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def mark_duplicates(df: pd.DataFrame, groups) -> pd.Series:
    """Keep one image per duplicate group, mark the rest 'duplicate'.

    Only images that passed the quality checks compete. MMIS sometimes files the
    same picture under two styles, so the keeper is the copy whose (style, room)
    cell currently has the fewest kept images; this keeps the subset balanced.
    """
    reason = df["reason"].copy()
    kept = reason == ""
    cell_count = df[kept].groupby(["style", "room"]).size().to_dict()
    for g in groups:
        alive = [i for i in g if kept[i]]
        if len(alive) < 2:
            continue
        keep = min(alive, key=lambda i: (cell_count[(df.at[i, "style"], df.at[i, "room"])], -df.at[i, "lap_var"]))
        for i in alive:
            if i != keep:
                reason[i] = "duplicate"
                cell_count[(df.at[i, "style"], df.at[i, "room"])] -= 1
    return reason


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_duplicate_distances(nn: np.ndarray, thr: int, out_path: str | Path) -> None:
    """Histogram of nearest-neighbour pHash distances with the Otsu threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(65), np.bincount(nn, minlength=65)[:65], width=1.6, color="#4C72B0")
    ax.axvline(thr + 1, color="red", lw=2, ls="--", label=f"Otsu threshold: duplicate if distance <= {thr}")
    ax.set_yscale("log")
    ax.set_xlim(-1, 34)
    ax.set_xlabel("Hamming distance to nearest other image (bits of 64)")
    ax.set_ylabel("images (log scale)")
    ax.set_title(f"pHash nearest-neighbour distances (n={len(nn)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_duplicate_pairs(df: pd.DataFrame, pairs, root: str | Path, out_path: str | Path,
                         per_distance: int = 3, seed: int = 42) -> None:
    """Example duplicate pairs, one row per Hamming distance found."""
    rng = np.random.default_rng(seed)
    by_d = {}
    for i, j, d in pairs:
        by_d.setdefault(d, []).append((i, j))
    dists = sorted(by_d)
    fig, axes = plt.subplots(len(dists), 2 * per_distance, figsize=(2.1 * 2 * per_distance, 2.4 * len(dists)), squeeze=False)
    for r, d in enumerate(dists):
        ps = by_d[d]
        pick = [ps[k] for k in rng.choice(len(ps), min(per_distance, len(ps)), replace=False)]
        for c in range(per_distance):
            for t in range(2):
                a = axes[r, 2 * c + t]
                a.axis("off")
                if c < len(pick):
                    i = pick[c][t]
                    img = cv2.imdecode(np.fromfile(str(Path(root) / df.at[i, "path"]), np.uint8), cv2.IMREAD_COLOR)
                    a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    a.set_title(f"{df.at[i, 'style'][:12]}\n{df.at[i, 'room']}", fontsize=7)
        axes[r, 0].text(-0.08, 0.5, f"distance {d}\n({len(ps)} pairs)", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle("Near-duplicate pairs found by pHash (each pair = 2 adjacent images)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
def plot_score_distributions(scores: pd.DataFrame, th: dict, out_path: str | Path, title: str = "") -> None:
    """Histograms of the four scores with the chosen thresholds as red lines."""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    panels = [
        (np.log10(scores["lap_var"]), [np.log10(th["lap_var_min"])], "log10(Laplacian variance)", "Blur: reject left of line"),
        (scores["brightness"], [th["brightness_min"], th["brightness_max"]], "mean brightness (0-255)", "Exposure: reject outside lines"),
        (np.log10(1 + scores["pct_dark"]), [np.log10(1 + th["pct_dark_max"])], f"log10(1 + % pixels <= {DARK_LEVEL})", "Dark clipping: reject right"),
        (np.log10(1 + scores["pct_bright"]), [np.log10(1 + th["pct_bright_max"])], f"log10(1 + % pixels >= {BRIGHT_LEVEL})", "Bright clipping: reject right"),
    ]
    for a, (x, lines, xlabel, t) in zip(ax, panels):
        a.hist(x.dropna(), bins=60, color="#4C72B0", alpha=0.85)
        for v in lines:
            a.axvline(v, color="red", lw=2, ls="--")
        a.set_xlabel(xlabel)
        a.set_ylabel("images")
        a.set_title(t)
    fig.suptitle(title or f"Cleaning score distributions (n={len(scores)}), Tukey fences k={th['k']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_kept_vs_rejected(df: pd.DataFrame, root: str | Path, out_path: str | Path,
                          n_per_row: int = 6, seed: int = 42) -> None:
    """Grid: one row of kept images, then one row per rejection reason (worst examples first)."""
    rows = [("kept", df[df["reason"] == ""].sample(n=min(n_per_row, (df["reason"] == "").sum()), random_state=seed))]
    order = {"blurry": ("lap_var", True), "underexposed": ("brightness", True), "overexposed": ("brightness", False),
             "clipped_dark": ("pct_dark", False), "clipped_bright": ("pct_bright", False)}
    for reason, (col, asc) in order.items():
        hit = df[df["reason"].str.contains(reason)]
        if len(hit):
            rows.append((f"rejected: {reason}", hit.sort_values(col, ascending=asc).head(n_per_row)))
    fig, axes = plt.subplots(len(rows), n_per_row, figsize=(2.3 * n_per_row, 2.6 * len(rows)), squeeze=False)
    for r, (label, sel) in enumerate(rows):
        for c in range(n_per_row):
            a = axes[r, c]
            a.axis("off")
            if c < len(sel):
                row = sel.iloc[c]
                img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(Path(root) / row["path"]), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                a.imshow(img)
                a.set_title(f"lap={row['lap_var']:.0f} br={row['brightness']:.0f}\n"
                            f"dark={row['pct_dark']:.1f}% bright={row['pct_bright']:.1f}%", fontsize=7)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per (style, room) before and after cleaning, plus rejected count and % kept."""
    g = df.groupby(["style", "room"])
    t = pd.DataFrame({"before": g.size(), "after": g.apply(lambda x: (x["reason"] == "").sum())})
    t["rejected"] = t["before"] - t["after"]
    t["kept_pct"] = (100 * t["after"] / t["before"]).round(1)
    return t.reset_index()
