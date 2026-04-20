"""
pipeline_transforms.py
----------------------
18 image preprocessing pipelines:
  Step 1 — contrast enhancement  : CLAHE | Original | Retinex
  Step 2 — edge-preserving filter : Bilateral | NLM | Kramer-Bruckner |
                                    EPOAF | SNN | BM3D
  Fast filters run first (12), slow filters (SNN × 3, BM3D × 3) run last.

All functions accept and return uint8 RGB numpy arrays.
"""

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Contrast Enhancement
# ══════════════════════════════════════════════════════════════════════════════

def contrast_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation on the L channel of LAB.
    clipLimit=2.0, tile grid 8×8.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def contrast_original(img_rgb: np.ndarray) -> np.ndarray:
    """No contrast modification — pass through."""
    return img_rgb.copy()


def contrast_retinex(img_rgb: np.ndarray,
                     sigmas: tuple = (25.0, 50.0, 100.0)) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) with per-channel normalisation.
    Models the visual system's luminance adaptation; reveals detail in shadows.
    R(x,y) = Σ_s [ log(I) - log(I ⊛ G_s) ]  averaged over scales.
    """
    img_f = img_rgb.astype(np.float32) + 1.0      # avoid log(0)
    msr   = np.zeros_like(img_f)
    max_sigma = min(img_f.shape[:2]) / 6.0         # cap so kernel never exceeds image
    for s in sigmas:
        blurred = cv2.GaussianBlur(img_f, (0, 0), min(s, max_sigma))
        msr    += np.log(img_f) - np.log(blurred + 1e-6)
    msr /= len(sigmas)
    # Per-channel min-max normalise to [0, 255]
    out = np.zeros_like(img_f)
    for c in range(3):
        ch       = msr[:, :, c]
        lo, hi   = ch.min(), ch.max()
        out[:, :, c] = (ch - lo) / (hi - lo + 1e-6) * 255.0
    return out.clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Edge-Preserving Filters
# ══════════════════════════════════════════════════════════════════════════════

def filter_bilateral(img_rgb: np.ndarray) -> np.ndarray:
    """
    Bilateral filter — Gaussian smoothing that weights pixels by both
    spatial distance and radiometric similarity, preserving sharp edges.
    d=50, sigmaColor=20, sigmaSpace=100  (large patch, tight colour gate,
    broad spatial reach — strong smoothing without edge bleed).
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(bgr, d=50, sigmaColor=20, sigmaSpace=100)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def filter_nlm(img_rgb: np.ndarray) -> np.ndarray:
    """
    Non-Local Means denoising.  Averages patches across the whole image
    weighted by patch similarity — very effective at removing grain while
    retaining fine texture.
    h=10, templateWindowSize=7, searchWindowSize=21.
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None,
                                          h=10, hColor=10,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def filter_bm3d(img_rgb: np.ndarray) -> np.ndarray:
    """
    BM3D (Block-Matching 3D) — state-of-the-art denoising.
    Groups similar patches into 3D stacks, applies collaborative filtering,
    then aggregates.  sigma_psd=25/255 (moderate noise level).
    """
    import bm3d as _bm3d
    img_f = img_rgb.astype(np.float32) / 255.0
    out   = np.zeros_like(img_f)
    sigma = 25.0 / 255.0
    for c in range(3):
        out[:, :, c] = _bm3d.bm3d(img_f[:, :, c], sigma_psd=sigma)
    return (out.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def filter_snn(img_rgb: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Symmetric Nearest Neighbour (SNN) filter.
    For each symmetric pair of neighbours (a, b) around centre pixel c,
    selects whichever is closer in value to c.  Result = mean of all
    selected values.  Smooths homogeneous regions; preserves edges and
    isolated bright/dark points.
    """
    img   = img_rgb.astype(np.float32)
    H, W  = img.shape[:2]
    pad   = radius
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    center = padded[pad:pad+H, pad:pad+W]     # (H,W,3)
    result = np.zeros_like(img)
    n_pairs = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            sym_dy, sym_dx = -dy, -dx
            a = padded[pad+dy   : pad+dy+H,    pad+dx   : pad+dx+W]
            b = padded[pad+sym_dy: pad+sym_dy+H, pad+sym_dx: pad+sym_dx+W]
            d_a = np.sum((a - center) ** 2, axis=2, keepdims=True)
            d_b = np.sum((b - center) ** 2, axis=2, keepdims=True)
            result += np.where(d_a <= d_b, a, b)
            n_pairs += 1

    return (result / n_pairs).clip(0, 255).astype(np.uint8)


def filter_kramer_bruckner(img_rgb: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Kramer & Bruckner (1975) non-linear edge-preserving filter.
    Processes symmetric neighbour pairs exactly as SNN, but starts the
    accumulator with the centre pixel (giving it double weight relative
    to each pair), which better preserves point features.
    """
    img   = img_rgb.astype(np.float32)
    H, W  = img.shape[:2]
    pad   = radius
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    center = padded[pad:pad+H, pad:pad+W]
    result = center.copy()        # start with centre (weight=1)
    count  = np.ones((H, W, 1), dtype=np.float32)

    # Only process each pair once (dy,dx) and its mirror
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            # Visit only the "positive" half to avoid counting pairs twice,
            # but we need both halves to maintain the SNN selection logic
            sym_dy, sym_dx = -dy, -dx
            a = padded[pad+dy   : pad+dy+H,    pad+dx   : pad+dx+W]
            b = padded[pad+sym_dy: pad+sym_dy+H, pad+sym_dx: pad+sym_dx+W]
            d_a = np.sum((a - center) ** 2, axis=2, keepdims=True)
            d_b = np.sum((b - center) ** 2, axis=2, keepdims=True)
            result += np.where(d_a <= d_b, a, b)
            count  += 1

    return (result / count).clip(0, 255).astype(np.uint8)


def filter_epoaf(img_rgb: np.ndarray) -> np.ndarray:
    """
    Edge-Preserving Oriented Adaptive Filter (EPOAF).
    Computes local gradient orientation via Sobel, then smooths each pixel
    along the edge tangent direction (not across it).  This removes noise
    parallel to edges while keeping edge sharpness intact.
    Sampling offsets: −2, −1, +1, +2 pixels along the tangent.
    """
    img_f = img_rgb.astype(np.float32)
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Gradient → edge tangent
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    # Tangent = perpendicular to gradient  (rotate 90°)
    tx = -gy / mag   # (H,W)
    ty =  gx / mag

    H, W = gray.shape
    base_x, base_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    accum = img_f.copy()
    steps = (-2.0, -1.0, 1.0, 2.0)
    for s in steps:
        map_x = (base_x + tx * s).clip(0, W - 1)
        map_y = (base_y + ty * s).clip(0, H - 1)
        sampled_bgr = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)
        accum += cv2.cvtColor(sampled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    return (accum / (1 + len(steps))).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline registry
# ══════════════════════════════════════════════════════════════════════════════

CONTRAST_STEPS = {
    "original": contrast_original,
    "clahe":    contrast_clahe,
    "retinex":  contrast_retinex,
}

EDGE_STEPS = {
    "bilateral":       filter_bilateral,
    "nlm":             filter_nlm,
    "kramer_bruckner": filter_kramer_bruckner,
    "epoaf":           filter_epoaf,
    "snn":             filter_snn,
    "bm3d":            filter_bm3d,
}

PIPELINES: dict[str, tuple] = {}
for _c, _cfn in CONTRAST_STEPS.items():
    for _e, _efn in EDGE_STEPS.items():
        PIPELINES[f"{_c}_{_e}"] = (_cfn, _efn)

# Fast filters first (12), slow filters last (SNN × 3, BM3D × 3)
_FAST = ["bilateral", "nlm", "kramer_bruckner", "epoaf"]
_SLOW = ["snn", "bm3d"]
PIPELINE_NAMES = (
    [f"{c}_{e}" for c in CONTRAST_STEPS for e in _FAST] +
    [f"{c}_{e}" for c in CONTRAST_STEPS for e in _SLOW]
)


def apply_pipeline(img_rgb: np.ndarray, name: str) -> np.ndarray:
    """Apply the named pipeline (contrast → edge filter) to an RGB uint8 image."""
    c_fn, e_fn = PIPELINES[name]
    return e_fn(c_fn(img_rgb))


def pipeline_label(name: str) -> str:
    """Human-readable label, e.g. 'CLAHE + Bilateral'."""
    c, e = name.split("_", 1)
    C = {"original": "Original", "clahe": "CLAHE", "retinex": "Retinex"}
    E = {"bilateral": "Bilateral", "nlm": "NLM", "bm3d": "BM3D",
         "snn": "SNN", "kramer_bruckner": "Kramer-Bruckner", "epoaf": "EPOAF"}
    return f"{C.get(c, c)} + {E.get(e, e)}"


if __name__ == "__main__":
    # Quick smoke-test on a synthetic image
    import os, sys
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    for name in PIPELINE_NAMES:
        out = apply_pipeline(test_img, name)
        assert out.shape == test_img.shape and out.dtype == np.uint8, \
            f"Pipeline {name} returned bad shape/dtype"
        print(f"  {name:35s} OK  mean={out.mean():.1f}")
    print(f"\nAll {len(PIPELINE_NAMES)} pipelines OK.")
