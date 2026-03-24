#!/usr/bin/env python3
"""
bee_enhance_pipeline.py
=======================
Auto-detects individual bee blobs from the pipeline blackout image,
then enhances each closeup for future bee identification.

Combines:
  - enhance_bee_closeups.py  : SNN filter (snnV2.m port) + multi-scale
                               Laplacian detail boost + adaptive gamma
  - bee_epoaf_closeups.py    : edge-adaptive unsharp masking + adaptive CLAHE
  + NEW: Multi-Scale Retinex – log-domain illumination normalisation that
         removes the residual left→right lighting gradient from individual crops,
         revealing true bee surface reflectance for consistent comparison.

Inputs  (pipeline_output/):
  inter_12_blackout.png        – used for bee blob detection
  inter_01_original_raw.png    – source image for crops (better native contrast)

Outputs (pipeline_output/bee_enhance/):
  after/       – enhanced 256×256 closeups
  compare/     – before | after side-by-side tiles
  montage.png  – full grid overview
"""

import cv2
import numpy as np
import os

OUT_ROOT = "pipeline_output/bee_enhance"
os.makedirs(f"{OUT_ROOT}/after",   exist_ok=True)
os.makedirs(f"{OUT_ROOT}/compare", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load images
# ─────────────────────────────────────────────────────────────────────────────

blackout  = cv2.imread("pipeline_output/inter_12_blackout.png",     cv2.IMREAD_GRAYSCALE)
original  = cv2.imread("pipeline_output/inter_01_original_raw.png", cv2.IMREAD_GRAYSCALE)

if blackout is None or original is None:
    raise FileNotFoundError("Run Pipeline_final.py first to generate pipeline_output/")

# Align sizes: the original is the full 1920×1200 raw image; the blackout is
# the cropped 1760×1115 version. Apply the same crop to the original.
if original.shape != blackout.shape:
    orig_h, orig_w = original.shape
    bk_h,  bk_w   = blackout.shape
    # Derive the crop offsets used in Pipeline_final.py
    y_off = (orig_h - bk_h) // 2   # approximate; actual is [40:1155]
    x_off = (orig_w - bk_w) // 2
    # Use the exact crop parameters from Pipeline_final.py
    original = original[40:40 + bk_h, 80:80 + bk_w]

img_h, img_w = blackout.shape
print(f"[load] blackout {img_w}×{img_h}  original (cropped) {original.shape[1]}×{original.shape[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Bee detection — grid sampling with bee-content filter
#
# All bees are touching → connected components yield one giant blob.
# Instead: slide a window at BEE_STEP px intervals; keep windows where
# ≥ MIN_BEE_FRAC of the blackout pixels are non-zero (= bee/content area)
# and the corrected image has enough local variance (= real texture, not flat).
# ─────────────────────────────────────────────────────────────────────────────

CROP_HALF     = 52   # half-side of the 104×104 crop window
BEE_STEP      = 68   # stride between crop centres
MIN_BEE_FRAC  = 0.30 # ≥ 30% non-black (cell-removed) pixels in window
MIN_STD       = 9.0  # minimum intensity std-dev in corrected window
BORDER_MARGIN = 70   # exclude crops within this many px of the image edge
                     # (avoids wooden/metal frame hardware artifacts)

bees = []
y_start = CROP_HALF + BORDER_MARGIN
y_end   = img_h - CROP_HALF - BORDER_MARGIN
x_start = CROP_HALF + BORDER_MARGIN
x_end   = img_w - CROP_HALF - BORDER_MARGIN
for cy in range(y_start, y_end, BEE_STEP):
    for cx in range(x_start, x_end, BEE_STEP):
        y1, y2 = cy - CROP_HALF, cy + CROP_HALF
        x1, x2 = cx - CROP_HALF, cx + CROP_HALF
        bee_frac = float((blackout[y1:y2, x1:x2] > 15).mean())
        if bee_frac < MIN_BEE_FRAC:
            continue
        if float(original[y1:y2, x1:x2].std()) < MIN_STD:
            continue
        bees.append({"cx": cx, "cy": cy, "bee_frac": bee_frac})

print(f"[detect] {len(bees)} bee-content windows  "
      f"(step={BEE_STEP}px, crop={CROP_HALF*2}×{CROP_HALF*2}, "
      f"bee≥{MIN_BEE_FRAC:.0%}, std≥{MIN_STD})")


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement primitives
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Multi-Scale Retinex (new) ─────────────────────────────────────────────
def multi_scale_retinex(gray_u8, sigmas=(15, 80, 250)):
    """
    Log-domain illumination normalisation (Jobson 1997 / Rahman 1996).

    Estimates the illumination at three spatial scales (local, mid, global)
    and divides it out in log-space.  This removes the residual left→right
    brightness gradient that survives single-image brightness correction and
    is still present in individual bee crops.  The result expresses true
    surface reflectance rather than scene luminance, making bee markings
    (yellow stripes, hair texture) comparable across crops regardless of
    where the bee sits on the frame.

    sigmas: (local 15px, medium 80px, global 250px)
    """
    img_f   = gray_u8.astype(np.float32) + 1.0      # +1 avoids log(0)
    log_img = np.log(img_f)
    retinex = np.zeros_like(img_f)
    w = 1.0 / len(sigmas)
    for sigma in sigmas:
        blur    = cv2.GaussianBlur(img_f, (0, 0), sigma)
        blur    = np.maximum(blur, 1.0)
        retinex += w * (log_img - np.log(blur))
    # Linear stretch → [0, 255]
    retinex -= retinex.min()
    if retinex.max() > 1e-6:
        retinex = retinex / retinex.max() * 255.0
    return np.clip(retinex, 0, 255).astype(np.uint8)


# ── 2. SNN filter (Python port of snnV2.m by Art Barnes / O. Idrissi) ────────
def snn_filter(img_f64, win_sz=3):
    """
    Symmetric Nearest Neighbor edge-preserving filter.
    For each pixel the output is the mean of: the pixel itself, and for every
    symmetric pair of neighbours (A, A') the one closest in intensity to the
    centre.  Smooths noise while keeping edges crisp.
    (Matches the logic of snnV2.m exactly.)
    """
    pad = win_sz // 2
    h, w = img_f64.shape
    padded = np.pad(img_f64, pad, mode='edge')

    offsets_top, offsets_bot = [], []
    for dy in range(-pad, 0):
        for dx in range(-pad, pad + 1):
            offsets_top.append((dy, dx))
            offsets_bot.append((-dy, -dx))
    for dx in range(-pad, 0):
        offsets_top.append((0, dx))
        offsets_bot.append((0, -dx))

    centre = padded[pad:pad + h, pad:pad + w]
    acc    = centre.copy()
    count  = np.ones_like(centre)

    for (dy_t, dx_t), (dy_b, dx_b) in zip(offsets_top, offsets_bot):
        v_top  = padded[pad + dy_t:pad + dy_t + h, pad + dx_t:pad + dx_t + w]
        v_bot  = padded[pad + dy_b:pad + dy_b + h, pad + dx_b:pad + dx_b + w]
        chosen = np.where(np.abs(v_top - centre) <= np.abs(v_bot - centre), v_top, v_bot)
        acc   += chosen
        count += 1

    return acc / count


# ── 3. Edge-adaptive unsharp masking (from bee_epoaf_closeups.py) ─────────────
def edge_adaptive_usm(gray_u8, amount=1.6, sigma=1.6, edge_boost=2.4):
    f    = gray_u8.astype(np.float32)
    blr  = cv2.GaussianBlur(f, (0, 0), sigma)
    hf   = f - blr
    gx   = cv2.Scharr(gray_u8, cv2.CV_32F, 1, 0)
    gy   = cv2.Scharr(gray_u8, cv2.CV_32F, 0, 1)
    emag = np.sqrt(gx ** 2 + gy ** 2)
    emax = np.percentile(emag, 95) + 1e-6
    emask = np.clip(emag / emax, 0, 1)
    emask = cv2.GaussianBlur(emask, (0, 0), 2.5)
    result = f + amount * hf + (edge_boost - amount) * emask * hf
    return np.clip(result, 0, 255).astype(np.uint8)


# ── 4. Full enhancement pipeline ─────────────────────────────────────────────
def enhance(crop_gray, target=256):
    """
    Enhancement chain (runs on one bee crop):

      0. Upscale to working resolution (1.5× target)
      1. Multi-Scale Retinex      – illumination normalisation [new]
      2. Adaptive gamma           – lift dark crops to ~47 % grey
      3. NL-means denoising       – sensor noise removal (adaptive h)
      4. Two-pass adaptive CLAHE  – smooth local contrast boost
      5. SNN filter (win=3)       – symmetric nearest-neighbour edge-preserving
      6. Multi-scale detail boost – Laplacian 2-level decomposition
      7. Edge-adaptive USM        – sharpen structure, leave flats quiet
      8. Percentile stretch       – full dynamic range utilisation
      9. Resize to target
    """
    h_c, w_c = crop_gray.shape
    scale    = (target * 1.2) / min(h_c, w_c)   # 1.2× headroom (was 1.5)
    nw       = max(target, int(w_c * scale))
    nh       = max(target, int(h_c * scale))
    work     = cv2.resize(crop_gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 1 : Multi-Scale Retinex (blended 40% to avoid halo artefacts) ────
    msr  = multi_scale_retinex(work, sigmas=(15, 80, 250))
    work = np.clip(0.75 * work.astype(np.float32) +
                   0.25 * msr.astype(np.float32), 0, 255).astype(np.uint8)

    # ── Step 2 : Adaptive gamma ───────────────────────────────────────────────
    mean_val = float(work.mean())
    if mean_val > 5:
        gamma = np.log(118.0 / 255.0) / np.log(max(mean_val / 255.0, 1e-6))
        gamma = float(np.clip(gamma, 0.5, 1.8))
        lut   = np.array([((i / 255.0) ** gamma) * 255.0
                          for i in range(256)], dtype=np.uint8)
        work  = cv2.LUT(work, lut)

    # ── Step 3 : NL-means denoising ───────────────────────────────────────────
    noise_est = float(cv2.Laplacian(work, cv2.CV_64F).std())
    nlm_h     = int(np.clip(noise_est * 0.35, 4, 9))
    work      = cv2.fastNlMeansDenoising(work, h=nlm_h,
                                          templateWindowSize=7,
                                          searchWindowSize=13)

    # ── Step 4 : Two-pass adaptive CLAHE (gentle) ────────────────────────────
    std_val = float(work.std())
    clip1   = float(np.clip(1.8 * max(1.0, 8.0 / (std_val + 1e-3)), 1.2, 3.5))
    work    = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(8, 8)).apply(work)
    work    = cv2.createCLAHE(clipLimit=1.0,   tileGridSize=(4, 4)).apply(work)

    # ── Step 5 : SNN filter (snnV2.m port) ───────────────────────────────────
    f    = work.astype(np.float64) / 255.0
    f    = snn_filter(f, win_sz=3)
    work = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # ── Step 6 : Multi-scale Laplacian detail boost (tuned down) ─────────────
    img_f        = work.astype(np.float64)
    base_coarse  = cv2.GaussianBlur(img_f, (0, 0), 5.0)
    base_fine    = cv2.GaussianBlur(img_f, (0, 0), 1.5)
    detail_mid   = base_fine - base_coarse
    detail_fine  = img_f     - base_fine
    img_f        = base_coarse + detail_mid * 1.35 + detail_fine * 1.10
    work         = np.clip(img_f, 0, 255).astype(np.uint8)

    # ── Step 7 : Edge-adaptive unsharp masking (gentler) ────────────────────
    work = edge_adaptive_usm(work, amount=1.1, sigma=1.4, edge_boost=1.8)

    # ── Step 8 : Percentile stretch ───────────────────────────────────────────
    p_lo, p_hi = np.percentile(work, 1), np.percentile(work, 99)
    if p_hi - p_lo > 10:
        work = np.clip(
            (work.astype(np.float64) - p_lo) / (p_hi - p_lo) * 245.0 + 5.0,
            0, 255).astype(np.uint8)

    # ── Step 9 : Final resize ─────────────────────────────────────────────────
    return cv2.resize(work, (target, target), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (honest before: CLAHE only)
# ─────────────────────────────────────────────────────────────────────────────

def baseline(crop_gray, target=256):
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(crop_gray)
    return cv2.resize(enhanced, (target, target), interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison tile builder
# ─────────────────────────────────────────────────────────────────────────────

HEADER  = 30
DIVIDER = 3

def make_comparison_tile(before, after, label, tw=256, th=256):
    W    = tw * 2 + DIVIDER
    H    = th + HEADER
    tile = np.full((H, W, 3), 18, dtype=np.uint8)

    bef  = cv2.resize(before, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    aft  = cv2.resize(after,  (tw, th), interpolation=cv2.INTER_LANCZOS4)

    tile[HEADER:, :tw]                 = cv2.cvtColor(bef, cv2.COLOR_GRAY2BGR)
    tile[HEADER:, tw + DIVIDER:]       = cv2.cvtColor(aft, cv2.COLOR_GRAY2BGR)
    tile[HEADER:, tw:tw + DIVIDER]     = (60, 60, 60)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tile, "BEFORE", (5, HEADER - 7),              font, 0.42, (130, 130, 130), 1)
    cv2.putText(tile, "AFTER",  (tw + DIVIDER + 5, HEADER - 7), font, 0.42, (140, 220, 140), 1)
    cv2.putText(tile, label,    (tw // 2 - 30, HEADER - 7),   font, 0.40, (240, 200, 80),  1)
    return tile


# ─────────────────────────────────────────────────────────────────────────────
# Montage builder
# ─────────────────────────────────────────────────────────────────────────────

def build_montage(tiles, cols=7, gap=4):
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows   = (len(tiles) + cols - 1) // cols
    canvas = np.full(
        (rows * (th + gap) + gap, cols * (tw + gap) + gap, 3),
        10, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0   = gap + r * (th + gap)
        x0   = gap + c * (tw + gap)
        canvas[y0:y0 + th, x0:x0 + tw] = t
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

TARGET       = 256
MONTAGE_COLS = 8
comparison_tiles = []

print(f"\nEnhancing {len(bees)} bee crops …")
for idx, bee in enumerate(bees):
    cx, cy = bee["cx"], bee["cy"]
    y1 = max(0,     cy - CROP_HALF)
    y2 = min(img_h, cy + CROP_HALF)
    x1 = max(0,     cx - CROP_HALF)
    x2 = min(img_w, cx + CROP_HALF)

    crop_src = original[y1:y2, x1:x2]
    if crop_src.size == 0:
        continue

    bef = baseline(crop_src, target=TARGET)
    aft = enhance(crop_src,  target=TARGET)

    stem = f"bee_{idx+1:03d}_bee{bee['bee_frac']:.2f}"
    cv2.imwrite(f"{OUT_ROOT}/after/{stem}.png", aft)

    label = f"#{idx+1:03d} {bee['bee_frac']:.0%}"
    tile  = make_comparison_tile(bef, aft, label, tw=TARGET, th=TARGET)
    cv2.imwrite(f"{OUT_ROOT}/compare/{stem}_cmp.png", tile)
    comparison_tiles.append(tile)

    if (idx + 1) % 30 == 0 or (idx + 1) == len(bees):
        print(f"  {idx+1:3d}/{len(bees)}  bee_frac={bee['bee_frac']:.0%}  "
              f"center=({cx},{cy})")

# ── Montage ───────────────────────────────────────────────────────────────────
print(f"\nBuilding montage ({len(comparison_tiles)} tiles, {MONTAGE_COLS} cols) …")
montage = build_montage(comparison_tiles, cols=MONTAGE_COLS, gap=4)
if montage is not None:
    mpath = f"{OUT_ROOT}/montage.png"
    cv2.imwrite(mpath, montage)
    mh, mw = montage.shape[:2]
    print(f"Montage → {mpath}  ({mw}×{mh} px)")

print(f"\nDone.")
print(f"  Individual : {OUT_ROOT}/after/")
print(f"  Comparisons: {OUT_ROOT}/compare/")
print(f"  Montage    : {OUT_ROOT}/montage.png")
