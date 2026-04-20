#!/usr/bin/env python3
"""
test_kb_filter.py
=================
Tests the Kramer & Brückner (1975) iterative non-linear contrast-enhancement
filter on image.png to evaluate suitability for bee detection.

Filter rule (per pixel, per iteration):
    midpoint D = (local_MAX + local_MIN) / 2 of NxN neighbourhood
    if pixel > D  →  replace with neighbourhood MAX
    if pixel <= D →  replace with neighbourhood MIN

This iteratively drives pixels toward local extremes, sharpening edges and
boosting contrast between dark bees and lighter honeycomb background.

Outputs (pipeline_output/kb_test/):
    kb_iter_01.png  …  kb_iter_10.png   – result after each iteration
    compare_grid.png                     – input + iter 1,3,5,10 side-by-side
    diff_iter10.png                      – absolute difference from original
    kb_iter10_thresh.png                 – Otsu threshold on iter-10 result
                                           (proxy for how cleanly bees separate)
Multi-scale schedule variant (new):
    Instead of a fixed window, a schedule of increasing window sizes is applied
    across iterations (e.g. 3×3 → 5×5 → 7×7 → 9×9 → 11×11, 2 passes each).
    Early passes sharpen fine structure (honeycomb cell edges, ~20-25px);
    later passes push bee-body vs background contrast (~50-80px structures).

    Additional outputs (pipeline_output/kb_test/ms_*):
        ms_step_*.png          – result after each schedule step
        ms_compare_grid.png    – original | fixed-10 | ms final side-by-side
        ms_otsu_compare.png    – Otsu on original | fixed iter10 | ms final
        ms_diff.png            – absolute difference ms-final vs original
"""

import cv2
import numpy as np
import os

INPUT = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
OUT = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_kramer_bruckner"
os.makedirs(OUT, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────

src = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
if src is None:
    raise FileNotFoundError("image.png not found in working directory")

print(
    f"[load] image.png  {src.shape[1]}×{src.shape[0]}  "
    f"mean={src.mean():.1f}  std={src.std():.1f}"
)


# ── Kramer & Brückner filter ──────────────────────────────────────────────────


def kb_filter(img_u8, win=3):
    """
    One pass of the Kramer-Brückner non-linear filter.

    Parameters
    ----------
    img_u8 : uint8 ndarray
    win    : neighbourhood half-size; kernel = (2*win+1) x (2*win+1)
             win=1 → 3x3,  win=2 → 5x5
    """
    img_f = img_u8.astype(np.float32)
    k = 2 * win + 1

    # Local min, max via morphological operations
    local_min = cv2.erode(img_f, np.ones((k, k), np.uint8))
    local_max = cv2.dilate(img_f, np.ones((k, k), np.uint8))

    # Midpoint threshold D = (max + min) / 2  (Kramer & Brückner original)
    midpoint = (local_max + local_min) / 2.0

    out = local_min.copy()  # default: pixel <= midpoint → min
    out[img_f > midpoint] = local_max[img_f > midpoint]  # pixel > midpoint → max

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Run iterations and save individual results ────────────────────────────────

ITERS = 10
WIN_SIZE = 1  # 3×3 neighbourhood — keeps local structure intact
SAVE_AT = {1, 3, 5, 10}

current = src.copy()
saved = {}

for i in range(1, ITERS + 1):
    current = kb_filter(current, win=WIN_SIZE)
    path = f"{OUT}/kb_iter_{i:02d}.png"
    cv2.imwrite(path, current)
    print(f"[iter {i:2d}] mean={current.mean():.1f}  std={current.std():.1f}  → {path}")
    if i in SAVE_AT:
        saved[i] = current.copy()


# ── Comparison grid: original | iter 1 | iter 3 | iter 5 | iter 10 ───────────

cols = [src] + [saved[k] for k in sorted(saved)]
labels = ["Original"] + [f"KB iter {k}" for k in sorted(saved)]

PAD = 4
LHEIGHT = 22
tiles = []
for img, lbl in zip(cols, labels):
    h, w = img.shape
    # label bar
    bar = np.full((LHEIGHT, w), 30, dtype=np.uint8)
    cv2.putText(
        bar, lbl, (4, LHEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1, cv2.LINE_AA
    )
    tile = np.vstack([bar, img])
    tiles.append(tile)

divider = np.full((tiles[0].shape[0], PAD), 80, dtype=np.uint8)
grid = tiles[0]
for t in tiles[1:]:
    grid = np.hstack([grid, divider, t])

cv2.imwrite(f"{OUT}/compare_grid.png", grid)
print(f"[grid] {OUT}/compare_grid.png")


# ── Absolute difference: iter-10 vs original ─────────────────────────────────

diff = cv2.absdiff(saved[10], src)
# stretch for visibility
diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(f"{OUT}/diff_iter10.png", diff_vis)
print(
    f"[diff] max_change={int(diff.max())}  mean_change={diff.mean():.1f}  → {OUT}/diff_iter10.png"
)


# ── Otsu threshold on iter-10 result (bee segmentation quality proxy) ─────────

_, thresh_orig = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, thresh_kb10 = cv2.threshold(
    saved[10], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# side-by-side thresh comparison
bar_o = np.full((LHEIGHT, src.shape[1]), 30, dtype=np.uint8)
bar_k = np.full((LHEIGHT, saved[10].shape[1]), 30, dtype=np.uint8)
cv2.putText(
    bar_o, "Otsu — Original", (4, LHEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1
)
cv2.putText(
    bar_k, "Otsu — KB iter 10", (4, LHEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1
)

thresh_compare = np.hstack(
    [
        np.vstack([bar_o, thresh_orig]),
        np.full((src.shape[0] + LHEIGHT, PAD), 80, dtype=np.uint8),
        np.vstack([bar_k, thresh_kb10]),
    ]
)
cv2.imwrite(f"{OUT}/otsu_compare.png", thresh_compare)

n_orig = int((thresh_orig > 0).sum())
n_kb10 = int((thresh_kb10 > 0).sum())
print(
    f"[otsu] foreground px — original: {n_orig:,}  KB iter10: {n_kb10:,}  "
    f"delta: {n_kb10 - n_orig:+,}  ({(n_kb10-n_orig)/n_orig*100:+.1f}%)"
)
print(f"[otsu] → {OUT}/otsu_compare.png")

print(f"\nDone. All outputs in {OUT}/")


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SCALE SCHEDULE VARIANT
# ═════════════════════════════════════════════════════════════════════════════

print("\n── Multi-scale K&B schedule ─────────────────────────────────────────────")

# Schedule: (window_half, n_passes)
#   win=1 → 3×3  targets ~3px features  (cell wall sharpening)
#   win=2 → 5×5  targets ~5px features  (cell interior / bee leg detail)
#   win=3 → 7×7  targets ~7px features  (bee head / thorax boundary)
#   win=4 → 9×9  targets ~9px features  (bee body mid-scale)
#   win=5 → 11×11 targets ~11px features (full bee-body vs background)
#
# 2 passes at each scale → 10 total (same count as fixed test for fair comparison)

MS_SCHEDULE = [
    (1, 2),  # 3×3  ×2
    (2, 2),  # 5×5  ×2
    (3, 2),  # 7×7  ×2
    (4, 2),  # 9×9  ×2
    (5, 2),  # 11×11 ×2
]

ms_current = src.copy()
ms_saved = {}  # keyed by cumulative step label
step_num = 0

for win, n_passes in MS_SCHEDULE:
    k = 2 * win + 1
    for p in range(n_passes):
        step_num += 1
        ms_current = kb_filter(ms_current, win=win)
    label = f"win{k}x{k}"
    ms_saved[label] = ms_current.copy()
    path = f"{OUT}/ms_step_{label}.png"
    cv2.imwrite(path, ms_current)
    print(
        f"[ms {label} ×{n_passes}] mean={ms_current.mean():.1f}  "
        f"std={ms_current.std():.1f}  → {path}"
    )

ms_final = ms_current


# ── Multi-scale comparison grid: original | fixed-10 | ms final ──────────────

ms_cols = [src, saved[10], ms_final]
ms_labels = ["Original", "Fixed 3×3 ×10", "MS schedule (3→11) ×10"]

ms_tiles = []
for img, lbl in zip(ms_cols, ms_labels):
    h, w = img.shape
    bar = np.full((LHEIGHT, w), 30, dtype=np.uint8)
    cv2.putText(
        bar, lbl, (4, LHEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 220, 1, cv2.LINE_AA
    )
    ms_tiles.append(np.vstack([bar, img]))

divider = np.full((ms_tiles[0].shape[0], PAD), 80, dtype=np.uint8)
ms_grid = ms_tiles[0]
for t in ms_tiles[1:]:
    ms_grid = np.hstack([ms_grid, divider, t])

cv2.imwrite(f"{OUT}/ms_compare_grid.png", ms_grid)
print(f"[ms grid] {OUT}/ms_compare_grid.png")


# ── Absolute difference: ms-final vs original ─────────────────────────────────

ms_diff = cv2.absdiff(ms_final, src)
ms_diff_vis = cv2.normalize(ms_diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite(f"{OUT}/ms_diff.png", ms_diff_vis)
print(
    f"[ms diff] max_change={int(ms_diff.max())}  mean_change={ms_diff.mean():.1f}"
    f"  → {OUT}/ms_diff.png"
)


# ── Otsu: original | fixed iter-10 | ms final — three-way comparison ─────────

_, thresh_ms = cv2.threshold(ms_final, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


def labeled_tile(img, text):
    bar = np.full((LHEIGHT, img.shape[1]), 30, dtype=np.uint8)
    cv2.putText(
        bar, text, (4, LHEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 220, 1, cv2.LINE_AA
    )
    return np.vstack([bar, img])


div = np.full((src.shape[0] + LHEIGHT, PAD), 80, dtype=np.uint8)
ms_otsu_compare = np.hstack(
    [
        labeled_tile(thresh_orig, "Otsu — Original"),
        div,
        labeled_tile(thresh_kb10, "Otsu — Fixed 3x3 x10"),
        div,
        labeled_tile(thresh_ms, "Otsu — MS schedule"),
    ]
)
cv2.imwrite(f"{OUT}/ms_otsu_compare.png", ms_otsu_compare)

n_ms = int((thresh_ms > 0).sum())
print(
    f"[ms otsu] foreground px — original: {n_orig:,}  "
    f"fixed-10: {n_kb10:,}  ms: {n_ms:,}"
)
print(
    f"          fixed delta: {n_kb10-n_orig:+,} ({(n_kb10-n_orig)/n_orig*100:+.1f}%)  "
    f"ms delta: {n_ms-n_orig:+,} ({(n_ms-n_orig)/n_orig*100:+.1f}%)"
)
print(f"[ms otsu] → {OUT}/ms_otsu_compare.png")

print(f"\nAll multi-scale outputs in {OUT}/ms_*")
