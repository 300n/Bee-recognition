import cv2
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
IMG_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_64/M01C02_000338-3.png"
OUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/figures"
os.makedirs(OUT_DIR, exist_ok=True)

img_gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
assert img_gray is not None, f"Could not load {IMG_PATH}"
print(f"[load] {img_gray.shape[1]}×{img_gray.shape[0]} grayscale")


# ── Helper ────────────────────────────────────────────────────────────────────
def save_side_by_side(
    original, processed, filename, title_orig="Original", title_proc="Processed"
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for ax, im, title in zip(axes, [original, processed], [title_orig, title_proc]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
    plt.tight_layout(pad=0.5)
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# CONTRAST METHODS
# ══════════════════════════════════════════════════════════════════════════════


def apply_clahe(img, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img)


def apply_msr(img, sigmas=(15, 80, 250)):
    """Multi-Scale Retinex (grayscale)."""
    img_f = img.astype(np.float32) + 1.0  # avoid log(0)
    log_img = np.log(img_f)
    retinex = np.zeros_like(img_f)
    for s in sigmas:
        # Use a kernel size large enough for the sigma
        ksize = int(6 * s + 1) | 1  # odd, ≥ 6σ
        blur = cv2.GaussianBlur(img_f, (ksize, ksize), s)
        retinex += log_img - np.log(blur + 1.0)
    retinex /= len(sigmas)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# DENOISING FILTERS
# ══════════════════════════════════════════════════════════════════════════════


def apply_bilateral(img, d=110, sc=20, ss=50):
    return cv2.bilateralFilter(img, d, sc, ss)


def apply_nlm(img, h=7, template=7, search=21):  # h=7 optimal: edge_pres=0.975 vs h=10→0.916
    return cv2.fastNlMeansDenoising(
        img, h=h, templateWindowSize=template, searchWindowSize=search
    )


def apply_bm3d(img, sigma_psd=10 / 255):  # σ≈9/255 estimated via MAD on this dataset
    import bm3d

    img_f = img.astype(np.float32) / 255.0
    denoised = bm3d.bm3d(img_f, sigma_psd=sigma_psd)
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def _snn_core(img_f, radius, center_weight=1):
    """Shared core for SNN and Kramer-Brückner."""
    h, w = img_f.shape
    pad = cv2.copyMakeBorder(img_f, radius, radius, radius, radius, cv2.BORDER_REFLECT)

    # Build symmetric pairs (each offset paired with its opposite)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]
    pairs, seen = [], set()
    for dy, dx in offsets:
        if (-dy, -dx) not in seen:
            pairs.append(((dy, dx), (-dy, -dx)))
            seen.add((dy, dx))

    # Accumulator starts at center_weight × center
    accum = img_f * float(center_weight)
    count = np.full_like(img_f, float(center_weight))

    for (dy1, dx1), (dy2, dx2) in pairs:
        a = pad[radius + dy1 : radius + dy1 + h, radius + dx1 : radius + dx1 + w]
        b = pad[radius + dy2 : radius + dy2 + h, radius + dx2 : radius + dx2 + w]
        closer = np.where(np.abs(a - img_f) <= np.abs(b - img_f), a, b)
        accum += closer
        count += 1.0

    return np.clip(accum / count, 0, 255).astype(np.uint8)


def apply_snn(img, radius=2):
    return _snn_core(img.astype(np.float32), radius, center_weight=0)
    # SNN: no special weight for center — average of selected neighbours only
    # (re-implement cleanly below)


def apply_snn(img, radius=2):
    h, w = img.shape
    img_f = img.astype(np.float32)
    pad = cv2.copyMakeBorder(img_f, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]
    pairs, seen = [], set()
    for dy, dx in offsets:
        if (-dy, -dx) not in seen:
            pairs.append(((dy, dx), (-dy, -dx)))
            seen.add((dy, dx))

    accum = np.zeros_like(img_f)
    count = np.zeros_like(img_f)
    for (dy1, dx1), (dy2, dx2) in pairs:
        a = pad[radius + dy1 : radius + dy1 + h, radius + dx1 : radius + dx1 + w]
        b = pad[radius + dy2 : radius + dy2 + h, radius + dx2 : radius + dx2 + w]
        closer = np.where(np.abs(a - img_f) <= np.abs(b - img_f), a, b)
        accum += closer
        count += 1.0
    return np.clip(accum / count, 0, 255).astype(np.uint8)


def apply_kramer_bruckner(img, radius=2):
    """SNN with centre pixel double-weighted (Kramer & Brückner 1975)."""
    h, w = img.shape
    img_f = img.astype(np.float32)
    pad = cv2.copyMakeBorder(img_f, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]
    pairs, seen = [], set()
    for dy, dx in offsets:
        if (-dy, -dx) not in seen:
            pairs.append(((dy, dx), (-dy, -dx)))
            seen.add((dy, dx))

    accum = img_f.copy()  # centre contributes once
    count = np.ones_like(img_f)  # → effectively double-weighted vs neighbours
    for (dy1, dx1), (dy2, dx2) in pairs:
        a = pad[radius + dy1 : radius + dy1 + h, radius + dx1 : radius + dx1 + w]
        b = pad[radius + dy2 : radius + dy2 + h, radius + dx2 : radius + dx2 + w]
        closer = np.where(np.abs(a - img_f) <= np.abs(b - img_f), a, b)
        accum += closer
        count += 1.0
    return np.clip(accum / count, 0, 255).astype(np.uint8)


def apply_epoaf(img, offsets=(-2, -1, 1, 2)):
    """Edge-Preserving Oriented Adaptive Filter — smooths along edge tangent."""
    img_f = img.astype(np.float32)
    h, w = img.shape

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6

    # Tangent = perpendicular to gradient: (-gy/mag, gx/mag)
    tx = -gy / mag  # tangent x component
    ty = gx / mag  # tangent y component

    xs = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    accum = img_f.copy()
    n = 1
    for s in offsets:
        map_x = np.clip(xs + s * tx, 0, w - 1).astype(np.float32)
        map_y = np.clip(ys + s * ty, 0, h - 1).astype(np.float32)
        sampled = cv2.remap(
            img_f, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        accum += sampled
        n += 1
    return np.clip(accum / n, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Contrast methods ──")
img_orig = img_gray
img_clahe = apply_clahe(img_gray)
print("  CLAHE done")
img_retinex = apply_msr(img_gray)
print("  MSR done")

contrast_variants = [
    ("Original", img_orig),
    ("CLAHE", img_clahe),
    ("Retinex", img_retinex),
]

print("\n── Filters (applied to each contrast variant) ──")
filter_fns = [
    ("Bilateral", apply_bilateral),
    ("NLM", apply_nlm),
    ("BM3D", apply_bm3d),
    ("SNN", apply_snn),
    ("Kramer-Brückner", apply_kramer_bruckner),
    ("EPOAF", apply_epoaf),
]

# Pre-compute all 18 combinations
grid_imgs = {}
for c_name, c_img in contrast_variants:
    for f_name, f_fn in filter_fns:
        key = (c_name, f_name)
        grid_imgs[key] = f_fn(c_img)
        print(f"  [{c_name} + {f_name}] done")


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL FIGURES (contrast)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Saving individual figures ──")

save_side_by_side(
    img_orig,
    img_orig,
    "pipeline_contrast_original.png",
    "Original",
    "Original (no modification)",
)

save_side_by_side(
    img_orig,
    img_clahe,
    "pipeline_contrast_clahe.png",
    "Original",
    "CLAHE  (clip=2.0, 8×8 tiles)",
)

save_side_by_side(
    img_orig,
    img_retinex,
    "pipeline_contrast_retinex.png",
    "Original",
    "Multi-Scale Retinex  (σ∈{15,80,250})",
)

# ── Individual filter figures (applied to original) ───────────────────────────
save_side_by_side(
    img_orig,
    grid_imgs[("Original", "Bilateral")],
    "pipeline_filter_bilateral.png",
    "Original",
    "Bilateral  (d=110, ss=20, sc=50)",
)

save_side_by_side(
    img_orig,
    grid_imgs[("Original", "NLM")],
    "pipeline_filter_nlm.png",
    "Original",
    "Non-Local Means  (h=7, 7×7, 21×21)",
)

save_side_by_side(
    img_orig,
    grid_imgs[("Original", "BM3D")],
    "pipeline_filter_bm3d.png",
    "Original",
    "BM3D  (σ=10/255)",
)

save_side_by_side(
    img_orig,
    grid_imgs[("Original", "SNN")],
    "pipeline_filter_snn.png",
    "Original",
    "SNN  (r=2)",
)

save_side_by_side(
    img_orig,
    grid_imgs[("Original", "Kramer-Brückner")],
    "pipeline_filter_kramer_bruckner.png",
    "Original",
    "Kramer-Brückner  (r=2, centre double-weighted)",
)

save_side_by_side(
    img_orig,
    grid_imgs[("Original", "EPOAF")],
    "pipeline_filter_epoaf.png",
    "Original",
    "EPOAF  (S={-2,-1,+1,+2})",
)


# ══════════════════════════════════════════════════════════════════════════════
# 3×6 GRID FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Saving 3×6 pipeline grid ──")

col_labels = [f_name for f_name, _ in filter_fns]
row_labels = [c_name for c_name, _ in contrast_variants]

fig = plt.figure(figsize=(22, 11), dpi=150)
gs = gridspec.GridSpec(
    3,
    6,
    figure=fig,
    hspace=0.08,
    wspace=0.05,
    left=0.07,
    right=0.99,
    top=0.93,
    bottom=0.02,
)

for i, (c_name, _) in enumerate(contrast_variants):
    for j, (f_name, _) in enumerate(filter_fns):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(grid_imgs[(c_name, f_name)], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if i == 0:
            ax.set_title(col_labels[j], fontsize=9, fontweight="bold", pad=4)
        if j == 0:
            ax.set_ylabel(
                row_labels[i], fontsize=9, fontweight="bold", rotation=90, labelpad=6
            )

fig.suptitle(
    "18 Preprocessing Pipeline Combinations  —  "
    "Rows: contrast step · Columns: denoising filter",
    fontsize=11,
    fontweight="bold",
    y=0.97,
)

path = os.path.join(OUT_DIR, "pipeline_grid.png")
fig.savefig(path, bbox_inches="tight")
plt.close()
print("  saved pipeline_grid.png")

print("\nAll figures saved to:", OUT_DIR)
