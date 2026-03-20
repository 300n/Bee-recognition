import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# Dépendances supplémentaires :
#   pip install bm3d PyWavelets
import bm3d
import pywt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/comparaison_filtres_MC02"

# ============================================================
#  FILTRES
# ============================================================


def filter_none(img):
    return img


def filter_gaussian(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def filter_median(img):
    return cv2.medianBlur(img, 5)


def filter_bilateral(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def filter_nlm(img):
    return cv2.fastNlMeansDenoising(
        img, None, h=10, templateWindowSize=7, searchWindowSize=21
    )


def filter_bm3d(img):
    img_float = img.astype(np.float64) / 255.0
    denoised = bm3d.bm3d(img_float, sigma_psd=15 / 255.0)
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


def filter_wavelet(img):
    img_float = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_float, wavelet="db4", level=3)
    coeffs_t = [coeffs[0]]
    for detail in coeffs[1:]:
        sigma = np.median(np.abs(detail[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(img_float.size))
        coeffs_t.append(
            tuple(pywt.threshold(sub, value=threshold, mode="soft") for sub in detail)
        )
    denoised = pywt.waverec2(coeffs_t, wavelet="db4")
    denoised = denoised[: img.shape[0], : img.shape[1]]
    return np.clip(denoised, 0, 255).astype(np.uint8)


# ============================================================
#  REGISTRE
# ============================================================
FILTERS = [
    ("Sans filtre (référence)", filter_none),
    ("Gaussien (k=5)", filter_gaussian),
    ("Médian (k=5)", filter_median),
    ("Bilatéral (d=9, σ=75)", filter_bilateral),
    ("NLM (h=10)", filter_nlm),
    ("BM3D (σ=15)", filter_bm3d),
    ("Ondelettes (db4, lvl=3)", filter_wavelet),
]

# ============================================================
#  GRILLE
# ============================================================


def create_grid(img_gray, output_dir, base_name):
    n = len(FILTERS)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols  # 2 lignes pour 7 filtres

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 7))
    axes_flat = axes.flatten()

    fig.suptitle(
        f"Comparaison des filtres de débruitage — image brute\n{base_name}.png",
        fontsize=20,
        fontweight="bold",
        y=1.01,
    )

    for j, (label, fn) in enumerate(FILTERS):
        print(f"  Application : {label}...", end=" ", flush=True)
        t0 = time.time()
        out = fn(img_gray.copy())
        print(f"({(time.time()-t0)*1000:.0f} ms)")

        ax = axes_flat[j]
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            label, fontsize=14, pad=10, fontweight="bold" if j == 0 else "normal"
        )
        ax.axis("off")

        # Encadré rouge sur la référence
        if j == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cc3333")
                spine.set_linewidth(3)
                spine.set_visible(True)

    # Masquer les sous-plots vides (si n % n_cols != 0)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"GRILLE_FILTRES_{base_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grille sauvegardée : {out_path}")
    plt.close()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur : impossible de charger {INPUT_IMAGE_PATH}")
    else:
        base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
        print(f"Image chargée : {base_name}  ({img.shape[1]}×{img.shape[0]} px)\n")
        create_grid(img, OUTPUT_DIR, base_name)
        print("\nTerminé.")
