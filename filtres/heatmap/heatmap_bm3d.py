import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Dépendance : pip install bm3d
import bm3d

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_bm3d_MC02"
)

# Paramètres heatmap
R = 40
T = 30

# Paramètres BM3D
# sigma_psd est l'écart-type estimé du bruit (en niveaux de gris, échelle 0-255).
# C'est le seul paramètre vraiment important de BM3D.
# Règle pratique :
#   - sigma_psd = 5-10  : bruit très faible (images quasi propres, légère compression PNG)
#   - sigma_psd = 15-25 : bruit modéré (caméra de ruche en conditions normales)
#   - sigma_psd = 30-50 : bruit fort (mauvais éclairage, capteur bas de gamme)
# Si tu ne connais pas le bruit de ta caméra, commence à sigma_psd=15 et ajuste
# selon le résultat : trop lisse → descends, grain visible → monte.
SIGMA_PSD = 15

# ============================================================
#  DÉBRUITAGE BM3D
# ============================================================
# BM3D (Block Matching and 3D filtering) — Dabov et al., IEEE TIP 2007
# Considéré comme l'état de l'art classique du débruitage (référence de facto).
#
# Principe en deux étapes :
#
# ÉTAPE 1 — Estimation de base (hard thresholding)
#   Pour chaque bloc de l'image, BM3D cherche dans toute l'image les blocs qui
#   lui ressemblent (block matching). Ces blocs similaires sont empilés en un
#   volume 3D. Une transformée 3D (DCT ou ondelettes) est appliquée sur ce volume :
#   les coefficients petits (= bruit) sont mis à zéro (seuillage dur), les grands
#   (= signal) sont conservés. On obtient une première image débruitée.
#
# ÉTAPE 2 — Estimation finale (filtre de Wiener collaboratif)
#   On refait un block matching sur la première image débruitée (plus précis).
#   Au lieu du seuillage dur, on applique un filtre de Wiener collaboratif :
#   chaque coefficient 3D est atténué proportionnellement à son rapport signal/bruit
#   estimé. Résultat : les détails fins (contours des abeilles) sont bien mieux
#   préservés que par un simple lissage gaussien.
#
# Avantage clé vs filtres classiques :
#   Le NLM et le gaussien lissent localement. BM3D exploite la REDONDANCE GLOBALE
#   de l'image : deux abeilles similaires dans des coins opposés de l'image
#   s'aident mutuellement à se débruiter. C'est particulièrement puissant pour
#   des ruches où les corps d'abeilles se ressemblent visuellement.
#
# Référence : K. Dabov et al., "Image denoising by sparse 3D transform-domain
#             collaborative filtering", IEEE TIP, vol. 16, no. 8, 2007.


def apply_bm3d(img_gray, sigma_psd=15):
    """
    Applique le filtre BM3D sur une image en niveaux de gris.

    Paramètres
    ----------
    img_gray  : image uint8 (0-255)
    sigma_psd : écart-type estimé du bruit (0-255).
                Plus grand = débruitage plus agressif.

    Retourne
    --------
    Image débruitée uint8 (0-255)
    """
    # BM3D travaille sur des valeurs normalisées [0.0, 1.0]
    img_float = img_gray.astype(np.float64) / 255.0
    sigma_norm = sigma_psd / 255.0

    denoised = bm3d.bm3d(img_float, sigma_psd=sigma_norm)

    # Remise sur 0-255 avec clipping pour éviter les débordements
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return denoised


# ============================================================
#  PIPELINE HEATMAP
# ============================================================


def generate_and_save_heatmap(
    img_gray, r, threshold_val, output_folder, base_name, sigma_psd=15
):
    """
    Pipeline heatmap avec prétraitement BM3D avant la FFT.
    L'overlay final est rendu sur l'image originale non modifiée.
    """
    # Étape 0 : Débruitage BM3D
    img_processed = apply_bm3d(img_gray, sigma_psd)

    rows, cols = img_processed.shape
    crow, ccol = rows // 2, cols // 2

    # Étape 1 : FFT + masque passe-bas
    dft = cv2.dft(np.float32(img_processed), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Étape 2 : Heatmap
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_raw = 255 - img_back.astype(np.uint8)
    _, heatmap_t = cv2.threshold(heatmap_raw, threshold_val, 255, cv2.THRESH_TOZERO)
    heatmap_color = cv2.applyColorMap(heatmap_t, cv2.COLORMAP_JET)

    # Étape 3 : Overlay sur l'image ORIGINALE (pas sur l'image débruitée)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    filename = f"heatmap_bm3d_sigma={sigma_psd}_r={r}_t={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename


# ============================================================
#  GRID SEARCH SUR sigma_psd
# ============================================================
SIGMA_VALUES = [5, 15, 25, 40]


def create_summary_grid(output_folder, base_name, sigma_vals, r, t):
    """Grille récapitulative des différents niveaux de débruitage BM3D."""
    n = len(sigma_vals)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    fig.suptitle(
        f"Grid Search BM3D : sigma_psd  (r={r}, t={t})\nImage : {base_name}.png",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Colonne 0 : référence sans débruitage
    ref_path = os.path.join(
        output_folder, f"heatmap_bm3d_sigma=0_r={r}_t={t}_{base_name}.png"
    )
    if os.path.exists(ref_path):
        axes[0].imshow(cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB))
    axes[0].set_title("Sans BM3D\n(référence)", fontsize=11, pad=8)
    axes[0].axis("off")

    for j, sigma in enumerate(sigma_vals):
        filename = f"heatmap_bm3d_sigma={sigma}_r={r}_t={t}_{base_name}.png"
        filepath = os.path.join(output_folder, filename)
        if os.path.exists(filepath):
            axes[j + 1].imshow(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))
        axes[j + 1].set_title(f"σ = {sigma}", fontsize=11, pad=8)
        axes[j + 1].axis("off")

    plt.tight_layout()
    summary_path = os.path.join(output_folder, f"SUMMARY_BM3D_{base_name}.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"Grille sauvegardée : {summary_path}")
    plt.close()


# ============================================================
#  MAIN
# ============================================================


def generate_no_filter_reference(img_gray, r, t, output_folder, base_name):
    """Génère la heatmap de référence sans débruitage pour comparaison."""
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_raw = 255 - img_back.astype(np.uint8)
    _, heatmap_t = cv2.threshold(heatmap_raw, t, 255, cv2.THRESH_TOZERO)
    heatmap_color = cv2.applyColorMap(heatmap_t, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(
        os.path.join(
            output_folder, f"heatmap_bm3d_sigma=0_r={r}_t={t}_{base_name}.png"
        ),
        final_overlay,
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erreur : impossible de charger {INPUT_IMAGE_PATH}")
    else:
        base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
        print(f"Image chargée : {base_name} ({img.shape[1]}×{img.shape[0]}px)\n")

        print("--- Génération de la référence sans filtre ---")
        generate_no_filter_reference(img, R, T, OUTPUT_DIR, base_name)

        print("\n--- Grid Search BM3D (sigma_psd) ---")
        print(
            "Note : BM3D est plus lent que les filtres classiques (~2-10s par image)."
        )
        for sigma in SIGMA_VALUES:
            print(f"  Traitement sigma={sigma}...", end=" ", flush=True)
            fname = generate_and_save_heatmap(
                img, R, T, OUTPUT_DIR, base_name, sigma_psd=sigma
            )
            print(f"OK → {fname}")

        print("\n--- Création de la grille récapitulative ---")
        create_summary_grid(OUTPUT_DIR, base_name, SIGMA_VALUES, R, T)
        print("\nTerminé.")
