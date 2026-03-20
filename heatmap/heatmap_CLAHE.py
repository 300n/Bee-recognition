import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_clahe_MC02"
)

# Paramètres heatmap
R = 40
T = 30

# Paramètres CLAHE
CLIP_LIMIT = 2.0  # Seuil d'écrêtage : 2.0-4.0. Plus élevé = plus de contraste local,
# mais risque d'amplifier le bruit dans les zones uniformes (cire).
TILE_SIZE = 8  # Taille des tuiles en pixels (8-16). Plus petit = correction
# plus locale, mais plus sensible au bruit de capteur.

# ============================================================
#  FILTRE CLAHE
# ============================================================


def apply_clahe(img_gray, clip_limit=2.0, tile_size=8):
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalization

    Problème visé : faible contraste local des abeilles par rapport
    à leur voisinage immédiat (fond de cire de même teinte), et zones
    sur- ou sous-exposées selon la position dans le cadre.

    Principe :
      L'égalisation d'histogramme classique calcule une seule courbe de
      redistribution des niveaux de gris pour TOUTE l'image — inefficace
      quand l'éclairage est inégal.
      CLAHE divise l'image en tuiles (tile_size × tile_size), calcule une
      courbe de redistribution LOCALE pour chacune, puis les recolle avec
      une interpolation bilinéaire pour éviter les artéfacts aux jonctions.
      Le paramètre clip_limit écrête l'histogramme local pour empêcher
      d'amplifier le bruit dans les zones uniformes (fond de cire).

    Paramètres :
      clip_limit : seuil d'écrêtage (2.0-4.0).
      tile_size  : taille des tuiles (8-16 px).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


# ============================================================
#  PIPELINE HEATMAP
# ============================================================


def generate_and_save_heatmap(
    img_gray, r, threshold_val, output_folder, base_name, clip_limit=2.0, tile_size=8
):
    """
    Génère une heatmap avec prétraitement CLAHE avant la FFT.
    L'overlay final est rendu sur l'image originale non modifiée.
    """
    # Étape 0 : Prétraitement CLAHE
    img_processed = apply_clahe(img_gray, clip_limit, tile_size)

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

    # Étape 3 : Overlay sur l'image ORIGINALE (pas sur l'image prétraitée)
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    filename = f"heatmap_clahe_clip={clip_limit}_tile={tile_size}_r={r}_t={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename


# ============================================================
#  GRID SEARCH SUR LES PARAMÈTRES CLAHE
# ============================================================

CLIP_VALUES = [1.0, 2.0, 4.0]
TILE_VALUES = [4, 8, 16]


def create_summary_grid(output_folder, base_name, clip_vals, tile_vals, r, t):
    """Grille récapitulative clip_limit (lignes) × tile_size (colonnes)."""
    fig, axes = plt.subplots(
        nrows=len(clip_vals), ncols=len(tile_vals), figsize=(18, 18)
    )
    fig.suptitle(
        f"Grid Search CLAHE : clip_limit vs tile_size  (r={r}, t={t})\nImage : {base_name}.png",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    for i, clip in enumerate(clip_vals):
        for j, tile in enumerate(tile_vals):
            filename = (
                f"heatmap_clahe_clip={clip}_tile={tile}_r={r}_t={t}_{base_name}.png"
            )
            filepath = os.path.join(output_folder, filename)
            ax = axes[i, j]

            if os.path.exists(filepath):
                img_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(f"clip={clip} | tile={tile}", fontsize=13, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(
                    f"clip={clip}", fontsize=14, fontweight="bold", labelpad=8
                )
            if i == 0:
                ax.set_xlabel(
                    f"tile={tile}", fontsize=14, fontweight="bold", labelpad=8
                )
                ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    summary_path = os.path.join(output_folder, f"SUMMARY_CLAHE_{base_name}.png")
    plt.savefig(summary_path, dpi=150)
    print(f"Grille sauvegardée : {summary_path}")
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
        print(f"Image chargée : {base_name} ({img.shape[1]}×{img.shape[0]}px)\n")

        print("--- Étape 1 : Génération des images individuelles ---")
        for clip in CLIP_VALUES:
            for tile in TILE_VALUES:
                fname = generate_and_save_heatmap(
                    img, R, T, OUTPUT_DIR, base_name, clip_limit=clip, tile_size=tile
                )
                print(f"  Généré : {fname}")

        print("\n--- Étape 2 : Création de la grille récapitulative ---")
        create_summary_grid(OUTPUT_DIR, base_name, CLIP_VALUES, TILE_VALUES, R, T)
        print("\nTerminé.")
