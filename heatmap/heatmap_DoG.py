import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_dog_MC02"
)

# Paramètres heatmap
R = 40
T = 30

# Paramètres DoG
# Règle de calibration selon le diamètre apparent d'une abeille en pixels :
#   sigma1 ≈ diametre_abeille_px / 6
#   sigma2 ≈ diametre_abeille_px / 3   (soit sigma2 = 2 × sigma1)
# Exemple : abeilles de ~30px → sigma1=5, sigma2=10
#           abeilles de ~15px → sigma1=2, sigma2=4
SIGMA1 = 2  # Écart-type du flou fin (limite basse du passe-bande)
SIGMA2 = 4  # Écart-type du flou large (limite haute du passe-bande)

# ============================================================
#  FILTRE DoG
# ============================================================


def apply_dog(img_gray, sigma1=2, sigma2=4):
    """
    DoG — Difference of Gaussians

    Problème visé : fond texturé (alvéoles, cire) et structures parasites
    de tailles diverses qui polluent le spectre FFT.

    Principe :
      On applique deux flous gaussiens d'écarts-types différents
      (sigma1 < sigma2), puis on les soustrait :
          DoG = GaussianBlur(sigma1) - GaussianBlur(sigma2)

      Cela crée un FILTRE PASSE-BANDE spatial :
        - Basses fréquences (fond lent, gradient d'éclairage) :
          présentes dans LES DEUX flous → leur différence → 0 → supprimées.
        - Hautes fréquences (bruit grain, poussières) :
          absentes des deux flous → supprimées aussi.
        - Fréquences intermédiaires (corps des abeilles, taille ∈ [σ1, σ2]) :
          présentes dans Blur(σ1) mais pas dans Blur(σ2) → conservées.

      Le DoG est mathématiquement une approximation du Laplacien d'une
      Gaussienne (LoG), ce qui en fait un détecteur de blobs : il répond
      fort aux structures circulaires dont le rayon correspond à σ1-σ2.

    Calibration :
      Mesurer le diamètre d'une abeille en pixels sur tes images, puis :
          sigma1 ≈ diametre_px / 6
          sigma2 ≈ diametre_px / 3  (rapport sigma2/sigma1 = 2, standard)
      Le rapport peut monter jusqu'à 3 si les abeilles sont très serrées
      et qu'on veut une réponse plus sélective.

    Paramètres :
      sigma1 : écart-type du flou fin (limite basse du passe-bande).
      sigma2 : écart-type du flou large (limite haute du passe-bande).
    """
    blur1 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma2)
    dog = blur1 - blur2
    # La soustraction peut produire des valeurs négatives → remise sur 0-255
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog.astype(np.uint8)


# ============================================================
#  PIPELINE HEATMAP
# ============================================================


def generate_and_save_heatmap(
    img_gray, r, threshold_val, output_folder, base_name, sigma1=2, sigma2=4
):
    """
    Génère une heatmap avec prétraitement DoG avant la FFT.
    L'overlay final est rendu sur l'image originale non modifiée.

    Note : avec le DoG, tu peux augmenter le rayon r du masque FFT
    (ex. 40 → 60) sans perdre en précision, car le DoG a déjà supprimé
    les grandes structures parasites en amont.
    """
    # Étape 0 : Prétraitement DoG
    img_processed = apply_dog(img_gray, sigma1, sigma2)

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

    filename = (
        f"heatmap_dog_s1={sigma1}_s2={sigma2}_r={r}_t={threshold_val}_{base_name}.png"
    )
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename


# ============================================================
#  GRID SEARCH SUR LES PARAMÈTRES DoG
# ============================================================
# Lignes : sigma1 (limite basse du passe-bande)
# Colonnes : sigma2 (limite haute du passe-bande)
# Conseil : garde toujours sigma2 = 2× ou 3× sigma1.
#           Les paires hors de cette plage produisent un passe-bande
#           trop étroit (peu de signal) ou trop large (fond non supprimé).

SIGMA1_VALUES = [2, 4, 6]
SIGMA2_VALUES = [4, 8, 12]  # sigma2[i] = 2 × sigma1[i] dans cette grille


def create_summary_grid(output_folder, base_name, s1_vals, s2_vals, r, t):
    """Grille récapitulative sigma1 (lignes) × sigma2 (colonnes)."""
    fig, axes = plt.subplots(nrows=len(s1_vals), ncols=len(s2_vals), figsize=(18, 18))
    fig.suptitle(
        f"Grid Search DoG : sigma1 (lignes) × sigma2 (colonnes)  (r={r}, t={t})\nImage : {base_name}.png",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    for i, s1 in enumerate(s1_vals):
        for j, s2 in enumerate(s2_vals):
            filename = f"heatmap_dog_s1={s1}_s2={s2}_r={r}_t={t}_{base_name}.png"
            filepath = os.path.join(output_folder, filename)
            ax = axes[i, j]

            if os.path.exists(filepath):
                img_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(f"σ1={s1} | σ2={s2}", fontsize=13, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"σ1={s1}", fontsize=14, fontweight="bold", labelpad=8)
            if i == 0:
                ax.set_xlabel(f"σ2={s2}", fontsize=14, fontweight="bold", labelpad=8)
                ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    summary_path = os.path.join(output_folder, f"SUMMARY_DOG_{base_name}.png")
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
        for s1 in SIGMA1_VALUES:
            for s2 in SIGMA2_VALUES:
                fname = generate_and_save_heatmap(
                    img, R, T, OUTPUT_DIR, base_name, sigma1=s1, sigma2=s2
                )
                print(f"  Généré : {fname}")

        print("\n--- Étape 2 : Création de la grille récapitulative ---")
        create_summary_grid(OUTPUT_DIR, base_name, SIGMA1_VALUES, SIGMA2_VALUES, R, T)
        print("\nTerminé.")
