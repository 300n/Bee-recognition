import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Dépendance : pip install PyWavelets
import pywt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_wavelet_MC02"

# Paramètres heatmap
R = 40
T = 30

# Paramètres ondelettes (voir commentaires détaillés plus bas)
WAVELET = "db4"  # Famille d'ondelette (db4, db8, sym8, bior2.2...)
LEVEL = 3  # Nombre de niveaux de décomposition
MODE = "soft"  # Mode de seuillage : "soft" (lisse) ou "hard" (agressif)

# ============================================================
#  DÉBRUITAGE PAR SEUILLAGE D'ONDELETTES
# ============================================================
# Ondelettes — Donoho & Johnstone, 1994 (VisuShrink) / Chang et al., 2000
# Méthode multirésolultion : analyse et supprime le bruit séparément
# à chaque échelle spatiale. Très utilisée en traitement du signal et de l'image.
#
# Principe (3 étapes) :
#
# 1. DÉCOMPOSITION
#    La transformée en ondelettes discrète (DWT) décompose l'image en
#    sous-bandes de différentes résolutions, à chaque niveau :
#       - Approximation (LL) : basses fréquences, "l'image lissée"
#       - Détails horizontaux (LH), verticaux (HL), diagonaux (HH)
#    Le bruit, étant non structuré, se retrouve principalement dans
#    les sous-bandes de détail à haute fréquence.
#    Les structures utiles (contours des abeilles) ont des coefficients
#    GRANDS même aux hautes fréquences.
#    Le bruit a des coefficients PETITS et répartis uniformément.
#
# 2. SEUILLAGE
#    Pour chaque sous-bande de détail, on estime le seuil optimal via
#    la méthode VisuShrink :
#       seuil = σ_bruit × √(2 × log(N))
#    où σ_bruit est estimé à partir de la médiane des coefficients
#    (robuste aux valeurs extrêmes) et N est le nombre de pixels.
#    Deux modes de seuillage :
#       - Seuillage doux ("soft")  : coeff. → max(0, |coeff| - seuil) × sign(coeff)
#         Les contours sont légèrement lissés mais sans artéfacts.
#       - Seuillage dur ("hard")   : coeff. → 0 si |coeff| < seuil, sinon conservé
#         Plus agressif, préserve mieux les détails fins mais peut créer des
#         artéfacts en "pointillés" sur les contours.
#    La sous-bande d'approximation (LL) n'est PAS seuillée : elle contient
#    l'essentiel du signal utile (la structure globale de l'image).
#
# 3. RECONSTRUCTION
#    La transformée inverse (IDWT) reconstruit l'image débruitée à partir
#    des coefficients seuillés.
#
# Paramètres clés :
#   wavelet : famille d'ondelette. "db4" (Daubechies 4) est le standard
#             pour les images naturelles. "sym8" préserve mieux les symétries.
#             "bior2.2" est parfois meilleur pour les images texturées.
#   level   : nombre de niveaux de décomposition. 1-2 = débruitage léger,
#             3-4 = débruitage modéré, 5+ = risque de perte de détails fins.
#             Règle : level ≤ log2(min(hauteur, largeur))
#   mode    : "soft" pour un rendu lisse, "hard" pour préserver les contours.
#
# Avantage vs NLM/gaussien :
#   Le seuillage d'ondelettes est MULTIÉCHELLE : il traite différemment
#   le bruit grain (haute fréquence, niveaux hauts) et les artefacts
#   (basse fréquence, niveaux bas). Un gaussien lisse tout pareil.
#   De plus, contrairement à NLM, il ne nécessite pas de chercher des
#   patches similaires → beaucoup plus rapide.
#
# Référence : D. Donoho & I. Johnstone, "Ideal spatial adaptation by
#             wavelet shrinkage", Biometrika, 81(3), 1994.
#             S. Chang et al., "Adaptive wavelet thresholding for image
#             denoising and compression", IEEE TIP, 2000.


def estimate_noise_sigma(detail_coeffs):
    """
    Estime l'écart-type du bruit via la médiane des coefficients de détail
    (méthode de Donoho & Johnstone, robuste aux valeurs aberrantes).
    """
    return np.median(np.abs(detail_coeffs)) / 0.6745


def apply_wavelet_denoising(img_gray, wavelet="db4", level=3, mode="soft"):
    """
    Débruitage par seuillage d'ondelettes (VisuShrink adaptatif).

    Paramètres
    ----------
    img_gray : image uint8 (0-255)
    wavelet  : famille d'ondelette ('db4', 'db8', 'sym8', 'bior2.2')
    level    : niveaux de décomposition (1-5, recommandé : 3)
    mode     : 'soft' (lissé) ou 'hard' (contours préservés)

    Retourne
    --------
    Image débruitée uint8 (0-255)
    """
    img_float = img_gray.astype(np.float64)

    # Décomposition en ondelettes sur 'level' niveaux
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)

    # Seuillage de toutes les sous-bandes de détail (pas de l'approximation)
    coeffs_thresh = [coeffs[0]]  # Approximation : non seuillée
    for detail_level in coeffs[1:]:
        # Chaque niveau contient 3 sous-bandes : (LH, HL, HH)
        # On estime le bruit sur la sous-bande HH du niveau le plus fin
        # (convention de Donoho : meilleur estimateur du bruit capteur)
        sigma = estimate_noise_sigma(detail_level[-1])
        N = img_float.size
        # Seuil VisuShrink universel : σ × √(2 × log(N))
        threshold = sigma * np.sqrt(2 * np.log(N))

        # Seuillage de chaque sous-bande du niveau courant
        detail_thresh = tuple(
            pywt.threshold(sub, value=threshold, mode=mode) for sub in detail_level
        )
        coeffs_thresh.append(detail_thresh)

    # Reconstruction par transformée inverse
    denoised = pywt.waverec2(coeffs_thresh, wavelet=wavelet)

    # Recadrage à la taille originale (waverec2 peut ajouter 1 px aux bords)
    denoised = denoised[: img_gray.shape[0], : img_gray.shape[1]]
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    return denoised


# ============================================================
#  PIPELINE HEATMAP
# ============================================================


def generate_and_save_heatmap(
    img_gray,
    r,
    threshold_val,
    output_folder,
    base_name,
    wavelet="db4",
    level=3,
    mode="soft",
):
    """
    Pipeline heatmap avec prétraitement par ondelettes avant la FFT.
    L'overlay final est rendu sur l'image originale non modifiée.
    """
    # Étape 0 : Débruitage par ondelettes
    img_processed = apply_wavelet_denoising(img_gray, wavelet, level, mode)

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

    filename = f"heatmap_wav_{wavelet}_lvl={level}_{mode}_r={r}_t={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename


# ============================================================
#  GRID SEARCH : famille d'ondelette × niveau × mode
# ============================================================
# On fait varier les paramètres les plus impactants :
# - La famille (influence la forme du filtre)
# - Le niveau (influence la profondeur du débruitage)
# Le mode est fixé à "soft" par défaut (plus adapté aux images naturelles)

WAVELETS = ["db4", "sym8", "bior2.2"]
LEVELS = [2, 3, 4]
MODES = ["soft", "hard"]


def create_summary_grid(output_folder, base_name, wavelets, levels, r, t, mode="soft"):
    """Grille récapitulative wavelet (lignes) × level (colonnes), mode fixé."""
    fig, axes = plt.subplots(nrows=len(wavelets), ncols=len(levels), figsize=(18, 18))
    fig.suptitle(
        f"Grid Search Ondelettes : famille × niveau  (mode={mode}, r={r}, t={t})\nImage : {base_name}.png",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    for i, wav in enumerate(wavelets):
        for j, lvl in enumerate(levels):
            filename = f"heatmap_wav_{wav}_lvl={lvl}_{mode}_r={r}_t={t}_{base_name}.png"
            filepath = os.path.join(output_folder, filename)
            ax = axes[i, j]
            if os.path.exists(filepath):
                ax.imshow(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{wav} | level={lvl}", fontsize=12, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(wav, fontsize=13, fontweight="bold", labelpad=8)
            if i == 0:
                ax.set_xlabel(
                    f"level={lvl}", fontsize=13, fontweight="bold", labelpad=8
                )
                ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    summary_path = os.path.join(
        output_folder, f"SUMMARY_WAVELET_{mode}_{base_name}.png"
    )
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

        # --- Grid search mode "soft" ---
        print(
            f"--- Grid Search (mode=soft) : {len(WAVELETS) * len(LEVELS)} combinaisons ---"
        )
        for wav in WAVELETS:
            for lvl in LEVELS:
                fname = generate_and_save_heatmap(
                    img,
                    R,
                    T,
                    OUTPUT_DIR,
                    base_name,
                    wavelet=wav,
                    level=lvl,
                    mode="soft",
                )
                print(f"  Généré : {fname}")

        print(
            f"\n--- Grid Search (mode=hard) : {len(WAVELETS) * len(LEVELS)} combinaisons ---"
        )
        for wav in WAVELETS:
            for lvl in LEVELS:
                fname = generate_and_save_heatmap(
                    img,
                    R,
                    T,
                    OUTPUT_DIR,
                    base_name,
                    wavelet=wav,
                    level=lvl,
                    mode="hard",
                )
                print(f"  Généré : {fname}")

        print("\n--- Création des grilles récapitulatives ---")
        create_summary_grid(OUTPUT_DIR, base_name, WAVELETS, LEVELS, R, T, mode="soft")
        create_summary_grid(OUTPUT_DIR, base_name, WAVELETS, LEVELS, R, T, mode="hard")
        print("\nTerminé.")
