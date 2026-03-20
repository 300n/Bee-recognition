import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_debruitage_MC02"

# Paramètres heatmap (identiques à heatmap_final.py)
R = 40
T = 30

# ============================================================
#  LES 4 FILTRES DE DÉBRUITAGE
# ============================================================
# Chaque filtre est une fonction (img_gray) -> img_gray_filtrée.
# Ils sont tous appliqués AVANT la FFT pour nettoyer l'image source.
#
# 1. GAUSSIEN  — flou doux, très rapide
#    Principe : chaque pixel est remplacé par une moyenne pondérée de ses
#    voisins, les voisins proches pesant plus lourd (courbe en cloche).
#    Avantage : élimine le bruit "granuleux" uniforme (bruit gaussien).
#    Inconvénient : efface aussi les contours fins (corps des abeilles).
#    Paramètre : ksize (ex. (5,5)) — plus grand = plus de lissage.
#
# 2. MÉDIAN    — robuste aux pixels isolés parasites ("sel et poivre")
#    Principe : chaque pixel est remplacé par la MÉDIANE de son voisinage.
#    La médiane n'est pas influencée par les valeurs extrêmes, donc un
#    pixel blanc isolé sur fond noir disparaît sans affecter ses voisins.
#    Avantage : préserve mieux les bords que le Gaussien.
#    Inconvénient : plus lent ; ksize doit être impair.
#    Paramètre : ksize (ex. 5) — doit être impair.
#
# 3. BILATERAL — lissage qui PRÉSERVE LES CONTOURS (le meilleur compromis)
#    Principe : comme le Gaussien, mais il ajoute un second poids basé sur
#    la différence d'intensité. Deux pixels proches spatialement mais de
#    valeurs très différentes (= un contour) ne sont PAS moyennés ensemble.
#    Avantage : supprime le bruit tout en gardant les bords nets des abeilles.
#    Inconvénient : 5 à 10× plus lent que le Gaussien.
#    Paramètres :
#       d          : diamètre du voisinage (ex. 9)
#       sigmaColor : plage d'intensité fusionnée (ex. 75) — grand = lisse plus
#       sigmaSpace : plage spatiale (ex. 75) — grand = voisins plus lointains
#
# 4. NLM (Non-Local Means) — le plus puissant, mais le plus lent
#    Principe : pour chaque pixel, on cherche des "patches" similaires dans
#    TOUTE l'image (pas seulement le voisinage local). Un pixel bruité est
#    remplacé par la moyenne de tous les pixels qui lui ressemblent, quelle
#    que soit leur position dans l'image.
#    Avantage : débruitage très fin, idéal pour les textures répétitives
#               (les corps d'abeilles sont semblables entre eux → parfait ici).
#    Inconvénient : significativement plus lent.
#    Paramètre :
#       h : force du filtre (ex. 10) — trop haut = image "plastique"
# ============================================================


def denoise_gaussian(img_gray, ksize=5):
    """Filtre Gaussien — lissage rapide, efface le bruit homogène."""
    return cv2.GaussianBlur(img_gray, (ksize, ksize), 0)


def denoise_median(img_gray, ksize=5):
    """Filtre Médian — élimine les pixels isolés (sel & poivre), préserve mieux les bords."""
    return cv2.medianBlur(img_gray, ksize)


def denoise_bilateral(img_gray, d=9, sigma_color=75, sigma_space=75):
    """Filtre Bilatéral — lisse le bruit SANS effacer les contours des abeilles.
    Meilleur compromis qualité/vitesse pour ce type d'images."""
    return cv2.bilateralFilter(img_gray, d, sigma_color, sigma_space)


def denoise_nlm(img_gray, h=10):
    """Non-Local Means — débruitage fin, exploite la répétition des textures.
    Idéal si les abeilles se ressemblent visuellement (cas fréquent en ruche)."""
    return cv2.fastNlMeansDenoising(
        img_gray, None, h=h, templateWindowSize=7, searchWindowSize=21
    )


# Dictionnaire des filtres disponibles (nom -> fonction)
DENOISE_METHODS = {
    "none": lambda img: img,  # Pas de filtre (référence)
    "gaussian": denoise_gaussian,
    "median": denoise_median,
    "bilateral": denoise_bilateral,
    "nlm": denoise_nlm,
}

# ============================================================
#  PIPELINE HEATMAP (identique à heatmap_final.py)
# ============================================================


def generate_heatmap(img_gray, r, threshold_val, denoise_fn=None):
    """
    Génère une heatmap avec un filtre de débruitage optionnel.

    Paramètres
    ----------
    img_gray     : image en niveaux de gris (numpy array)
    r            : rayon du masque FFT (filtre passe-bas)
    threshold_val: seuil THRESH_TOZERO pour isoler les zones chaudes
    denoise_fn   : fonction de débruitage à appliquer avant la FFT,
                   ou None pour aucun filtre.

    Retourne
    --------
    final_overlay : image BGR (overlay heatmap + image originale)
    """
    # --- ÉTAPE 0 : Débruitage (NOUVEAU) ---
    if denoise_fn is not None:
        img_processed = denoise_fn(img_gray)
    else:
        img_processed = img_gray

    rows, cols = img_processed.shape
    crow, ccol = rows // 2, cols // 2

    # --- ÉTAPE 1 : FFT + masque passe-bas ---
    dft = cv2.dft(np.float32(img_processed), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # --- ÉTAPE 2 : Création de la heatmap ---
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_raw = 255 - img_back.astype(np.uint8)  # inversion : abeilles = blanc
    _, heatmap_t = cv2.threshold(heatmap_raw, threshold_val, 255, cv2.THRESH_TOZERO)
    heatmap_color = cv2.applyColorMap(heatmap_t, cv2.COLORMAP_JET)

    # --- ÉTAPE 3 : Overlay sur l'image originale ---
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    return final_overlay


def save_heatmap(overlay, output_folder, filename):
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, overlay)
    return save_path


# ============================================================
#  COMPARAISON VISUELLE : grille des 5 méthodes
# ============================================================


def run_comparison(img_gray, r, t, output_dir, base_name):
    """
    Génère et sauvegarde une image comparative montrant les 5 méthodes
    (référence + 4 filtres) côte à côte.
    """
    results = {}
    for name, fn in DENOISE_METHODS.items():
        overlay = generate_heatmap(
            img_gray, r, t, denoise_fn=(fn if name != "none" else None)
        )
        results[name] = overlay
        fname = f"heatmap_{name}_r={r}_t={t}_{base_name}.png"
        save_heatmap(overlay, output_dir, fname)
        print(f"  Généré : {fname}")

    # Grille matplotlib 1×5
    fig, axes = plt.subplots(1, len(results), figsize=(22, 5))
    titles = {
        "none": "Sans filtre\n(référence)",
        "gaussian": "Gaussien\n(ksize=5)",
        "median": "Médian\n(ksize=5)",
        "bilateral": "Bilatéral\n(d=9, σ=75)",
        "nlm": "NLM\n(h=10)",
    }
    for ax, (name, img_bgr) in zip(axes, results.items()):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titles[name], fontsize=11, pad=8)
        ax.axis("off")

    fig.suptitle(
        f"Comparaison des filtres de débruitage — r={r}, t={t}\n{base_name}.png",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    grid_path = os.path.join(output_dir, f"COMPARAISON_FILTRES_{base_name}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grille comparative : {grid_path}")
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
        print(f"Image chargée : {base_name} ({img.shape[1]}×{img.shape[0]}px)")
        print("\nGénération des 5 variantes ...\n")
        run_comparison(img, R, T, OUTPUT_DIR, base_name)
        print("\nTerminé.")
