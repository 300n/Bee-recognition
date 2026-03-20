import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_filtres_structures_MC02"

R = 40
T = 30

# ============================================================
#  FILTRES ADAPTÉS AU BRUIT STRUCTUREL
# ============================================================
# POURQUOI les filtres classiques (Gaussian, Median, NLM) ne fonctionnaient pas :
# Tes images de ruche contiennent du BRUIT STRUCTUREL, pas aléatoire :
#   - Texture répétitive des alvéoles (hautes fréquences organisées)
#   - Illumination non uniforme (gradient lumineux selon la position de la lampe)
#   - Abeilles qui se chevauchent (contours difficiles à séparer)
# Un filtre gaussien lisse TOUT ensemble → le signal "abeille" est autant
# atténué que le bruit. D'où l'effet quasi nul.
#
# Les 3 filtres ci-dessous s'attaquent spécifiquement à ces problèmes.
# ============================================================


def filter_tophat(img_gray, kernel_size=25):
    """
    FILTRE TOP-HAT (morphologique)
    --------------------------------
    Problème visé : fond non uniforme (cire des alvéoles, gradients d'éclairage)
                    qui "noie" le contraste des corps d'abeilles.

    Principe :
      - Opening morphologique = erosion puis dilatation avec un élément structurant.
        C'est comme "passer un rouleau" sur l'image : les structures plus petites
        que le rouleau (= les abeilles) sont effacées, ce qui donne une estimation
        du fond.
      - Top-Hat = Image originale - Opening(Image)
        On soustrait cette estimation du fond à l'image originale.
        Résultat : seules les structures plus petites que le noyau (les abeilles)
        subsistent, sur un fond rendu uniforme.

    Paramètre :
      kernel_size : taille du noyau (en pixels). Doit être PLUS GRAND que les
                    abeilles et plus petit que les structures de fond (alvéoles).
                    → Essaie 20-40 px selon la taille apparente des abeilles.

    Idéal pour : ruches avec fond texturé en cire, illumination inégale.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)


def filter_clahe(img_gray, clip_limit=2.0, tile_size=8):
    """
    FILTRE CLAHE (Contrast Limited Adaptive Histogram Equalization)
    ---------------------------------------------------------------
    Problème visé : faible contraste local des abeilles par rapport à leur voisinage
                    immédiat (fond de cire de même teinte).

    Principe :
      L'égalisation d'histogramme classique (cv2.equalizeHist) calcule une seule
      courbe de redistribution des niveaux de gris pour TOUTE l'image. Si une zone
      est sombre et une autre claire, le résultat global est médiocre.
      CLAHE divise l'image en petites tuiles (tile_size × tile_size), calcule une
      courbe de redistribution LOCALE pour chacune, puis les recolle avec interpolation
      bilinéaire pour éviter les artéfacts aux bords.
      La limite "clip_limit" empêche de trop amplifier le bruit dans les zones
      homogènes (fond de cire uniforme).

    Paramètres :
      clip_limit : seuil d'écrêtage (2.0-4.0). Plus élevé = plus de contraste,
                   mais aussi plus d'artéfacts dans les zones uniformes.
      tile_size  : taille des tuiles en pixels (8-16). Plus petit = correction
                   plus locale, mais plus sensible au bruit.

    Idéal pour : améliorer la visibilité des abeilles dans des zones d'ombre ou
                 face à des variations d'éclairage.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


def filter_dog(img_gray, sigma1=2, sigma2=4):
    """
    FILTRE DoG (Difference of Gaussians)
    --------------------------------------
    Problème visé : fond texturé et structures parasites de DIFFÉRENTES tailles
                    (alvéoles larges, débris fins) qui polluent la FFT.

    Principe :
      On applique deux flous gaussiens d'écarts-types différents (sigma1 < sigma2),
      puis on les soustrait : DoG = Blur(sigma1) - Blur(sigma2).
      Cela crée un FILTRE PASSE-BANDE spatial :
        - Les très basses fréquences (fond lent, gradient d'éclairage) sont dans les
          DEUX images floues → leur différence est nulle → supprimées.
        - Les très hautes fréquences (bruit grain, poussières) sont dans AUCUNE des
          deux images floues → supprimées aussi.
        - Les fréquences INTERMÉDIAIRES (corps des abeilles, de taille σ1 à σ2)
          sont présentes dans Blur(sigma1) mais pas dans Blur(sigma2) → conservées.

      En choisissant sigma1 et sigma2 autour de la taille caractéristique d'une
      abeille (en pixels), on crée un "détecteur de blob" accordé sur les abeilles.

    Paramètres :
      sigma1 : écart-type du premier flou (plus fin), en pixels.
               Règle : sigma1 ≈ rayon_abeille / 3
      sigma2 : écart-type du second flou (plus large).
               Règle : sigma2 ≈ 2 × sigma1

    Idéal pour : séparer les abeilles des structures d'arrière-plan de taille
                 différente (alvéoles plus grandes, débris plus petits).
    """
    blur1 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(img_gray.astype(np.float32), (0, 0), sigma2)
    dog = blur1 - blur2
    # Remise sur 0-255 (la soustraction peut donner des valeurs négatives)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog.astype(np.uint8)


# Dictionnaire des filtres (nom -> fonction)
FILTER_METHODS = {
    "none": lambda img: img,
    "tophat": filter_tophat,
    "clahe": filter_clahe,
    "dog": filter_dog,
}


# ============================================================
#  PIPELINE HEATMAP
# ============================================================


def generate_heatmap(img_gray, r, threshold_val, preprocess_fn=None):
    """
    Pipeline heatmap avec prétraitement optionnel avant la FFT.

    Le prétraitement s'applique sur img_gray AVANT la FFT.
    L'overlay final est toujours rendu sur l'image originale (non modifiée)
    pour garder une visualisation réaliste.
    """
    # Étape 0 : Prétraitement structurel
    img_processed = preprocess_fn(img_gray) if preprocess_fn else img_gray

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

    return final_overlay


# ============================================================
#  COMPARAISON VISUELLE
# ============================================================


def run_comparison(img_gray, r, t, output_dir, base_name):
    results = {}
    for name, fn in FILTER_METHODS.items():
        overlay = generate_heatmap(
            img_gray, r, t, preprocess_fn=(fn if name != "none" else None)
        )
        results[name] = overlay
        fname = f"heatmap_{name}_r={r}_t={t}_{base_name}.png"
        cv2.imwrite(os.path.join(output_dir, fname), overlay)
        print(f"  Généré : {fname}")

    titles = {
        "none": "Sans filtre\n(référence)",
        "tophat": "Top-Hat\n(supprime fond variable)",
        "clahe": "CLAHE\n(contraste local adaptatif)",
        "dog": "DoG σ=(2,4)\n(détecteur de blobs)",
    }

    fig, axes = plt.subplots(1, len(results), figsize=(22, 5))
    for ax, (name, img_bgr) in zip(axes, results.items()):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[name], fontsize=11, pad=8)
        ax.axis("off")

    fig.suptitle(
        f"Filtres structurels — r={r}, t={t} | {base_name}.png",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"COMPARAISON_STRUCTURES_{base_name}.png")
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
        print(f"Image chargée : {base_name} ({img.shape[1]}×{img.shape[0]}px)\n")
        run_comparison(img, R, T, OUTPUT_DIR, base_name)
        print("\nTerminé.")
