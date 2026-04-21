import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# Dépendances supplémentaires :
#   pip install bm3d PyWavelets
import bm3d
import pywt

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_IMAGE_PATH = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000338.png"
OUTPUT_DIR = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/comparaison_filtres"

# Région d'intérêt pour le zoom (optionnel) : (x, y, largeur, hauteur)
# Mettre à None pour désactiver le zoom
ZOOM_ROI = None  # ex: (100, 100, 200, 200)

# ============================================================
#  DÉFINITION DE TOUS LES FILTRES
# ============================================================

# ------ Génération 1 : Filtres classiques ------


def filter_none(img):
    """Référence — aucun filtre appliqué."""
    return img


def filter_gaussian(img, ksize=5):
    """
    Gaussien (k=5)
    Lissage spatial par moyenne pondérée en cloche.
    Élimine le bruit granuleux uniforme.
    Inconvénient : efface aussi les contours fins.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def filter_median(img, ksize=5):
    """
    Médian (k=5)
    Remplace chaque pixel par la médiane de son voisinage.
    Robuste aux pixels isolés (bruit sel & poivre).
    Préserve mieux les bords que le Gaussien.
    """
    return cv2.medianBlur(img, ksize)


def filter_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """
    Bilatéral (d=9, σ=75)
    Lissage gaussien pondéré par la différence d'intensité.
    Supprime le bruit sans effacer les contours.
    5-10× plus lent que le Gaussien.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def filter_nlm(img, h=10):
    """
    NLM — Non-Local Means (h=10)
    Chaque pixel est débruité par tous les patches similaires
    dans toute l'image (non local).
    Exploite la répétition des textures (corps d'abeilles).
    Plus lent que les filtres précédents.
    """
    return cv2.fastNlMeansDenoising(
        img, None, h=h, templateWindowSize=7, searchWindowSize=21
    )


# ------ Génération 2 : État de l'art classique ------


def filter_bm3d(img, sigma_psd=15):
    """
    BM3D — Block Matching & 3D filtering (σ=15)
    Gold standard du débruitage classique (Dabov et al., 2007).
    Block matching → empilement 3D → seuillage + filtre de Wiener.
    Exploite la redondance globale des blocs similaires.
    Référence : IEEE TIP, vol. 16, no. 8, 2007.
    """
    img_float = img.astype(np.float64) / 255.0
    sigma_norm = sigma_psd / 255.0
    denoised = bm3d.bm3d(img_float, sigma_psd=sigma_norm)
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


def filter_wavelet(img, wavelet="db4", level=3, mode="soft"):
    """
    Ondelettes — VisuShrink (db4, level=3, soft)
    Décomposition multirésolution + seuillage par sous-bande.
    Supprime le bruit haute fréquence sans toucher les basses fréquences.
    Seuil adaptatif : σ × √(2 × log(N)), estimé par médiane des coefficients.
    Référence : Donoho & Johnstone, Biometrika, 1994.
    """
    img_float = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)
    coeffs_t = [coeffs[0]]
    for detail in coeffs[1:]:
        sigma = np.median(np.abs(detail[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(img_float.size))
        coeffs_t.append(
            tuple(pywt.threshold(sub, value=threshold, mode=mode) for sub in detail)
        )
    denoised = pywt.waverec2(coeffs_t, wavelet=wavelet)
    denoised = denoised[: img.shape[0], : img.shape[1]]
    return np.clip(denoised, 0, 255).astype(np.uint8)


# ============================================================
#  REGISTRE DES FILTRES
#  Chaque entrée : (label_court, label_long, fonction)
# ============================================================
FILTERS = [
    (
        "Sans filtre\n(référence)",
        "Aucun prétraitement",
        filter_none,
    ),
    (
        "Gaussien\n(k=5)",
        "GaussianBlur — lissage local",
        filter_gaussian,
    ),
    (
        "Médian\n(k=5)",
        "medianBlur — robuste sel & poivre",
        filter_median,
    ),
    (
        "Bilatéral\n(d=9, σ=75)",
        "bilateralFilter — préserve les contours",
        filter_bilateral,
    ),
    (
        "NLM\n(h=10)",
        "Non-Local Means — patches globaux",
        filter_nlm,
    ),
    (
        "BM3D\n(σ=15)",
        "Block Matching 3D — gold standard",
        filter_bm3d,
    ),
    (
        "Ondelettes\n(db4, lvl=3)",
        "Wavelet VisuShrink — multiéchelle",
        filter_wavelet,
    ),
]

N_FILTERS = len(FILTERS)


# ============================================================
#  CALCUL DE MÉTRIQUES DE QUALITÉ
# ============================================================


def compute_metrics(img_orig, img_filtered):
    """
    Calcule trois métriques pour évaluer l'effet du filtre
    par rapport à l'image originale.

    PSNR (Peak Signal-to-Noise Ratio) :
      Mesure la fidélité à l'original en dB.
      PSNR élevé = peu de modification de l'image.
      Note : ici l'image originale n'est PAS la "vraie image sans bruit",
      donc un PSNR élevé signifie que le filtre a peu modifié l'image,
      pas nécessairement qu'il est meilleur.

    Gradient moyen (netteté) :
      Moyenne de la magnitude du gradient de Sobel.
      Valeur élevée = image plus nette / contours bien préservés.
      Un filtre qui floute trop réduira fortement cette valeur.

    Écart-type (contraste global) :
      Mesure la dispersion des niveaux de gris.
      Chute importante → le filtre aplatit les contrastes.
    """
    orig_f = img_orig.astype(np.float64)
    filt_f = img_filtered.astype(np.float64)
    mse = np.mean((orig_f - filt_f) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    grad_x = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    std = float(np.std(img_filtered))
    return psnr, sharpness, std


# ============================================================
#  GÉNÉRATION DE LA GRILLE COMPARATIVE
# ============================================================


def create_comparison_grid(img_gray, output_dir, base_name, zoom_roi=None):
    """
    Génère une grille comparative :
      - Ligne 1 : images débruitées (plein format ou zoom selon zoom_roi)
      - Ligne 2 : différences amplifiées × 5 vs image originale
      - Ligne 3 : tableau de métriques (PSNR, netteté, std)
    """
    print(f"\n{'='*60}")
    print(f"  Application des {N_FILTERS} filtres...")
    print(f"{'='*60}")

    results = []
    durations = []

    for label, desc, fn in FILTERS:
        t0 = time.time()
        out = fn(img_gray.copy())
        dt = time.time() - t0
        durations.append(dt)

        psnr, sharp, std = compute_metrics(img_gray, out)
        results.append(
            {
                "label": label,
                "desc": desc,
                "image": out,
                "psnr": psnr,
                "sharpness": sharp,
                "std": std,
                "time_ms": dt * 1000,
            }
        )

        marker = "∞" if psnr == float("inf") else f"{psnr:.1f}"
        print(
            f"  {label.split(chr(10))[0]:20s}  PSNR={marker:>7} dB  "
            f"net={sharp:6.1f}  std={std:5.1f}  ({dt*1000:.0f} ms)"
        )

    # ---- Mise en page ----
    # 3 lignes : images / différences / métriques
    fig = plt.figure(figsize=(4 * N_FILTERS, 14))
    gs = gridspec.GridSpec(
        3,
        N_FILTERS,
        figure=fig,
        height_ratios=[3, 3, 1.8],
        hspace=0.35,
        wspace=0.05,
    )

    fig.suptitle(
        f"Comparaison des filtres de débruitage — image brute\n{base_name}.png",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # --- Ligne 1 : images ---
    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[0, j])

        if zoom_roi is not None:
            x, y, w, h = zoom_roi
            disp = r["image"][y : y + h, x : x + w]
        else:
            disp = r["image"]

        ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
        ax.set_title(r["label"], fontsize=10, pad=6, linespacing=1.4)
        ax.axis("off")

        # Bordure rouge sur la référence
        if j == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cc3333")
                spine.set_linewidth(2)
                spine.set_visible(True)

    # --- Ligne 2 : différences amplifiées ---
    orig_ref = results[0]["image"].astype(np.float64)
    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[1, j])
        diff = np.abs(r["image"].astype(np.float64) - orig_ref)
        diff_amp = np.clip(diff * 5, 0, 255).astype(np.uint8)

        if zoom_roi is not None:
            x, y, w, h = zoom_roi
            diff_amp = diff_amp[y : y + h, x : x + w]

        ax.imshow(diff_amp, cmap="hot", vmin=0, vmax=255)
        if j == 0:
            ax.set_title("Référence\n(diff = 0)", fontsize=9, pad=4)
        else:
            ax.set_title("Différence × 5\nvs référence", fontsize=9, pad=4)
        ax.axis("off")

    # --- Ligne 3 : métriques sous forme de barres textuelles ---
    metrics_labels = [
        "PSNR (dB)",
        "Netteté\n(gradient)",
        "Écart-type\n(contraste)",
        "Temps (ms)",
    ]
    metrics_keys = ["psnr", "sharpness", "std", "time_ms"]
    colors_bar = ["#3a7dc9", "#2aa876", "#c07430", "#9060c0"]

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[2, j])
        ax.axis("off")

        vals = [
            r["psnr"] if r["psnr"] != float("inf") else 999,
            r["sharpness"],
            r["std"],
            r["time_ms"],
        ]
        fmt = ["{:.1f}", "{:.1f}", "{:.1f}", "{:.0f} ms"]
        y_pos = [0.88, 0.66, 0.44, 0.22]

        for k, (lbl, val, f, yp, col) in enumerate(
            zip(metrics_labels, vals, fmt, y_pos, colors_bar)
        ):
            display = "∞" if (k == 0 and r["psnr"] == float("inf")) else f.format(val)
            ax.text(
                0.5,
                yp + 0.1,
                lbl,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#888888",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                yp - 0.02,
                display,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=col,
                transform=ax.transAxes,
            )

    # ---- Sauvegarde ----
    zoom_tag = "_zoom" if zoom_roi is not None else ""
    out_path = os.path.join(
        output_dir, f"COMPARAISON_TOUS_FILTRES{zoom_tag}_{base_name}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Grille sauvegardée : {out_path}")
    plt.close()
    return out_path


# ============================================================
#  SAUVEGARDE DES IMAGES INDIVIDUELLES
# ============================================================


def save_individual_images(img_gray, output_dir, base_name):
    """Sauvegarde chaque image débruitée individuellement."""
    print(f"\n--- Sauvegarde des images individuelles ---")
    names = {
        "Sans filtre\n(référence)": "none",
        "Gaussien\n(k=5)": "gaussian",
        "Médian\n(k=5)": "median",
        "Bilatéral\n(d=9, σ=75)": "bilateral",
        "NLM\n(h=10)": "nlm",
        "BM3D\n(σ=15)": "bm3d",
        "Ondelettes\n(db4, lvl=3)": "wavelet",
    }
    for label, desc, fn in FILTERS:
        short = names.get(label, label.split("\n")[0].lower().replace(" ", "_"))
        out = fn(img_gray.copy())
        path = os.path.join(output_dir, f"denoised_{short}_{base_name}.png")
        cv2.imwrite(path, out)
        print(f"  {short:20s} → {os.path.basename(path)}")


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
        print(f"Image chargée : {base_name}  ({img.shape[1]}×{img.shape[0]} px)")

        # Grille comparative plein format
        create_comparison_grid(img, OUTPUT_DIR, base_name, zoom_roi=ZOOM_ROI)

        # Sauvegarde des images individuelles
        save_individual_images(img, OUTPUT_DIR, base_name)

        print("\nTerminé.")
