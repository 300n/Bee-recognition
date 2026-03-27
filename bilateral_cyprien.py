import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

# Chemin de votre image source
img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"

# Dossier de sauvegarde pour les résultats
save_dir = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/resultats_filtre_math"
)

# --- Paramètres du Filtre Bilatéral ---
# f: Rayon du patch (Taille du patch = 2*f + 1)
# sigma_r: Filtre d'intensité (Tolérance de contraste)
# sigma_s: Filtre spatial (Décroissance géométrique)

# Paramètres idéaux pour lisser le nid d'abeille et faire ressortir les abeilles
# f=3 -> Fenêtre de 7x7
# sigma_r=100.0 -> Très tolérant, lissera la texture du nid d'abeille
# sigma_s=1.0 -> Lissage spatial très concentré pour préserver les abeilles
params = {"f": 3, "sigma_r": 100.0, "sigma_s": 1.0}

# Paramètre du bruit gaussien ajouté
noise_sigma = 10.0

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================


def im_extract_rgb(U, i, j, f):
    """Extrait un patch couleur centré en (i,j). Gère les bords."""
    H, W, C = U.shape
    i0 = max(0, i - f)
    i1 = min(H, i + f + 1)
    j0 = max(0, j - f)
    j1 = min(W, j + f + 1)
    patch = U[i0:i1, j0:j1, :]
    return patch, (i0, i1, j0, j1)


def psnr_rgb(U, V, data_range=255.0, eps=1e-12):
    """PSNR entre images float RGB (même taille)."""
    mse = np.mean((U - V) ** 2)
    return 10.0 * np.log10((data_range**2) / (mse + eps))


def show_color(img, title=""):
    """Affiche une image couleur float [0, 255]."""
    plt.imshow(img.astype(np.uint8))
    plt.title(title)
    plt.axis("off")


def add_gaussian_noise_rgb(U, sigma=10.0, seed=None):
    """Ajout de bruit Gaussien additif RGB N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    Ub = U + rng.normal(0.0, sigma, size=U.shape)
    return np.clip(Ub, 0.0, 255.0)


def gaussian_spatial_kernel(f, sigma_s):
    """Pré-calcule g_s sur la grille [-f..f]x[-f..f]."""
    ax = np.arange(-f, f + 1)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    dist2 = xx**2 + yy**2
    return np.exp(-dist2 / (2 * sigma_s**2))


def im_bilateral_rgb(U_input, f, sigma_r, sigma_s):
    """Applique un filtre bilatéral sur une image couleur float64."""
    U = U_input.astype(np.float64)  # S'assurer d'être en float pour les calculs
    H, W, C = U.shape
    V = np.zeros_like(U)
    two_sr2 = 2 * (sigma_r**2)

    gs_full = gaussian_spatial_kernel(f, sigma_s)  # taille (2f+1,2f+1)

    # Barre de progression pour suivre l'avancement (c'est lent !)
    for i in tqdm(range(H), desc="Filtrage bilatéral"):
        for j in range(W):
            patch, (i0, i1, j0, j1) = im_extract_rgb(U, i, j, f)
            center = U[i, j, :]  # Pixel central RGB

            # Ajuster le kernel spatial si on est au bord
            di0 = i0 - (i - f)
            dj0 = j0 - (j - f)
            di1 = di0 + (i1 - i0)
            dj1 = dj0 + (j1 - j0)
            gs = gs_full[di0:di1, dj0:dj1]

            # Calcul du poids d'intensité gr (vectorisé sur les canaux)
            # Distance euclidienne de couleur entre chaque pixel du patch et le center
            color_dist = np.sqrt(np.sum((patch - center) ** 2, axis=2))
            gr = np.exp(-(color_dist**2) / two_sr2)

            # Combinaison des poids gs (spatial) et gr (intensité)
            w = gs * gr

            # Normalisation et calcul du pixel de sortie
            C_sum = np.sum(w)
            if C_sum < 1e-12:
                V[i, j] = center
            else:
                # Moyenne pondérée sur tous les canaux
                w_expanded = w[:, :, np.newaxis]  # Pour l'appliquer aux canaux RGB
                V[i, j] = np.sum(w_expanded * patch, axis=(0, 1)) / C_sum
    return np.clip(V, 0.0, 255.0)


# ==========================================
# EXECUTION DU TRAITEMENT
# ==========================================

# 1. Chargement de l'image
print(f"Chargement de l'image : {img_path}")
U_pil = Image.open(img_path)
U = np.array(U_pil, dtype=np.float64)
H, W, C = U.shape
print(f"Image chargée : {H}x{W} pixels, {C} canaux (RGB).")

# 2. Ajout de bruit (pour simuler et tester le filtre)
print(f"Ajout de bruit gaussien (σ={noise_sigma})...")
Ub = add_gaussian_noise_rgb(U, sigma=noise_sigma, seed=4)

# 3. Application du filtre bilatéral (cette étape est très longue !)
print(
    f"Application du filtre bilatéral : f={params['f']}, σr={params['sigma_r']}, σs={params['sigma_s']}..."
)
V_bi = im_bilateral_rgb(Ub, params["f"], params["sigma_r"], params["sigma_s"])

# 4. Calcul et affichage des PSNR
print(f"PSNR noisy : {psnr_rgb(U, Ub):.2f} dB")
print(f"PSNR bilateral : {psnr_rgb(U, V_bi):.2f} dB")

# 5. Affichage des résultats
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
show_color(U, "1_Originale")
plt.subplot(1, 3, 2)
show_color(Ub, f"2_Bruitee (σ={noise_sigma})")
plt.subplot(1, 3, 3)
show_color(
    V_bi, f"3_Filtree (f={params['f']}, σr={params['sigma_r']}, σs={params['sigma_s']})"
)
plt.tight_layout()
plt.show()

# ==========================================
# SAUVEGARDE DES IMAGES
# ==========================================

print("\nSauvegarde des images...")

# Créer le dossier de sauvegarde s'il n'existe pas
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Convertir les matrices float64 en images PIL 8-bit (uint8)
img_originale = Image.fromarray(np.clip(U, 0, 255).astype(np.uint8))
img_bruitee = Image.fromarray(np.clip(Ub, 0, 255).astype(np.uint8))
img_filtree = Image.fromarray(np.clip(V_bi, 0, 255).astype(np.uint8))

# Sauvegarder
img_originale.save(os.path.join(save_dir, "1_originale.png"))
img_bruitee.save(os.path.join(save_dir, "2_bruitee.png"))
img_filtree.save(os.path.join(save_dir, "3_filtree.png"))

print(f"Les 3 images ont été sauvegardées avec succès dans :\n{save_dir}")
