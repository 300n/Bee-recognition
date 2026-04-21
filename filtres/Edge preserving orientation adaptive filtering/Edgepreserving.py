import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_structure_tensor(img, sigma_d, sigma_i):
    # Dérivées Gaussiennes avec sigma_d (sigma de dérivation)
    grad_x = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), sigma_d), cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(cv2.GaussianBlur(img, (0, 0), sigma_d), cv2.CV_64F, 0, 1)

    Jxx = grad_x**2
    Jxy = grad_x * grad_y
    Jyy = grad_y**2

    # ← ÉTAPE CRITIQUE ABSENTE DANS TON CODE : averaging Gaussien du tenseur
    ksize = int(6 * sigma_i + 1) | 1
    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), sigma_i)
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), sigma_i)
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), sigma_i)

    return Jxx, Jxy, Jyy


def calculate_anisotropy(Jxx, Jxy, Jyy):
    """
    Calcule l'anisotropy A = (lambda1 - lambda2) / (lambda1 + lambda2)
    Formule directe sans décomposition propre explicite pour la vitesse.
    """
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy**2

    # Différence des valeurs propres: sqrt(trace^2 - 4*det)
    # A = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy + epsilon)
    discriminant = np.sqrt((Jxx - Jyy) ** 2 + 4 * (Jxy**2))

    # On ajoute un petit epsilon pour éviter la division par zéro
    anisotropy = discriminant / (trace + 1e-6)
    return anisotropy


def generalized_kuwahara_tensor_filter(Jxx, Jxy, Jyy, kernel_size):
    h = kernel_size // 2
    sub_k = h + 1  # taille de la sous-fenêtre (h+1 x h+1)

    # Les 4 quadrants, chacun ancré sur le pixel courant
    # via un padding asymétrique
    quadrant_tensors = []
    quadrant_anisotropies = []

    # (top, bottom, left, right) padding pour forcer l'ancrage
    anchor_configs = [
        ((h, 0, h, 0), (0, 0)),  # Q1: Nord-Ouest  → anchor en bas-droite
        ((h, 0, 0, h), (0, sub_k - 1)),  # Q2: Nord-Est
        ((0, h, h, 0), (sub_k - 1, 0)),  # Q3: Sud-Ouest
        ((0, h, 0, h), (sub_k - 1, sub_k - 1)),  # Q4: Sud-Est
    ]

    for (pt, pb, pl, pr), anchor in anchor_configs:
        padded_Jxx = cv2.copyMakeBorder(Jxx, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        padded_Jxy = cv2.copyMakeBorder(Jxy, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        padded_Jyy = cv2.copyMakeBorder(Jyy, pt, pb, pl, pr, cv2.BORDER_REFLECT)

        s_Jxx = cv2.boxFilter(
            padded_Jxx, cv2.CV_64F, (sub_k, sub_k), anchor=anchor, normalize=True
        )[pt or None : -pb or None, pl or None : -pr or None]
        s_Jxy = cv2.boxFilter(
            padded_Jxy, cv2.CV_64F, (sub_k, sub_k), anchor=anchor, normalize=True
        )[pt or None : -pb or None, pl or None : -pr or None]
        s_Jyy = cv2.boxFilter(
            padded_Jyy, cv2.CV_64F, (sub_k, sub_k), anchor=anchor, normalize=True
        )[pt or None : -pb or None, pl or None : -pr or None]

        aniso = calculate_anisotropy(s_Jxx, s_Jxy, s_Jyy)
        quadrant_tensors.append((s_Jxx, s_Jxy, s_Jyy))
        quadrant_anisotropies.append(aniso)

    stacked = np.stack(quadrant_anisotropies, axis=0)
    best_idx = np.argmax(stacked, axis=0)

    final_Jxx = np.choose(best_idx, [q[0] for q in quadrant_tensors])
    final_Jxy = np.choose(best_idx, [q[1] for q in quadrant_tensors])
    final_Jyy = np.choose(best_idx, [q[2] for q in quadrant_tensors])

    return final_Jxx, final_Jxy, final_Jyy


def apply_kuwahara_on_image(img, kernel_size):
    h = kernel_size // 2
    sub_k = h + 1
    quadrant_means = []
    quadrant_vars = []

    anchor_configs = [
        ((h, 0, h, 0), (0, 0)),
        ((h, 0, 0, h), (0, sub_k - 1)),
        ((0, h, h, 0), (sub_k - 1, 0)),
        ((0, h, 0, h), (sub_k - 1, sub_k - 1)),
    ]

    for (pt, pb, pl, pr), anchor in anchor_configs:
        padded = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        mean = cv2.boxFilter(
            padded, cv2.CV_64F, (sub_k, sub_k), anchor=anchor, normalize=True
        )
        mean = mean[pt or None : -pb or None, pl or None : -pr or None]

        mean2 = cv2.boxFilter(
            padded**2, cv2.CV_64F, (sub_k, sub_k), anchor=anchor, normalize=True
        )
        mean2 = mean2[pt or None : -pb or None, pl or None : -pr or None]

        var = mean2 - mean**2  # variance = E[X²] - E[X]²
        quadrant_means.append(mean)
        quadrant_vars.append(var)

    # Sélection de la fenêtre la plus homogène (variance minimale)
    stacked_var = np.stack(quadrant_vars, axis=0)
    best_idx = np.argmin(stacked_var, axis=0)  # ← argmin ici, pas argmax
    result = np.choose(best_idx, quadrant_means)
    return result


def get_orientation_from_tensor(Jxx, Jxy, Jyy):
    """
    Extrait l'orientation locale (angle en radians) depuis le tenseur.
    Angle = 0.5 * arctan2(2*Jxy, Jxx - Jyy)
    """
    angle = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    return angle


def adaptive_directional_filtering(img, angles, anisotropy, k_long=9, k_short=1):
    """
    Applique un lissage orienté.
    Pour l'optimisation, on discrétise les angles en N directions.
    """
    height, width = img.shape
    result = np.zeros_like(img, dtype=np.float64)
    total_weight = np.zeros_like(img, dtype=np.float64)

    # Nombre de directions discrètes (ex: 16 directions pour couvrir 180 degrés)
    n_dirs = 16

    for i in range(n_dirs):
        # Angle courant (en degrés pour getRotationMatrix2D)
        theta_deg = (i / n_dirs) * 180.0
        theta_rad = np.deg2rad(theta_deg)

        # Création du noyau gaussien orienté
        # On crée une gaussienne allongée (sigma_x > sigma_y)
        sigma_x = k_long / 2.0
        sigma_y = max(1.0, k_short / 2.0)

        # Taille du kernel suffisante pour contenir la gaussienne tournée
        k_size = int(max(k_long, k_short) * 2) | 1
        center = k_size // 2

        # Création d'un noyau gaussien 2D aligné sur X
        kernel_1d_x = cv2.getGaussianKernel(k_size, sigma_x)
        kernel_1d_y = cv2.getGaussianKernel(k_size, sigma_y)
        kernel = kernel_1d_x * kernel_1d_y.T

        # Rotation du noyau
        rot_mat = cv2.getRotationMatrix2D((center, center), theta_deg, 1.0)
        rotated_kernel = cv2.warpAffine(kernel, rot_mat, (k_size, k_size))

        # Normalisation du noyau
        rotated_kernel /= rotated_kernel.sum()

        # Filtrage de toute l'image avec ce noyau
        filtered_img = cv2.filter2D(img, cv2.CV_64F, rotated_kernel)

        # Calcul du masque de contribution pour cet angle
        # On compare l'angle du noyau avec l'angle local estimé
        # Attention: l'orientation est modulo PI (180°), pas 2PI.
        diff_angle = np.abs(angles - theta_rad)
        diff_angle = np.minimum(diff_angle, np.pi - diff_angle)

        # Poids : élevé si l'angle local correspond à l'angle du filtre
        # On utilise une fonction "porte" ou gaussienne sur la différence angulaire
        # Ici une puissance élevée pour une sélection stricte
        weight = np.exp(-10.0 * (diff_angle**2))

        # Pondération par l'anisotropie (optionnel, mais améliore le résultat dans les zones plates)
        # Si anisotropie faible => pas d'orientation => on lisse moins ou de manière isotrope
        # Le papier suggère d'utiliser l'orientation adaptative surtout là où c'est orienté.

        result += filtered_img * weight
        total_weight += weight

    # Normalisation finale
    output = result / (total_weight + 1e-6)

    # Dans les zones très isotropes (Anisotropie ~ 0), le calcul d'angle est bruité.
    # On peut blender avec un lissage gaussien simple selon l'anisotropie.
    isotropic_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    final_output = output * anisotropy + isotropic_blur * (1 - anisotropy)

    return final_output


def edge_preserving_orientation_filter(image_path):
    # Lecture image
    img = cv2.imread(image_path, 0)
    if img is None:
        print("Erreur de chargement de l'image.")
        return

    img_float = img.astype(np.float64) / 255.0

    # 1. Kuwahara sur l'IMAGE pour lisser les domaines grey-value
    img_kuwahara = apply_kuwahara_on_image(img_float, kernel_size)

    # 2. GST sur l'image originale (ou kuwahara) + averaging
    Jxx, Jxy, Jyy = compute_structure_tensor(img_float, sigma_d, sigma_i)

    # 3. Kuwahara sur le TENSEUR pour avoir des bords d'orientation nets
    Jxx_k, Jxy_k, Jyy_k = generalized_kuwahara_tensor_filter(Jxx, Jxy, Jyy)

    # 4. Filtrage directionnel sur img_kuwahara guidé par le tenseur lissé
    result = adaptive_directional_filtering(
        img_kuwahara, orientation, anisotropy, k_long, k_short
    )

    # Affichage
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Image Originale")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Anisotropie (Kuwahara Tensor)")
    plt.imshow(anisotropy, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Carte d'Orientation")
    plt.imshow(orientation, cmap="hsv")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Résultat Filtré")
    plt.imshow(result, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


sigma_d = 1.0
sigma_i = 2.0
kernel_size = 5
ksize = 3
k_long = 15
k_short = 1

# ================= EXÉCUTION (Paramètres 1920x1200) =================
if __name__ == "__main__":
    import os

    # Votre image en 1920x1200
    my_image_path = "/Volumes/Seagate/Final_videos/Mc01/M01C01_003471.png"

    if os.path.exists(my_image_path):
        print(f"Traitement de l'image : {my_image_path}")

        # Lecture et conversion
        img = cv2.imread(my_image_path, 0)
        img_float = img.astype(np.float64) / 255.0

        # --- ÉTAPE 1 : Tenseur de Structure ---
        # sigma=2.0 est un bon compromis pour du 1080p/1200p
        print("1. Calcul du Tenseur...")
        Jxx, Jxy, Jyy = compute_structure_tensor(img_float, sigma_d, sigma_i)

        # --- ÉTAPE 2 : Kuwahara Généralisé ---
        # kernel_size=13 correspond environ à 13 pixels, suffisant pour couvrir un bord
        print("2. Filtrage Kuwahara...")
        Jxx_k, Jxy_k, Jyy_k = generalized_kuwahara_tensor_filter(
            Jxx, Jxy, Jyy, kernel_size=13
        )

        # --- ÉTAPE 3 : Orientation ---
        orientation = get_orientation_from_tensor(Jxx_k, Jxy_k, Jyy_k)
        anisotropy = calculate_anisotropy(Jxx_k, Jxy_k, Jyy_k)

        # --- ÉTAPE 4 : Filtrage Directionnel ---
        # k_long=21 : Lisse fort le long des lignes (nettoie les alvéoles)
        # k_short=3 : Lisse très peu à travers les lignes (garde la netteté)
        print("3. Lissage Adaptatif...")
        result = adaptive_directional_filtering(
            img_float, orientation, anisotropy, k_long, k_short
        )

        # Affichage
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Originale (1920x1200)")
        plt.subplot(1, 2, 2)
        plt.imshow(result, cmap="gray")
        plt.title("Filtre Kuwahara Adaptatif")
        plt.tight_layout()
        plt.show()

        # Sauvegarde optionnelle
        cv2.imwrite("Resultat_Kuwahara_1920.png", (result * 255).astype(np.uint8))
        print("Terminé.")
    else:
        print("Image non trouvée.")
