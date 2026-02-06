import cv2
import numpy as np
import os
import glob
import random

def align_image_ecc(ref_img, target_img):
    """
    Aligne l'image cible sur l'image de référence en utilisant l'algorithme ECC.
    Pour la rapidité sur 3536px, le calcul se fait sur une image réduite.
    """
    # 1. Redimensionnement pour accélérer le calcul de la matrice de transformation
    scale = 0.2
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    target_small = cv2.resize(target_img, None, fx=scale, fy=scale)

    # 2. Définir le modèle de mouvement (Translation est souvent suffisant pour des vibrations)
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # 3. Paramètres de l'algorithme
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

    try:
        # Trouver la transformation sur la petite image
        (_, warp_matrix) = cv2.findTransformECC(ref_small, target_small, warp_matrix, warp_mode, criteria)
        
        # Ajuster la matrice pour la taille réelle
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

        # Appliquer la transformation à l'image haute résolution
        aligned_img = cv2.warpAffine(target_img, warp_matrix, (target_img.shape[1], target_img.shape[0]), 
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img
    except:
        # Si l'alignement échoue (mouvement trop brusque), on renvoie l'image originale
        return target_img

def build_master_pattern(folder_path, num_samples=50):
    """
    Crée le pattern statique des alvéoles en alignant un échantillon d'images.
    """
    files = glob.glob(os.path.join(folder_path, "*.png"))
    if not files: return None
    
    sample_files = random.sample(files, min(num_samples, len(files)))
    
    # On prend la première image comme référence pour l'alignement
    ref_img = cv2.imread(sample_files[0], cv2.IMREAD_GRAYSCALE)
    aligned_stack = [ref_img]

    print(f"Construction du Pattern Maître avec {len(sample_files)} images...")
    for i in range(1, len(sample_files)):
        img = cv2.imread(sample_files[i], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            aligned = align_image_ecc(ref_img, img)
            aligned_stack.append(aligned)
        if i % 10 == 0: print(f" Alignement : {i}/{len(sample_files)}")

    # Calcul de la médiane pour effacer les abeilles et garder les alvéoles
    master_pattern = np.median(np.stack(aligned_stack), axis=0).astype(np.uint8)
    return master_pattern

def show_final_result(image_path, master_pattern, r=55, threshold_val=100):
    """
    Affiche l'image originale avec les abeilles en rouge basées sur la corrélation.
    """
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # 1. Alignement de l'image actuelle sur le master pattern
    img_aligned = align_image_ecc(master_pattern, img_original)

    # 2. Détection de la "Chaleur" (Différence entre image et ruche vide)
    # On utilise la FFT sur la différence pour lisser les micro-résidus
    diff = cv2.absdiff(img_aligned, master_pattern)
    
    # Filtrage FFT pour isoler les formes organiques (abeilles)
    dft = cv2.dft(np.float32(diff), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = diff.shape
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (cols//2, rows//2), r, (1, 1), -1)
    
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask_fft))
    heatmap_values = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(heatmap_values, heatmap_values, 0, 255, cv2.NORM_MINMAX)
    heatmap_values = heatmap_values.astype(np.uint8)

    # 3. Logique Conditionnelle
    result = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    mask_bees = heatmap_values > threshold_val

    # Coloration Rouge Dynamique
    result[mask_bees, 0] = 0   # Bleu
    result[mask_bees, 1] = 0   # Vert
    result[mask_bees, 2] = 255 # Rouge

    # 4. Affichage
    display = cv2.resize(result, (1200, 800))
    cv2.imshow(f"Analyse Finale - r={r} t={threshold_val}", display)
    print("Fenêtre ouverte. Appuyez sur une touche pour quitter.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================= EXÉCUTION =================
# Chemins
FOLDER_CROP = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
TEST_IMAGE = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'

# 1. Générer le pattern universel des alvéoles (à faire une seule fois)
master_hive = build_master_pattern(FOLDER_CROP, num_samples=40)

# 2. Lancer la visualisation
if master_hive is not None:
    show_final_result(TEST_IMAGE, master_hive, r=60, threshold_val=110)