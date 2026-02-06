import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# --- 1. FONCTIONS CŒUR (Multiprocessing & Alignement) ---

def worker_align_ecc(target_path, ref_img_small, scale=0.2):
    """
    Fonction exécutée par les workers CPU pour aligner une image.
    """
    try:
        # Lecture
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None: return None

        # Redimensionnement pour calcul rapide de la matrice
        target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
        
        # Paramètres ECC
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

        try:
            # Calcul de la matrice de transformation (sur petite image)
            (_, warp_matrix) = cv2.findTransformECC(ref_img_small, target_small, warp_matrix, warp_mode, criteria)
            
            # Mise à l'échelle pour l'image HD
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale
            
            # Application de la transformation
            aligned_img = cv2.warpAffine(target_img, warp_matrix, (target_img.shape[1], target_img.shape[0]), 
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned_img
        except:
            # Si l'alignement échoue, on renvoie l'originale
            return target_img

    except Exception as e:
        print(f"Erreur worker : {e}")
        return None

def build_master_pattern_multicore(folder_path, num_samples=100):
    """
    Construit l'image de référence (ruche vide) en utilisant le CPU multicœur.
    """
    files = glob.glob(os.path.join(folder_path, "*.png"))
    # Support pour .jpg si nécessaire
    if not files: files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    if not files: 
        print("Erreur : Aucun fichier image trouvé pour créer le pattern.")
        return None
    
    # Échantillonnage
    sample_files = random.sample(files, min(num_samples, len(files)))
    
    # Initialisation avec la première image
    ref_path = sample_files[0]
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None: return None
    
    # Version réduite pour référence (optimisation vitesse)
    scale = 0.2
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    
    targets_to_process = sample_files[1:]
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"--- Construction du Master Pattern ({len(sample_files)} images sur {num_cores} cœurs) ---")
    
    aligned_stack = [ref_img]
    worker_func = partial(worker_align_ecc, ref_img_small=ref_small, scale=scale)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(worker_func, targets_to_process)
        for res in results:
            if res is not None:
                aligned_stack.append(res)
    
    print("Calcul de la médiane...")
    master_pattern = np.median(np.stack(aligned_stack), axis=0).astype(np.uint8)
    return master_pattern

# --- 2. FONCTION DE TRAITEMENT ET VISUALISATION ---

def process_and_compare(image_path, master_pattern, r, threshold_val, output_folder):
    """
    Traite une image et sauvegarde une comparaison Avant/Après.
    """
    # A. Chargement
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # B. Alignement sur le Master Pattern (Indispensable !)
    # On réutilise la logique d'alignement pour que la soustraction soit parfaite
    scale = 0.2
    ref_small = cv2.resize(master_pattern, None, fx=scale, fy=scale)
    img_aligned = worker_align_ecc(image_path, ref_small, scale) # Appel direct (pas besoin de pool ici pour 1 image)
    if img_aligned is None: img_aligned = img_original

    # C. Détection (Différence + FFT)
    diff = cv2.absdiff(img_aligned, master_pattern)
    
    dft = cv2.dft(np.float32(diff), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = diff.shape
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (cols//2, rows//2), r, (1, 1), -1)
    
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask_fft))
    heatmap = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # D. Création du visuel "Détection"
    result_bgr = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)
    mask_bees = heatmap > threshold_val
    
    # Coloration en Rouge
    result_bgr[mask_bees, 0] = 0   
    result_bgr[mask_bees, 1] = 0   
    result_bgr[mask_bees, 2] = 255 

    # E. Création du panneau Avant/Après (Side-by-Side)
    original_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    
    # Ajout de texte pour identifier
    cv2.putText(original_bgr, "Original (Raw)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    cv2.putText(result_bgr, f"Detection (t={threshold_val})", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

    # Concatenation horizontale
    comparison = cv2.hconcat([original_bgr, result_bgr])
    
    # Sauvegarde
    filename = "COMPARE_" + os.path.basename(image_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, comparison)
    print(f"--> Résultat généré : {filename}")

# ================= 3. EXÉCUTION PRINCIPALE =================

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Dossier contenant tes 8000 images (pour construire le pattern)
    FOLDER_SOURCE = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
    
    # Dossier contenant tes images de test spécifiques
    FOLDER_TEST = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_test'
    
    # Dossier de sortie des résultats
    FOLDER_OUTPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Resultats_Test'
    
    # Paramètres de détection (à ajuster selon tes tests précédents)
    R_PARAM = 60
    THRESH_PARAM = 110

    # Création du dossier de sortie
    if not os.path.exists(FOLDER_OUTPUT):
        os.makedirs(FOLDER_OUTPUT)

    # ÉTAPE 1 : Générer (ou charger) le Pattern Maître
    # On utilise le dossier source (crop) pour avoir une bonne statistique
    master_hive = build_master_pattern_multicore(FOLDER_SOURCE, num_samples=50)
    
    if master_hive is not None:
        # Sauvegarde du pattern pour info
        cv2.imwrite(os.path.join(FOLDER_OUTPUT, "MASTER_PATTERN.png"), master_hive)

        # ÉTAPE 2 : Récupérer les 10 images de test
        # On cherche les .png et .jpg
        test_files = glob.glob(os.path.join(FOLDER_TEST, "*.png")) + glob.glob(os.path.join(FOLDER_TEST, "*.jpg"))
        test_files.sort() # Pour avoir un ordre constant
        
        # On ne garde que les 10 premiers (ou moins si y'en a pas 10)
        subset_test = test_files[:10]
        
        print(f"\n--- Lancement du test sur {len(subset_test)} images ---")
        
        # ÉTAPE 3 : Boucle de traitement
        for img_path in subset_test:
            process_and_compare(img_path, master_hive, R_PARAM, THRESH_PARAM, FOLDER_OUTPUT)
            
        print(f"\nTerminé ! Ouvre le dossier : {FOLDER_OUTPUT}")
    else:
        print("Échec de la création du Master Pattern.")