import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# --- GARDER LES FONCTIONS D'ALIGNEMENT TELLES QUELLES (worker_align_ecc, build_master_pattern_multicore) ---
# Copiez-collez ici les fonctions worker_align_ecc et build_master_pattern_multicore du code précédent
# Pour éviter de tout remettre et saturer la réponse, je mets juste la partie modifiée ci-dessous.

def worker_align_ecc(target_path, ref_img_small, scale=0.2):
    try:
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None: return None
        target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        try:
            (_, warp_matrix) = cv2.findTransformECC(ref_img_small, target_small, warp_matrix, warp_mode, criteria)
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale
            aligned_img = cv2.warpAffine(target_img, warp_matrix, (target_img.shape[1], target_img.shape[0]), 
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned_img
        except:
            return target_img
    except:
        return None

def build_master_pattern_multicore(folder_path, num_samples):
    files = glob.glob(os.path.join(folder_path, "*.png"))
    if not files: files = glob.glob(os.path.join(folder_path, "*.jpg"))
    if not files: return None
    sample_files = random.sample(files, min(num_samples, len(files)))
    ref_img = cv2.imread(sample_files[0], cv2.IMREAD_GRAYSCALE)
    scale = 0.2
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    targets_to_process = sample_files[1:]
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"--- Construction Master Pattern ({len(sample_files)} images) ---")
    aligned_stack = [ref_img]
    worker_func = partial(worker_align_ecc, ref_img_small=ref_small, scale=scale)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(worker_func, targets_to_process)
        for res in results:
            if res is not None: aligned_stack.append(res)
    return np.median(np.stack(aligned_stack), axis=0).astype(np.uint8)

# --- LA FONCTION MODIFIÉE POUR LE DEBUG ---

def process_and_compare_debug(image_path, master_pattern, r, threshold_val, output_folder):
    # A. Chargement
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # B. Alignement
    scale = 0.2
    ref_small = cv2.resize(master_pattern, None, fx=scale, fy=scale)
    img_aligned = worker_align_ecc(image_path, ref_small, scale)
    if img_aligned is None: img_aligned = img_original

    # C. Détection (Heatmap brute)
    diff = cv2.absdiff(img_aligned, master_pattern)
    
    # FFT pour nettoyer le bruit de fond
    dft = cv2.dft(np.float32(diff), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = diff.shape
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (cols//2, rows//2), r, (1, 1), -1)
    
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask_fft))
    heatmap = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalisation STRICTE (0-255)
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # D. Masque et Colorisation
    result_bgr = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)
    mask_bees = heatmap > threshold_val
    
    # On peint en ROUGE VIF
    result_bgr[mask_bees, 0] = 0   
    result_bgr[mask_bees, 1] = 0   
    result_bgr[mask_bees, 2] = 255 

    # E. VISUALISATION 3 VOLETS (DEBUG)
    # 1. Image Originale
    original_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    cv2.putText(original_bgr, "Original", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    
    # 2. Heatmap (Ce que voit l'ordinateur) - Converti en "Fausse couleur" pour mieux voir les intensités
    heatmap_visu = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    cv2.putText(heatmap_visu, "Raw Heatmap", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    # 3. Résultat Final
    cv2.putText(result_bgr, f"Seuil > {threshold_val}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

    # Concaténation des 3
    comparison = cv2.hconcat([original_bgr, heatmap_visu, result_bgr])
    
    filename = "DEBUG_" + os.path.basename(image_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, comparison)
    print(f"--> Debug généré : {filename}")

# ================= MAIN =================

if __name__ == '__main__':
    FOLDER_SOURCE = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
    FOLDER_TEST = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_test'
    FOLDER_OUTPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Resultats_Test4_500img_v1'
    
    # --- PARAMÈTRES MODIFIÉS ---
    # On baisse le seuil drastiquement pour voir quelque chose
    R_PARAM = 60
    THRESH_PARAM = 35  # <-- ESSAYEZ 35, puis 50, puis 20 si besoin. 110 était trop haut.

    if not os.path.exists(FOLDER_OUTPUT):
        os.makedirs(FOLDER_OUTPUT)

    master_hive = build_master_pattern_multicore(FOLDER_SOURCE, 500)
    
    if master_hive is not None:
        cv2.imwrite(os.path.join(FOLDER_OUTPUT, "MASTER_PATTERN.png"), master_hive)

        test_files = glob.glob(os.path.join(FOLDER_TEST, "*.png")) + glob.glob(os.path.join(FOLDER_TEST, "*.jpg"))
        subset_test = test_files[:10]
        
        print(f"\n--- Test Debug avec Seuil={THRESH_PARAM} ---")
        for img_path in subset_test:
            process_and_compare_debug(img_path, master_hive, R_PARAM, THRESH_PARAM, FOLDER_OUTPUT)
            
        print(f"\nTerminé ! Regardez l'image du MILIEU (Raw Heatmap) dans : {FOLDER_OUTPUT}")