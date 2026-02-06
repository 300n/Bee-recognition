import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing


# --- 1. ALIGNEMENT AFFINE (Plus précis que Translation) ---
def worker_align_ecc(target_path, ref_img_small, scale):
    try:
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None: return None

        # Redimensionnement
        target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
        
        # CHANGEMENT ICI : MOTION_AFFINE gère rotation + scale + translation
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

        try:
            (_, warp_matrix) = cv2.findTransformECC(ref_img_small, target_small, warp_matrix, warp_mode, criteria)
            
            # Mise à l'échelle de la matrice
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale
            
            # Application de la transformation
            aligned_img = cv2.warpAffine(target_img, warp_matrix, (target_img.shape[1], target_img.shape[0]), 
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned_img
        except:
            return target_img # Fallback si échec
    except:
        return None

# --- 2. MASTER PATTERN VIA PERCENTILE (Le secret pour effacer les abeilles) ---
def build_master_pattern_percentile(folder_path, num_samples, percentile):
    files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg"))
    if not files: return None
    
    # On augmente l'échantillon pour être sûr de voir la cire au moins une fois
    sample_files = random.sample(files, min(num_samples, len(files)))
    
    ref_img = cv2.imread(sample_files[0], cv2.IMREAD_GRAYSCALE)
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    
    targets_to_process = sample_files[1:]
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"--- Construction Pattern (Percentile {percentile}) sur {len(sample_files)} images ---")
    
    aligned_stack = [ref_img]
    worker_func = partial(worker_align_ecc, ref_img_small=ref_small, scale=scale)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(worker_func, targets_to_process))
        for res in results:
            if res is not None: aligned_stack.append(res)
    
    # CHANGEMENT MAJEUR : np.percentile au lieu de np.median
    # On prend le 85ème centile (pixels clairs = cire) pour éliminer les abeilles (sombres)
    print("Calcul statistique du fond...")
    stack_array = np.stack(aligned_stack, axis=0)
    master_pattern = np.percentile(stack_array, percentile, axis=0).astype(np.uint8)
    
    return master_pattern

# --- 3. DÉTECTION AMÉLIORÉE (CLAHE + Morphologie) ---
def process_advanced_debug(image_path, master_pattern, r, threshold_val, output_folder):
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # A. Alignement
    ref_small = cv2.resize(master_pattern, None, fx=scale, fy=scale)
    img_aligned = worker_align_ecc(image_path, ref_small, scale)
    if img_aligned is None: img_aligned = img_original

    # B. Pré-traitement CLAHE (Égalisation locale de l'histogramme)
    # Cela permet de booster le contraste dans les zones sombres (couvain)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_aligned)
    master_clahe = clahe.apply(master_pattern)

    # C. Calcul de la différence
    # On utilise absdiff sur les versions CLAHE
    diff = cv2.absdiff(img_clahe, master_clahe)
    
    # D. Filtrage fréquentiel (FFT) pour enlever le bruit de grille résiduel
    dft = cv2.dft(np.float32(diff), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = diff.shape
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (cols//2, rows//2), r, (1, 1), -1) # r=60 est bon
    
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask_fft))
    heatmap = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(heatmap, heatmap, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # E. Post-traitement Morphologique (Nouveau !)
    # On dilate les points pour connecter les morceaux d'abeilles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    heatmap_smooth = cv2.morphologyEx(heatmap, cv2.MORPH_CLOSE, kernel)

    # F. Seuillage
    mask_bees = heatmap_smooth > threshold_val

    # G. Visualisation
    result_bgr = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)
    result_bgr[mask_bees] = [0, 0, 255] # Rouge

    # Panel de comparaison
    original_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    heatmap_visu = cv2.applyColorMap(heatmap_smooth, cv2.COLORMAP_INFERNO)
    
    # Texte
    cv2.putText(original_bgr, "Original", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.putText(heatmap_visu, "CLAHE Heatmap", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(result_bgr, f"Seuil {threshold_val}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    comparison = cv2.hconcat([original_bgr, heatmap_visu, result_bgr])
    
    filename = "ADVANCED_" + os.path.basename(image_path)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, comparison)
    print(f"--> Généré : {filename}")

scale=0.2
num_samples=100
percentile=90
r=80
threshold_val=50


# ================= EXÉCUTION =================
if __name__ == '__main__':
    FOLDER_SOURCE = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
    FOLDER_TEST = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_test'
    FOLDER_OUTPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Resultats_Test8_img_v2'
    
    if not os.path.exists(FOLDER_OUTPUT): os.makedirs(FOLDER_OUTPUT)

    # 1. On construit un MEILLEUR pattern (85ème centile pour virer les abeilles noires)
    # Augmente num_samples si ton ordi le permet (ex: 100)
    master_hive = build_master_pattern_percentile(FOLDER_SOURCE, num_samples, percentile)
    
    if master_hive is not None:
        cv2.imwrite(os.path.join(FOLDER_OUTPUT, "MASTER_PERCENTILE_85.png"), master_hive)
        
        # 2. Test sur les images
        test_files = glob.glob(os.path.join(FOLDER_TEST, "*.png"))[:10]
        
        # Le seuil devra peut-être être ajusté car CLAHE change la dynamique des gris
        # Garder r=60 ou r=70 ou r=80
        for img in test_files:
            process_advanced_debug(img, master_hive, r, threshold_val, output_folder=FOLDER_OUTPUT)