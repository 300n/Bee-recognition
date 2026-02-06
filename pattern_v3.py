import cv2
import numpy as np
import os
import glob
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

# --- 1. ALIGNEMENT (Inchangé) ---
def worker_align_ecc(target_path, ref_img_small, scale):
    try:
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target_img is None: return None
        target_small = cv2.resize(target_img, None, fx=scale, fy=scale)
        warp_mode = cv2.MOTION_AFFINE
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

# --- 2. MASTER PATTERN + VARIANCE MAP (Z-SCORE) ---
def build_statistical_model(folder_path, num_samples, percentile):
    files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg"))
    if not files: return None, None
    
    sample_files = random.sample(files, min(num_samples, len(files)))
    ref_img = cv2.imread(sample_files[0], cv2.IMREAD_GRAYSCALE)
    ref_small = cv2.resize(ref_img, None, fx=scale, fy=scale)
    
    print(f"--- Construction Modèle Statistique ({len(sample_files)} images) ---")
    
    aligned_stack = [ref_img]
    worker_func = partial(worker_align_ecc, ref_img_small=ref_small, scale=scale)
    
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(worker_func, sample_files[1:])) # Exclut la ref pour map
        for res in results:
            if res is not None: aligned_stack.append(res)
    
    print("Calcul des statistiques pixel par pixel...")
    stack_array = np.stack(aligned_stack, axis=0).astype(np.float32)
    
    # 1. Le Fond (Percentile pour ignorer les abeilles)
    master_bg = np.percentile(stack_array, percentile, axis=0).astype(np.uint8)
    
    # 2. La Tolérance (Écart-Type / Variance)
    # On calcule la STD. Les zones qui bougent beaucoup (bords) auront une valeur haute.
    master_std = np.std(stack_array, axis=0).astype(np.float32)
    
    # Normalisation de la STD pour l'affichage/sauvegarde (optionnel mais utile pour debug)
    master_std_visu = cv2.normalize(master_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return master_bg, master_std

# --- 3. DÉTECTION DE TEXTURE (GABOR)  ---
def detect_texture_gabor(img_gray):
    """Détecte les ruptures de pattern hexagonal (abeilles)."""
    # Filtres de Gabor optimisés pour casser les hexagones
    gabor_accum = np.zeros_like(img_gray, dtype=np.float32)
    ksize = 21 # Taille du noyau adaptée aux alvéoles (~20-30px)
    
    # On teste 4 orientations pour attraper les bords des abeilles dans tous les sens
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
        gabor_accum += np.abs(fimg) # Accumulation d'énergie
        
    # Normalisation
    cv2.normalize(gabor_accum, gabor_accum, 0, 255, cv2.NORM_MINMAX)
    return gabor_accum.astype(np.uint8)

# --- 4. PIPELINE FINAL HYBRIDE ---
def process_hybrid_detection(image_path, master_bg, master_std, r, threshold_zscore, output_folder):
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # A. Alignement
    ref_small = cv2.resize(master_bg, None, fx=scale, fy=scale)
    img_aligned = worker_align_ecc(image_path, ref_small, scale)
    if img_aligned is None: img_aligned = img_original

    # --- BRANCHE 1 : Z-SCORE (Intensité Intelligente) ---
    # Calcul de la différence
    diff = cv2.absdiff(img_aligned, master_bg).astype(np.float32)
    
    # Calcul du Z-Score : Différence pondérée par la volatilité locale
    # On ajoute un petit epsilon (+5) pour éviter de diviser par 0 et réduire le bruit sur les zones trop calmes
    z_score_map = diff / (master_std + 5.0)
    
    # Normalisation du Z-Score pour en faire une heatmap 0-255
    # Un z-score de 3 (très significatif) deviendra brillant
    heatmap_z = cv2.normalize(z_score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Filtrage fréquentiel sur le Z-Score (pour nettoyer la grille résiduelle)
    dft = cv2.dft(np.float32(heatmap_z), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = heatmap_z.shape
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (cols//2, rows//2), r, (1, 1), -1)
    img_back = cv2.idft(np.fft.ifftshift(dft_shift * mask_fft))
    heatmap_z_clean = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    heatmap_z_clean = cv2.normalize(heatmap_z_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- BRANCHE 2 : TEXTURE (Gabor) ---
    # Détecte les abeilles qui ont la même couleur que le fond mais une texture différente
    heatmap_texture = detect_texture_gabor(img_aligned)
    
    # On inverse Gabor car les abeilles (floues) ont souvent MOINS d'énergie haute fréquence que les alvéoles nettes
    # OU on l'utilise tel quel si elles créent des ruptures.
    # Ici, on utilise la VARIANCE locale de Gabor :
    # Les zones calmes (abeilles) vs zones actives (alvéoles).
    # Simplifions : On fusionne juste les heatmaps.
    
    # --- FUSION & DÉCISION ---
    # On donne plus de poids au Z-Score (0.7) qu'à la texture (0.3)
    final_heatmap = cv2.addWeighted(heatmap_z_clean, 0.7, heatmap_texture, 0.3, 0)
    
    # Morphologie pour remplir les corps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_heatmap = cv2.morphologyEx(final_heatmap, cv2.MORPH_CLOSE, kernel)

    # Seuillage final
    mask_bees = final_heatmap > threshold_zscore

    # --- VISUALISATION ---
    result_bgr = cv2.cvtColor(img_aligned, cv2.COLOR_GRAY2BGR)
    
    # Contours verts pour bien voir les détections
    contours, _ = cv2.findContours(mask_bees.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contours, -1, (0, 255, 0), 2)
    # Remplissage rouge semi-transparent
    overlay = result_bgr.copy()
    overlay[mask_bees] = [0, 0, 255]
    cv2.addWeighted(overlay, 0.4, result_bgr, 0.6, 0, result_bgr)

    # Création du panneau de contrôle
    hm_visu = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)
    std_visu = cv2.applyColorMap(cv2.normalize(master_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    
    cv2.putText(hm_visu, "Heatmap Z-Score+Gabor", (30, 80), 1, 3, (255, 255, 255), 3)
    cv2.putText(std_visu, "Carte de Variance (Tolerance)", (30, 80), 1, 3, (255, 255, 255), 3)
    
    # Sauvegarde : [Original | Variance Map | Heatmap | Résultat]
    # Variance Map est très instructive : elle montre où l'algo est "indulgent"
    top_row = cv2.hconcat([cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR), std_visu])
    bot_row = cv2.hconcat([hm_visu, result_bgr])
    # Redimensionner pour assembler si besoin, ici on suppose même taille
    grid = cv2.vconcat([top_row, bot_row])
    
    # Resize pour affichage écran
    preview = cv2.resize(grid, (1800, 1200)) # Taille fixe raisonnable

    filename = f"HYBRID_ZSCORE_t{threshold_zscore}_{os.path.basename(image_path)}"
    cv2.imwrite(os.path.join(output_folder, filename), grid)
    print(f"--> Traité : {filename}")

scale=0.2
num_samples=100
percentile=90
r=80
threshold_val=50

# ================= MAIN =================
if __name__ == '__main__':
    FOLDER_SOURCE = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
    FOLDER_TEST = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_test'
    FOLDER_OUTPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Resultats_Test_v3'
    
    if not os.path.exists(FOLDER_OUTPUT): os.makedirs(FOLDER_OUTPUT)

    # 1. Construction du modèle STATISTIQUE (Fond + Variance)
    # Augmentez num_samples si possible (150 est idéal pour une bonne variance)
    master_bg, master_std = build_statistical_model(FOLDER_SOURCE, num_samples, percentile)
    
    if master_bg is not None:
        cv2.imwrite(os.path.join(FOLDER_OUTPUT, "MASTER_BG.png"), master_bg)
        # La carte de variance vous montrera les zones "instables" de la ruche
        norm_std = cv2.normalize(master_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(FOLDER_OUTPUT, "MASTER_STD.png"), norm_std)
        
        # 2. Test
        test_files = glob.glob(os.path.join(FOLDER_TEST, "*.png"))[:10]
        
        # Le seuil Z-Score est différent d'un seuil 0-255 classique.
        # Commencez bas (20-30) car nous avons normalisé le Z-Score.
        for img in test_files:
            process_hybrid_detection(img, master_bg, master_std, r, threshold_val, output_folder=FOLDER_OUTPUT)