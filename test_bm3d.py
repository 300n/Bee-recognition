import cv2
import numpy as np
import os
import bm3d
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_bm3d"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# LES PARAMÈTRES À TESTER
# ==========================================

# sigma_psd : niveau de bruit supposé (normalisé sur [0,1])
# Le bruit estimé sur nos images est σ ≈ 9/255 ≈ 0.035
# En dessous de 5/255 → quasi pas de débruitage
# Au-dessus de 25/255 → sur-lissage, perte de texture alvéoles
sigma_values = [5, 8, 10, 12, 15, 20, 25, 35]

# stage : passes de l'algorithme
# ALL_STAGES    → 2 passes (hard-thresholding + Wiener) — meilleure qualité
# HARD_THRESHOLDING → 1 passe — plus rapide, légèrement moins bon
stages = {
    "full": bm3d.BM3DStages.ALL_STAGES,
    "hard": bm3d.BM3DStages.HARD_THRESHOLDING,
}

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

img_f = img.astype(np.float32) / 255.0

combinations = [(s, sname, stage) for s in sigma_values for sname, stage in stages.items()]
print(f"Début des tests : {len(combinations)} combinaisons.")

for sigma, stage_name, stage in tqdm(combinations, desc="BM3D"):
    denoised = bm3d.bm3d(img_f, sigma_psd=sigma / 255.0, stage_arg=stage)
    result = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    filename = f"test_sigma{sigma}_{stage_name}.png"
    cv2.imwrite(os.path.join(output_dir, filename), result)

print(f"\nTerminé ! Résultats dans :\n{output_dir}")
print("Conseil : sigma=10 + full est le point de départ optimal (σ≈9/255 estimé sur ces images).")
