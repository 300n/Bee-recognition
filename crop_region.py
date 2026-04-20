import cv2
import os

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/crop_region"
output_name = "crop_640x640_centre_droit.png"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# PARAMÈTRES DU CROP
# ==========================================

CROP_W = 640
CROP_H = 640
TOP    = 350   # bande à exclure en haut (px)

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

H, W = img.shape
x_start = W // 2
y_start = TOP
x_end   = x_start + CROP_W
y_end   = y_start + CROP_H

# Vérifications
assert x_end <= W, f"Dépassement à droite : {x_end} > {W}"
assert y_end <= H, f"Dépassement en bas : {y_end} > {H}"

crop = img[y_start:y_end, x_start:x_end]

save_path = os.path.join(output_dir, output_name)
cv2.imwrite(save_path, crop)

print(f"Image source    : {W}×{H}")
print(f"Région cropée   : x=[{x_start}:{x_end}]  y=[{y_start}:{y_end}]")
print(f"Taille résultat : {crop.shape[1]}×{crop.shape[0]}")
print(f"Marges exclues  : gauche={x_start}px  haut={y_start}px  droite={W-x_end}px  bas={H-y_end}px")
print(f"Sauvegardé dans : {save_path}")
