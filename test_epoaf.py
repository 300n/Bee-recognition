import cv2
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_epoaf"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# IMPLÉMENTATION EPOAF
# ==========================================

def apply_epoaf(img, offsets, ksize=3):
    """
    Edge-Preserving Oriented Adaptive Filter.
    Lisse chaque pixel le long de la tangente au bord (direction Sobel ⊥).
    ksize : taille du noyau Sobel (3 ou 5)
    offsets : positions d'échantillonnage le long de la tangente
    """
    img_f = img.astype(np.float32)
    h, w = img.shape

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6

    # Tangente = perpendiculaire au gradient : (-gy/mag, gx/mag)
    tx = -gy / mag
    ty =  gx / mag

    xs = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    accum = img_f.copy()
    n = 1
    for s in offsets:
        map_x = np.clip(xs + s * tx, 0, w - 1).astype(np.float32)
        map_y = np.clip(ys + s * ty, 0, h - 1).astype(np.float32)
        accum += cv2.remap(img_f, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        n += 1
    return np.clip(accum / n, 0, 255).astype(np.uint8)

# ==========================================
# LES PARAMÈTRES À TESTER
# ==========================================

# offsets : positions d'échantillonnage le long de la tangente (en pixels)
# ±1       → très doux (3 points : centre + 2)
# ±1,±2    → standard (5 points)
# ±1,±2,±3 → plus fort (7 points), risque de traverser les bords
# ±2,±4    → lissage long-range (saute les voisins immédiats)
offset_configs = {
    "pm1":       (-1, 1),
    "pm1_2":     (-2, -1, 1, 2),
    "pm1_2_3":   (-3, -2, -1, 1, 2, 3),
    "pm2_4":     (-4, -2, 2, 4),
    "pm1_3":     (-3, -1, 1, 3),
}

# ksize : taille du noyau Sobel pour estimer la direction du bord
# 3 → sensible aux petits bords (bords des alvéoles)
# 5 → plus robuste au bruit, suit mieux les grands contours (corps des abeilles)
ksize_values = [3, 5]

# passes : nombre d'applications successives
passes_values = [1, 2]

combinations = [
    (off_name, off_vals, ks, passes)
    for off_name, off_vals in offset_configs.items()
    for ks in ksize_values
    for passes in passes_values
]

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

print(f"Début des tests : {len(combinations)} combinaisons.")

for off_name, off_vals, ks, passes in tqdm(combinations, desc="EPOAF"):
    result = img.copy()
    for _ in range(passes):
        result = apply_epoaf(result, off_vals, ksize=ks)
    filename = f"test_{off_name}_ks{ks}_passes{passes}.png"
    cv2.imwrite(os.path.join(output_dir, filename), result)

print(f"\nTerminé ! Résultats dans :\n{output_dir}")
print("Conseil : offsets=pm1_2, ksize=3, passes=1 est le point de départ (edge_pres=0.945).")
