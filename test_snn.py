import cv2
import numpy as np
import os
import itertools
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_snn"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# IMPLÉMENTATION SNN
# ==========================================

def apply_snn(img, radius):
    """
    Symmetric Nearest Neighbour filter.
    Pour chaque paire symétrique (a, b) autour du centre c,
    on garde le plus proche de c. On fait la moyenne des élus.
    """
    h, w = img.shape
    img_f = img.astype(np.float32)
    pad = cv2.copyMakeBorder(img_f, radius, radius, radius, radius, cv2.BORDER_REFLECT)

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]
    pairs, seen = [], set()
    for dy, dx in offsets:
        if (-dy, -dx) not in seen:
            pairs.append(((dy, dx), (-dy, -dx)))
            seen.add((dy, dx))

    accum = np.zeros_like(img_f)
    count = np.zeros_like(img_f)
    for (dy1, dx1), (dy2, dx2) in pairs:
        a = pad[radius + dy1 : radius + dy1 + h, radius + dx1 : radius + dx1 + w]
        b = pad[radius + dy2 : radius + dy2 + h, radius + dx2 : radius + dx2 + w]
        closer = np.where(np.abs(a - img_f) <= np.abs(b - img_f), a, b)
        accum += closer
        count += 1.0

    return np.clip(accum / count, 0, 255).astype(np.uint8)

# ==========================================
# LES PARAMÈTRES À TESTER
# ==========================================

# radius : rayon du voisinage (1 = 8 voisins, 2 = 24, 3 = 48)
# r=1 → très doux (4 paires) ; r=3 → risque de traverser les bords des alvéoles
radius_values = [1, 2, 3, 4]

# passes : appliquer le filtre N fois de suite (lissage progressif)
passes_values = [1, 2, 3]

combinations = list(itertools.product(radius_values, passes_values))

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

print(f"Début des tests : {len(combinations)} combinaisons.")

for radius, passes in tqdm(combinations, desc="SNN"):
    result = img.copy()
    for _ in range(passes):
        result = apply_snn(result, radius)
    filename = f"test_r{radius}_passes{passes}.png"
    cv2.imwrite(os.path.join(output_dir, filename), result)

print(f"\nTerminé ! Résultats dans :\n{output_dir}")
print("Conseil : r=2, passes=1 est le point de départ (edge_pres=0.940).")
