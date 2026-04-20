import cv2
import numpy as np
import os
import itertools
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_kramer_bruckner"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# IMPLÉMENTATION KRAMER-BRÜCKNER
# ==========================================

def apply_kb(img, radius, center_weight):
    """
    Kramer-Brückner filter.
    Identique au SNN mais le pixel central est pondéré center_weight fois
    avant d'accumuler les voisins élus. center_weight=1 → double-pondération
    standard (Kramer 1975) ; center_weight=0 → SNN pur.
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

    accum = img_f * float(center_weight)
    count = np.full_like(img_f, float(center_weight))
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

# radius : rayon du voisinage
radius_values = [1, 2, 3, 4]

# center_weight : poids du pixel central
# 0 → SNN pur (aucun poids centre) ; 1 → Kramer standard (double-pondération)
# 2 → triple-pondération (préserve encore plus les points isolés)
center_weight_values = [1, 2, 3]

# passes : nombre d'applications successives
passes_values = [1, 2, 3]

combinations = list(itertools.product(radius_values, center_weight_values, passes_values))

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

print(f"Début des tests : {len(combinations)} combinaisons.")

for radius, cw, passes in tqdm(combinations, desc="Kramer-Brückner"):
    result = img.copy()
    for _ in range(passes):
        result = apply_kb(result, radius, cw)
    filename = f"test_r{radius}_cw{cw}_passes{passes}.png"
    cv2.imwrite(os.path.join(output_dir, filename), result)

print(f"\nTerminé ! Résultats dans :\n{output_dir}")
print("Conseil : r=2, cw=1, passes=1 est le point de départ (Kramer 1975 standard).")
