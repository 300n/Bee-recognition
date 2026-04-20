import cv2
import os
import itertools
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/tests_nlm"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# LES PARAMÈTRES À TESTER
# ==========================================

# h : force du filtre (doit ≈ bruit σ ≈ 9 sur nos images)
# Trop bas → bruit résiduel ; trop haut → flou des contours
h_values = [3, 5, 7, 9, 12, 15]

# templateWindowSize : taille du patch de comparaison (impair)
# Petit → plus discriminant mais bruit ; grand → plus stable mais coûteux
template_values = [5, 7, 9]

# searchWindowSize : zone de recherche des patchs similaires (impair)
# 21 = standard ; 27 → meilleure dénoisation mais 2× plus lent
search_values = [15, 21, 27]

combinations = list(itertools.product(h_values, template_values, search_values))

# ==========================================
# EXECUTION
# ==========================================

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

print(f"Début des tests : {len(combinations)} combinaisons.")

for h, tmpl, srch in tqdm(combinations, desc="NLM"):
    filtered = cv2.fastNlMeansDenoising(
        img, h=h, templateWindowSize=tmpl, searchWindowSize=srch
    )
    filename = f"test_h{h}_tmpl{tmpl}_srch{srch}.png"
    cv2.imwrite(os.path.join(output_dir, filename), filtered)

print(f"\nTerminé ! Résultats dans :\n{output_dir}")
print("Cherchez le meilleur compromis : contours nets + bruit supprimé.")
