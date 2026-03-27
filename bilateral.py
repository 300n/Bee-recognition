import cv2
import os
import itertools
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

# Chemin de votre image source (à vérifier)
img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"

# Dossier où les dizaines de tests seront sauvegardés
output_dir = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/tests_bilateral_grille_2"
)

# Création du dossier s'il n'existe pas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# LES PARAMÈTRES À TESTER
# ==========================================
# On définit des listes de valeurs à tester.
# d : de 9 (petit patch) à 50 (très grand patch, va lisser de larges zones)
d_values = [9, 15, 30, 50]

# sigmaColor : de 50 (préserve les contrastes) à 200 (mélange presque tout, casse les arêtes des alvéoles)
sigma_color_values = [50, 100, 150, 200]

# sigmaSpace : de 50 (lissage local) à 200 (lissage sur une très grande distance)
sigma_space_values = [50, 100, 200]

# Création de toutes les combinaisons possibles (4 x 4 x 3 = 48 tests)
combinations = list(itertools.product(d_values, sigma_color_values, sigma_space_values))

# ==========================================
# EXECUTION
# ==========================================

# Lecture de l'image (en niveaux de gris vu votre image source)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"ERREUR : Impossible de lire l'image à {img_path}")
    exit()

print(f"Début des tests : {len(combinations)} combinaisons vont être générées.")
print("Cela devrait prendre une à deux minutes avec OpenCV...\n")

# Boucle sur toutes les combinaisons avec une barre de progression
for d, sc, ss in tqdm(combinations, desc="Génération des tests"):

    # Application du filtre bilatéral OpenCV
    filtered = cv2.bilateralFilter(img, d, sc, ss)

    # Création du nom de fichier dynamique pour s'y retrouver
    # Exemple : "test_d30_sc150_ss100.png"
    filename = f"test_d{d}_sc{sc}_ss{ss}.png"
    save_path = os.path.join(output_dir, filename)

    # Sauvegarde
    cv2.imwrite(save_path, filtered)

print(f"\nTerminé ! Allez ouvrir le dossier :")
print(output_dir)
print(
    "Regardez les images en grand format pour trouver celle qui isole le mieux vos abeilles."
)
