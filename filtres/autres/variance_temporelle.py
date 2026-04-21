import cv2
import numpy as np
import os
import glob

def compute_variance_map(image_paths):
    """Calcule l'écart-type pixel par pixel sur un lot d'images."""
    samples = []
    
    print(f"Lecture de {len(image_paths)} images...")
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Si tes images sont déjà dans 'images_crop', elles font déjà 2486x3536
            # On vérifie quand même la taille pour éviter les erreurs
            samples.append(img)
        else:
            print(f"Attention : impossible de lire {p}")
    
    if len(samples) < 2:
        print("Erreur : Pas assez d'images valides pour calculer une variance.")
        return None

    # Transformation en pile 3D
    samples_array = np.stack(samples, axis=0)
    
    # Calcul de l'écart-type (Standard Deviation)
    # C'est ce qui isole les abeilles (mouvement) de la ruche (statique)
    std_map = np.std(samples_array, axis=0)
    
    # Normalisation pour l'affichage (0-255)
    std_map_norm = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(std_map_norm)

# --- CONFIGURATION DES CHEMINS ---
folder_path = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
# On récupère la liste des 50 premières images .png (ou .jpg)
files = glob.glob(os.path.join(folder_path, "*.png"))[:50]

if not files:
    print("Aucun fichier trouvé. Vérifie l'extension (.png ou .jpg) !")
else:
    # On génère la carte de variance
    result_map = compute_variance_map(files)
    
    if result_map is not None:
        cv2.imshow('Carte de Variance Temporelle', cv2.resize(result_map, (800, 600)))
        cv2.imwrite('variance_result.png', result_map)
        cv2.waitKey(0)