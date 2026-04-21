import cv2
import numpy as np
import os
import random

def compute_background_model(image_folder, num_samples):
    """
    Calcule la médiane temporelle pour créer une image de la ruche 'vide'.
    """
    all_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    sample_paths = random.sample(all_images, min(num_samples, len(all_images)))
    
    samples = []
    print(f"Chargement de {num_samples} échantillons pour le modèle de fond...")
    
    for path in sample_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            samples.append(img)
    
    # Calcul de la médiane sur l'axe temporel (stack d'images)
    # Note : Sur 3536x3536, cela demande de la RAM. 
    # Si ça plante, on peut traiter par blocs ou redimensionner.
    samples_array = np.stack(samples, axis=0)
    background_median = np.median(samples_array, axis=0).astype(np.uint8)
    
    return background_median

# --- Paramétrage ---
folder_path = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
bg_model = compute_background_model(folder_path,200)
cv2.imwrite('hive_background_reference.jpg', bg_model)

def isolate_bees(current_image_path, background_model):
    # 1. Chargement et normalisation lumineuse
    frame = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    frame_clahe = clahe.apply(frame)
    bg_clahe = clahe.apply(background_model)

    # 2. Soustraction de fond (Soustraction absolue)
    # On obtient la différence entre l'image actuelle et la ruche vide
    diff = cv2.absdiff(frame_clahe, bg_clahe)

    # 3. Seuillage (Thresholding) pour éliminer le bruit résiduel des alvéoles
    # On utilise le seuillage d'Otsu pour trouver automatiquement le meilleur seuil
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Nettoyage morphologique (supprimer les pixels isolés)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. Masquage (Optionnel) : On applique le masque sur l'image CLAHE 
    # pour garder les détails des abeilles sur fond noir
    result = cv2.bitwise_and(frame_clahe, frame_clahe, mask=clean)

    return result, clean

# --- Test sur une image ---
final_bees, mask = isolate_bees('/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png', bg_model)

# Visualisation rapide
cv2.imshow('Abeilles Isolees', cv2.resize(final_bees, (800, 800)))
cv2.waitKey(0)