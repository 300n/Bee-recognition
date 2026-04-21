import cv2
import numpy as np
import glob
import os

def compute_temporal_std(image_folder, num_frames=100):
    # 1. Sélectionner un échantillon d'images
    paths = glob.glob(os.path.join(image_folder, "*.png"))[:num_frames]
    
    # 2. Charger les images dans une pile (stack)
    frames = []
    print(f"Analyse de la dynamique sur {len(paths)} images...")
    for p in paths:
        f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if f is not None:
            frames.append(f)
            
    # Empilement 3D (Temps, Hauteur, Largeur)
    stack = np.stack(frames, axis=0)
    
    # 3. Calcul de l'écart-type sur l'axe du temps (axis 0)
    # C'est ici que la magie opère : les alvéoles disparaissent !
    std_map = np.std(stack, axis=0)
    
    # Normalisation pour obtenir une image exploitable
    std_map = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return std_map

# --- Application ---
folder = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'
# Cette carte te montre TOUTES les zones où les abeilles sont passées
activity_map = compute_temporal_std(folder, num_frames=50)

cv2.imwrite('activity_map.png', activity_map)
cv2.imshow('Carte d Activite (Bees only)', cv2.resize(activity_map, (800, 600)))
cv2.waitKey(0)