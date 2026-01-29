import cv2
import numpy as np
import os

def create_selective_red_view(image_path, r, threshold_val, output_folder):
    # 1. Chargement et vérification
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Erreur de lecture.")
        return
    
    # 2. Génération du détecteur (FFT)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (ccol, crow), r, (1, 1), -1)
    
    fshift = dft_shift * mask_fft
    img_back = cv2.idft(np.fft.ifftshift(fshift))
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # 3. Création du masque d'abeilles (Points chauds)
    # Normalisation pour occuper tout l'espace 0-255
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    # Inversion : les abeilles deviennent blanches sur fond noir
    bee_presence = 255 - img_back.astype(np.uint8)
    
    # --- LA CORRECTION EST ICI ---
    # On utilise un seuillage binaire strict pour isoler les abeilles
    # Si l'image est encore trop rouge, augmentez 'threshold_val' (ex: 120, 150)
    _, mask_binary = cv2.threshold(bee_presence, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Un petit flou sur le masque pour adoucir les bords du rouge
    mask_binary = cv2.GaussianBlur(mask_binary, (7, 7), 0)
    # -----------------------------

    # 4. Construction de l'image hybride
    # Image de base en couleur (mais qui contient du gris)
    final_view = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Création d'une couche rouge pur
    red_overlay = np.zeros_like(final_view)
    red_overlay[:, :] = (0, 0, 255) # BGR
    
    # Application du rouge UNIQUEMENT là où le masque est blanc
    # On utilise le masque comme un coefficient de mélange (alpha)
    alpha = mask_binary.astype(float) / 255.0
    for c in range(3): # Pour chaque canal B, G, R
        final_view[:, :, c] = (final_view[:, :, c] * (1 - alpha * 0.8) + 
                               red_overlay[:, :, c] * (alpha * 0.8)).astype(np.uint8)

    # 5. Sauvegarde
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"RED_FOCUS_r={r}_t={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_view)
    print(f"Succès : {filename}")

# Paramètres suggérés : r petit (50) et seuil élevé (100+) pour éviter le "tout rouge"
PATH = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
OUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output'

# Testez avec un seuil plus haut si l'image est trop rouge
create_selective_red_view(PATH, r=55, threshold_val=130, output_folder=OUT)