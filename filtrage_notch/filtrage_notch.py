import cv2
import numpy as np

def fft_remove_honeycomb(image):
    # 1. Passage en flottant et calcul de la FFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 2. Création d'un filtre (Masque)
    # Pour automatiser, on peut seuiller le spectre de puissance
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    
    # On crée un masque pour supprimer les hautes fréquences régulières
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.ones((rows, cols, 2), np.uint8)
    
    # On définit un rayon autour du centre pour préserver les formes basses fréquences (les abeilles)
    # Tout ce qui est à l'extérieur (les détails fins des alvéoles) sera atténué
    r = 60 
    center_mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(center_mask, (ccol, crow), r, (1, 1), -1)
    
    # Filtrage passe-bas : on ne garde que le centre (les formes globales)
    dft_shift = dft_shift * center_mask

    # 3. Transformée inverse
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalisation
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# --- Correction de l'appel ---
image_path = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # On charge l'image ici

if img is not None:
    # On passe l'objet 'img' (le tableau de pixels) et non le texte 'image_path'
    image_fft = fft_remove_honeycomb(img) 
    
    cv2.imshow('Resultat FFT', cv2.resize(image_fft, (800, 800)))
    cv2.waitKey(0)
else:
    print(f"Erreur : Impossible de lire l'image à l'emplacement {image_path}")