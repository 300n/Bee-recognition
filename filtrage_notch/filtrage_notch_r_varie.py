import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_filter_with_r(image, r_value):
    # 1. FFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 2. Création du masque circulaire (Passe-bas)
    mask = np.zeros((rows, cols, 2), np.uint8)
    # On dessine un cercle blanc (1) sur fond noir (0)
    cv2.circle(mask, (ccol, crow), r_value, (1, 1), -1)
    
    # Application du masque
    fshift = dft_shift * mask

    # 3. FFT Inverse
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalisation
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# --- Configuration ---
image_path = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erreur : Image non trouvée.")
else:
    # Définition des 8 valeurs de r à tester
    # On part de 50 (ton test flou) jusqu'à 1200 (très net mais risqué)
    r_list = [50, 60]
    
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Analyse de l'influence du rayon r sur la netteté (FFT)", fontsize=16)

    for i, r in enumerate(r_list):
        # Traitement
        res = fft_filter_with_r(img, r)
        
        # Affichage dans la grille 2x4
        plt.subplot(1, 2, i + 1)
        plt.imshow(res, cmap='gray')
        plt.title(f"Rayon r = {r}")
        plt.axis('off')
        
        # Optionnel : Sauvegarde individuelle si besoin
        # cv2.imwrite(f'test_r_{r}.png', res)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()