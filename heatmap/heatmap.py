import cv2
import numpy as np

def create_bee_heatmap(original_img, r=50):
    # 1. Génération de l'image floue (ton détecteur)
    dft = cv2.dft(np.float32(original_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = original_img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # 2. Inversion et Normalisation pour créer la Heatmap
    # On veut que les abeilles (points noirs) deviennent des zones claires (points blancs)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_gray = 255 - img_back.astype(np.uint8) # Inversion
    
    # 3. Application d'un seuillage pour nettoyer le fond
    # On ne garde que les "points chauds" les plus intenses
    _, heatmap_gray = cv2.threshold(heatmap_gray, 70, 255, cv2.THRESH_TOZERO)
    
    # 4. Colorisation de la Heatmap (Gris -> Couleur)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    
    # 5. Fusion avec l'image originale nette
    # On convertit l'image originale en BGR pour pouvoir mettre de la couleur dessus
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # alpha est la transparence de l'image originale, beta celle de la heatmap
    combined = cv2.addWeighted(original_bgr, 0.7, heatmap_color, 0.3, 0)
    
    return combined, heatmap_gray

# --- Exécution ---
path = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
img_net_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img_net_gray is not None:
    result_heatmap, mask_only = create_bee_heatmap(img_net_gray, r=60)
    
    # Affichage du résultat
    cv2.imshow('Heatmap Abeilles sur Image Nette', cv2.resize(result_heatmap, (1000, 800)))
    cv2.imwrite('bee_heatmap_result.jpg', result_heatmap)
    cv2.waitKey(0)