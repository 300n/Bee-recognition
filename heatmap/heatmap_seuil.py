import cv2
import numpy as np
import os

def generate_conditional_heatmap(image_path, r, threshold_val, output_folder):
    # 1. Chargement de l'image originale (Nette)
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None: return

    # 2. Calcul de la "Chaleur" (Heatmap) via FFT
    rows, cols = img_original.shape
    crow, ccol = rows // 2, cols // 2
    
    dft = cv2.dft(np.float32(img_original), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    mask_fft = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask_fft, (ccol, crow), r, (1, 1), -1)
    
    fshift = dft_shift * mask_fft
    img_back = cv2.idft(np.fft.ifftshift(fshift))
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalisation et Inversion : Les abeilles sont proches de 255 (chaud)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_values = 255 - img_back.astype(np.uint8)

    # 3. Création de l'image de sortie (3 canaux pour la couleur)
    # On commence avec l'image originale en niveaux de gris dupliquée sur 3 canaux
    result = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

    # 4. Application du seuil et de la couleur
    # On crée un masque binaire là où la chaleur dépasse le seuil
    mask_bees = heatmap_values > threshold_val

    # Pour ces pixels, on remplace la valeur par du rouge
    # On peut utiliser la valeur de heatmap_values pour varier l'intensité du rouge
    result[mask_bees, 0] = 0                # Canal Bleu
    result[mask_bees, 1] = 0                # Canal Vert
    result[mask_bees, 2] = heatmap_values[mask_bees]  # Canal Rouge (intensité liée à la chaleur)

    display_res = cv2.resize(result, (1000, 750)) 
    cv2.imshow(f"Detection r={r} t={threshold_val}", display_res)
    
    print("Appuyez sur une touche dans la fenêtre d'image pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """
    # 5. Sauvegarde
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_folder, f"FINAL_CONDITIONAL_r{r}_t{threshold_val}_{base_name}.png")
    cv2.imwrite(save_path, result)
    print(f"Image générée : {save_path}")
    """

# --- Paramètres ---
# Si l'image est trop rouge, monte le seuil (ex: 150)
# Si elle n'est pas assez rouge, descends-le (ex: 80)
R_VAL = 40
THRESHOLD_VAL = 70 
INPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
OUTPUT = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output'

generate_conditional_heatmap(INPUT, R_VAL, THRESHOLD_VAL, OUTPUT)