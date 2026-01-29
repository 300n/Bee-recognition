import cv2
import numpy as np

# --- Chargement & recadrage (pour virer les LED, etc.) ---
img = cv2.imread("/Users/valentindaveau/Downloads/1499.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
# Exemple : on enlève 10% haut/bas et 5% gauche/droite (à adapter)
crop = img[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]

# --- 1) Lissage léger pour réduire le bruit ---
denoised = cv2.medianBlur(crop, 3)  # ou 5

# --- 2) Correction d’illumination (estimation du fond) ---
# On estime le "fond" lentement variant avec un flou gaussien de grand rayon
background = cv2.GaussianBlur(denoised, (101, 101), 0)
# Normalisation type homomorphique : division
illum_corr = cv2.divide(denoised, background, scale=255)

# --- 3) Contraste local : CLAHE ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(illum_corr)

# --- 4) Renforcement des abeilles par morphologie (black-hat) ---
# Abeilles = objets sombres sur fond plus clair
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
blackhat = cv2.morphologyEx(clahe_img, cv2.MORPH_BLACKHAT, kernel)

# --- 5) Exemple : image 3 canaux pour le réseau ---
# canal 0 : image corrigée & CLAHE
# canal 1 : black-hat (met en valeur les abeilles)
# canal 2 : gradient (contours)
sobelx = cv2.Sobel(clahe_img, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(clahe_img, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(sobelx, sobely)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

multi_channel = cv2.merge([clahe_img, blackhat, grad])
# -> multi_channel.shape = (H, W, 3), prêt pour un backbone CNN standard

# --- 6) Visualisation comparaison image brute / image traitée ---
# Redimensionnement pour affichage (OpenCV aime les tailles raisonnables)
display_raw = cv2.resize(crop, (800, 800))
display_proc = cv2.resize(clahe_img, (800, 800))

# Normalisation pour être sûr d'un affichage correct
display_raw = cv2.normalize(display_raw, None, 0, 255, cv2.NORM_MINMAX)
display_proc = cv2.normalize(display_proc, None, 0, 255, cv2.NORM_MINMAX)

# Concaténation côte à côte
comparison = np.hstack((display_raw, display_proc))

cv2.imshow("Comparaison - Gauche : brute | Droite : traitée", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()