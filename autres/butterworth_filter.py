import cv2
import numpy as np

def fft_sharper_filter(image, d0=100, n=2):
    """
    Filtre de Butterworth Passe-Bas.
    d0 : Rayon de coupure (plus il est grand, plus c'est net)
    n : Ordre du filtre (plus il est petit, plus la transition est douce)
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 1. FFT
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 2. Création du filtre de Butterworth
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Formule du Butterworth
    butterworth_lp = 1 / (1 + (distance / d0)**(2 * n))
    
    # Application du masque sur les deux canaux (réel/imaginaire)
    mask = np.repeat(butterworth_lp[:, :, np.newaxis], 2, axis=2)
    fshift = dft_shift * mask

    # 3. Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    # Normalisation
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# --- Test ---
img = cv2.imread('/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png', 0)
# Joue sur d0 pour la netteté (essaie 150 ou 200)
result = fft_sharper_filter(img, d0=150, n=2)


cv2.imshow('FFT Butterworth (Plus Net)', cv2.resize(result, (800, 800)))
cv2.waitKey(0)