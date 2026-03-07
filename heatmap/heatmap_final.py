import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_IMAGE_PATH_MC01 = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images1_crop/M01C01_002182.png"
OUTPUT_DIR_MC01 = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_final_MC01"
)

INPUT_IMAGE_PATH_MC02 = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_030004.png"
OUTPUT_DIR_MC02 = (
    "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap_final_MC01"
)

r = 40
t = 30


def generate_and_save_heatmap(img_gray, r, threshold_val, output_folder, base_name):
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # 1. Détecteur flou (FFT)
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 2. Heatmap
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_raw = 255 - img_back.astype(np.uint8)
    _, heatmap_thresh = cv2.threshold(
        heatmap_raw, threshold_val, 255, cv2.THRESH_TOZERO
    )
    heatmap_color = cv2.applyColorMap(heatmap_thresh, cv2.COLORMAP_JET)

    # 3. Overlay et Sauvegarde
    img_bgr_net = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr_net, 0.6, heatmap_color, 0.4, 0)

    filename = f"heatmap_r={r}_t_={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_MC01):
        os.makedirs(OUTPUT_DIR_MC01)
        print(f"Dossier créé : {OUTPUT_DIR_MC01}")

    img_net = cv2.imread(INPUT_IMAGE_PATH_MC01, cv2.IMREAD_GRAYSCALE)

    if img_net is None:
        print(f"Impossible de charger l'image : {INPUT_IMAGE_PATH_MC01}")
    else:
        base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH_MC01))[0]

        print("Génération de la heatmap ...")
        fname = generate_and_save_heatmap(img_net, r, t, OUTPUT_DIR_MC01, base_name)
        print(f"Généré : {fname}")
