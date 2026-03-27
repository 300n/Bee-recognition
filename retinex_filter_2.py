import skimage
from retinex import msrcr
import cv2
import numpy as np


def main():
    img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
    save_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/retinex/retinex_result_optimise2.png"

    img_original = skimage.io.imread(img_path)
    bande_superieure = img_original[0:74, 0:1691]

    # 1. Sigmas réduits, adaptés à une hauteur de 74 pixels
    bande_filtree = msrcr(
        bande_superieure,
        sigmas=(2.0, 5.0, 15.0),
    )

    # 2. Normalisation classique de la sortie Retinex
    if bande_filtree.dtype != "uint8":
        bande_filtree = cv2.normalize(
            bande_filtree, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    # 3. LE BLENDING (Mélange pondéré)
    # On mixe la bande originale (sombre, garde le noir des abeilles)
    # avec la bande filtrée (claire, révèle la ruche).
    # poids_original = 0.4 (40%) | poids_retinex = 0.6 (60%)
    bande_mixte = cv2.addWeighted(bande_superieure, 0.4, bande_filtree, 0.6, 0)

    # On remplace dans l'image d'origine
    img_original[0:74, 0:1691] = bande_mixte

    # Conversion BGR pour la sauvegarde
    img_final_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, img_final_bgr)
    print(f"✅ Image optimisée sauvegardée dans : {save_path}")


if __name__ == "__main__":
    main()
