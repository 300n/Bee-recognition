import skimage
from retinex import msrcr
import cv2


def main():
    img_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop/M01C02_000066.png"
    save_path = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output//retinex/retinex_result_bande.png"

    # 1. Lecture de l'image complète (en RGB)
    img_original = skimage.io.imread(img_path)

    # 2. Extraction de la bande supérieure (Hauteur: 0 à 74, Largeur: 0 à 1691)
    bande_superieure = img_original[0:74, 0:1691]

    # 3. Application du filtre UNIQUEMENT sur cette bande
    bande_filtree = msrcr(
        bande_superieure,
        sigmas=(25.0, 50.0, 100.0),
    )

    # 4. Normalisation de la bande filtrée (pour repasser en format 8-bits standard)
    if bande_filtree.dtype != "uint8":
        bande_filtree = cv2.normalize(
            bande_filtree, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    # 5. Remplacement de l'ancienne bande par la nouvelle bande filtrée
    # dans l'image d'origine
    img_original[0:74, 0:1691] = bande_filtree

    # 6. Conversion de TOUTE l'image (RGB -> BGR) pour la sauvegarde
    img_final_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    # 7. Sauvegarde
    cv2.imwrite(save_path, img_final_bgr)
    print(f"✅ Image avec bande filtrée sauvegardée dans : {save_path}")


if __name__ == "__main__":
    main()
