import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def test_bilateral_filter(image_path, d=15, sigmaColor=45, sigmaSpace=15):
    """
    Applique le filtre bilatéral sur une image et affiche le résultat.

    Paramètres adaptés pour une image 1920x1200 :
    - d : Diamètre du voisinage pixel. 15 est un bon compromis lissage/vitesse.
    - sigmaColor : Tolérance de différence de gris. À 45, on efface le bruit de la cire,
                   mais on préserve la transition abelle/cire (qui a une différence > 100).
    - sigmaSpace : Étendue du flou spatial. Environ égal à 'd'.
    """
    if not os.path.exists(image_path):
        print(f"Erreur : Image introuvable à {image_path}")
        return

    print(f"Traitement de {os.path.basename(image_path)}...")

    # 1. Chargement en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Application du filtre Bilatéral
    # cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    img_filtered = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # 3. Calcul de la différence (pour voir le bruit retiré)
    # On ajoute 128 pour que le fond neutre soit gris (facilite la visualisation)
    diff = cv2.absdiff(img, img_filtered)
    diff_visu = cv2.add(diff, 128)

    # 4. Affichage Matplotlib
    plt.figure(figsize=(18, 6))

    # Titre global avec les paramètres
    plt.suptitle(
        f"Filtre Bilatéral (d={d}, sigColor={sigmaColor}, sigSpace={sigmaSpace})",
        fontsize=16,
        fontweight="bold",
    )

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("1. Originale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_filtered, cmap="gray")
    plt.title("2. Filtrée (Bords préservés)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    # Colormap 'magma' pour bien voir les résidus de bruit
    plt.imshow(diff_visu, cmap="magma")
    plt.title("3. Bruit supprimé (Différence)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # (Optionnel) Sauvegarde
    # cv2.imwrite("bilateral_result.png", img_filtered)


# ================= EXÉCUTION =================
if __name__ == "__main__":
    # Remplacez par le chemin vers une de vos images aléatoires 1920x1200
    IMAGE_TEST = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images/M01C01_005601.png"

    # Lancement avec les paramètres recommandés pour vos abeilles
    test_bilateral_filter(IMAGE_TEST, d=15, sigmaColor=45, sigmaSpace=15)
