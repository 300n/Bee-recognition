import cv2
import json
import os


def extract_bees_from_predictions(
    image_path, json_path, output_folder, confidence_threshold=0.5
):
    """
    Extrait les sous-images (crops) des abeilles à partir d'une image et d'un fichier JSON Roboflow.

    :param image_path: Chemin vers l'image originale.
    :param json_path: Chemin vers le fichier JSON contenant les prédictions.
    :param output_folder: Dossier où sauvegarder les abeilles découpées.
    :param confidence_threshold: Ne garder que les prédictions au-dessus de ce seuil de confiance.
    """
    # 1. Création du dossier de sortie
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Dossier créé : {output_folder}")

    # 2. Chargement de l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Erreur : Impossible de lire l'image '{image_path}'")
        return

    height_img, width_img, _ = img.shape

    # 3. Lecture du fichier JSON
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du JSON : {e}")
        return

    predictions = data.get("predictions", [])
    if not predictions:
        print("⚠️ Aucune prédiction trouvée dans le fichier JSON.")
        return

    print(f"🔍 {len(predictions)} prédictions trouvées. Début de l'extraction...")

    # 4. Boucle sur chaque prédiction
    count = 0
    for pred in predictions:
        # Filtrer par confiance (optionnel mais recommandé)
        if pred["confidence"] < confidence_threshold:
            continue

        # Roboflow fournit souvent (x, y) comme le CENTRE de la bounding box.
        # Il faut calculer les coordonnées (x_min, y_min, x_max, y_max)
        center_x = pred["x"]
        center_y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        # Calcul des coins supérieurs gauches et inférieurs droits
        x_min = int(center_x - (w / 2))
        y_min = int(center_y - (h / 2))
        x_max = int(center_x + (w / 2))
        y_max = int(center_y + (h / 2))

        # Sécurité : S'assurer que les coordonnées ne sortent pas de l'image
        # (Sinon OpenCV va planter lors du découpage)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width_img, x_max)
        y_max = min(height_img, y_max)

        # 5. Découpage (Crop) de l'image
        # En OpenCV/Numpy, l'image est un tableau indexé par [Y, X]
        bee_crop = img[y_min:y_max, x_min:x_max]

        # Vérifier que le crop n'est pas vide (arrive si la box est hors cadre)
        if bee_crop.size == 0:
            continue

        # 6. Sauvegarde de l'abeille
        # On utilise l'ID de détection pour avoir un nom de fichier unique
        # et on inclut la confiance pour référence
        conf_str = f"{pred['confidence']:.2f}".replace(".", "")
        filename = f"bee_{conf_str}_{pred['detection_id'][:8]}.png"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, bee_crop)
        count += 1

    print(
        f"✅ Terminé ! {count} abeilles ont été extraites avec succès dans '{output_folder}'."
    )


# ================= EXÉCUTION =================
if __name__ == "__main__":

    # --- Instructions préliminaires ---
    # 1. Sauvegardez votre texte JSON ci-dessus dans un fichier nommé "predictions.json"
    # 2. Placez l'image correspondante dans le même dossier
    # 3. Remplacez les chemins ci-dessous par les vôtres

    CHEMIN_IMAGE = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images/M01C01_016043.png"  # <-- Remplacez par le nom de votre image
    CHEMIN_JSON = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/predictions.json"  # <-- Le fichier contenant vos données
    DOSSIER_SORTIE = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/abeilles_extraites"  # <-- Dossier de destination

    # Seuil de confiance : ici réglé à 0.5 (50%).
    # Mettez 0.0 pour extraire absolument toutes les boxes du JSON.
    SEUIL_CONFIANCE = 0.0

    extract_bees_from_predictions(
        CHEMIN_IMAGE, CHEMIN_JSON, DOSSIER_SORTIE, SEUIL_CONFIANCE
    )
