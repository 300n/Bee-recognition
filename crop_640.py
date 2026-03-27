import os
import random  # Ajout de la bibliothèque random
from PIL import Image

# Définition des dossiers d'entrée et de sortie
input_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_crop"
output_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_64"

# Paramètres
tile_size = 424
max_images = 100  # Définition de votre limite

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Lister tous les fichiers qui sont des images dans le dossier
all_files = os.listdir(input_dir)
valid_images = [
    f
    for f in all_files
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
]

# 2. (Optionnel mais recommandé) Mélanger la liste pour prendre 100 images au hasard
# Enlevez le "#" au début de la ligne suivante si vous voulez un choix aléatoire :
random.shuffle(valid_images)

# 3. Conserver uniquement le nombre d'images souhaité (les 100 premières de la liste)
selected_images = valid_images[:max_images]

print(f"{len(selected_images)} images sélectionnées pour le découpage.")

# 4. Parcourir uniquement les 100 images sélectionnées
for filename in selected_images:
    img_path = os.path.join(input_dir, filename)

    try:
        img = Image.open(img_path)
        width, height = img.size

        base_name, ext = os.path.splitext(filename)
        count = 1

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if x + tile_size <= width and y + tile_size <= height:
                    box = (x, y, x + tile_size, y + tile_size)
                    cropped_img = img.crop(box)

                    new_filename = f"{base_name}-{count}{ext}"
                    save_path = os.path.join(output_dir, new_filename)

                    cropped_img.save(save_path)
                    count += 1

        print(f"L'image {filename} a été découpée en {count - 1} petits carrés.")

    except Exception as e:
        print(f"Erreur lors du traitement de l'image {filename} : {e}")

print("Découpage de l'échantillon terminé !")
