import os
import random
import shutil
from tqdm import tqdm

# Définition des dossiers (N'oubliez pas de modifier le chemin du dossier source !)
source_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_64"  # <-- À remplacer par votre dossier de départ
dest_dir = "/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images2_roboflow"

# Nombre d'images à copier
num_images_to_copy = 1000

# Création du dossier de destination s'il n'existe pas
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 1. Lister tous les fichiers qui sont des images dans le dossier source
all_files = os.listdir(source_dir)
valid_images = [
    f
    for f in all_files
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
]

# 2. Vérifier qu'il y a bien assez d'images, puis sélectionner au hasard
if len(valid_images) < num_images_to_copy:
    print(f"Attention : Le dossier source ne contient que {len(valid_images)} images.")
    print("Toutes les images vont être copiées.")
    selected_images = valid_images
else:
    # random.sample choisit N éléments uniques au hasard
    selected_images = random.sample(valid_images, num_images_to_copy)

print(f"Début de la copie de {len(selected_images)} images vers {dest_dir}...")

# 3. Copier les images avec une barre de progression
for filename in tqdm(selected_images, desc="Copie en cours"):
    src_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)

    try:
        # shutil.copy2 copie le fichier en conservant ses métadonnées (date de création, etc.)
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        tqdm.write(f"Erreur lors de la copie de l'image {filename} : {e}")

print("Copie terminée avec succès !")
