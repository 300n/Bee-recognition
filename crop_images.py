import cv2
import os
import glob

def crop_hive_bottom(image):
    # On garde de 0 à 2486 (3536 - 1050)
    return image[0:2486, 0:3536]

# --- Chemins ---
# J'ajoute /*.jpg pour cibler les fichiers à l'intérieur
input_dir = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_test/*.png'
output_folder = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Utilisation de os.path.join pour éviter les erreurs de slash
search_path = os.path.join(input_dir)
files = glob.glob(search_path)

print(f"Nombre de fichiers trouvés : {len(files)}")

for img_path in files:
    img = cv2.imread(img_path)
    
    if img is not None:
        img_res = crop_hive_bottom(img)
        filename = os.path.basename(img_path)
        
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img_res)
    else:
        print(f"Erreur de lecture pour : {img_path}")