import cv2
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm # Bibliothèque pour la barre de progression
import multiprocessing

# --- Fonction Worker (exécutée par chaque cœur) ---
def process_single_image(file_data):
    """
    Fonction autonome qui charge, coupe et sauvegarde une image.
    file_data est un tuple : (chemin_source, dossier_sortie)
    """
    img_path, output_folder = file_data
    
    try:
        # Lecture
        img = cv2.imread(img_path)
        
        if img is None:
            return f"Erreur lecture : {os.path.basename(img_path)}"
        
        # Crop (0 à 2486 en hauteur)
        # Vérification de sécurité si l'image est plus petite que prévu
        if img.shape[0] > 2486:
            img_res = img[0:2486, 0:3536]
        else:
            img_res = img # Pas de crop si trop petite
            
        # Sauvegarde
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        
        # Optimisation : compression PNG (0 à 9). 
        # 1 est rapide mais fichier plus gros, 9 est lent mais fichier petit. 3 est un bon équilibre.
        # Si c'est du JPG, utilisez [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(save_path, img_res, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        return None # Pas d'erreur
        
    except Exception as e:
        return f"Exception sur {os.path.basename(img_path)} : {str(e)}"

# --- Main ---
if __name__ == '__main__':
    # Chemins
    input_pattern = '/Volumes/SanDisk/dynamic_bee_IR/*.png' # Assure-toi de l'extension (*.png ou *.jpg)
    output_folder = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Récupération de la liste des fichiers
    print("Recherche des fichiers...")
    files = glob.glob(input_pattern)
    print(f"--> {len(files)} images trouvées à traiter.")

    # Préparation des arguments pour les workers
    # On crée une liste de tuples (fichier, dossier_sortie) pour chaque image
    tasks = [(f, output_folder) for f in files]

    # Détection du nombre de cœurs CPU
    # On laisse 1 cœur libre pour que le système reste réactif (facultatif)
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Démarrage du traitement sur {num_cores} cœurs en parallèle...")

    # Lancement du pool de processus
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # tqdm affiche la barre de progression
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"))

    # Bilan des erreurs éventuelles
    errors = [res for res in results if res is not None]
    if errors:
        print(f"\nTerminé avec {len(errors)} erreurs :")
        for e in errors[:10]: print(e) # Affiche les 10 premières erreurs
    else:
        print("\nTraitement terminé avec succès sans erreur !")