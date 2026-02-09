import cv2
import os
import multiprocessing
import time

# --- FONCTION EXÉCUTÉE PAR LES WORKERS (CŒURS) ---
def worker_save_frame(queue, output_folder, file_prefix):
    """
    Récupère des frames depuis la queue et les sauvegarde sur le disque.
    S'arrête quand il reçoit 'None'.
    """
    while True:
        # Récupération de la donnée
        item = queue.get()
        
        # Signal d'arrêt
        if item is None:
            break
            
        frame_idx, frame = item
        
        # Construction du nom : M01C01_000000.png
        filename = f"{file_prefix}_{frame_idx:06d}.png"
        path = os.path.join(output_folder, filename)
        
        # Sauvegarde (Compression PNG niveau 1 pour la vitesse)
        cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

# --- PROCESSUS PRINCIPAL ---
def extract_frames_multicore(video_path, output_folder, file_prefix):
    
    # 1. Vérifications et Création dossier
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lecture vidéo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extraction de {total_frames} frames sur {multiprocessing.cpu_count()} cœurs...")

    # 2. Configuration du Multiprocessing
    # On limite la queue à 128 images pour ne pas saturer la RAM si la lecture est plus rapide que l'écriture
    manager = multiprocessing.Manager()
    queue = manager.Queue(maxsize=128)
    
    # On utilise tous les cœurs sauf 1 (pour laisser le système respirer)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(processes=num_workers)

    # Lancement des workers qui écoutent la queue
    # On lance 'num_workers' fois la fonction worker_save_frame en tâche de fond
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker_save_frame, args=(queue, output_folder, file_prefix))
        p.start()
        workers.append(p)

    # 3. Boucle de lecture (Producteur)
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # On met l'image dans la file d'attente.
        # Si la queue est pleine (128), le script attend ici que les workers libèrent de la place.
        queue.put((frame_count, frame))
        
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"  -> Lu et envoyé : {frame_count}/{total_frames}")

    cap.release()

    # 4. Arrêt des workers
    # On envoie un signal 'None' pour chaque worker pour leur dire de s'arrêter
    for _ in range(num_workers):
        queue.put(None)
    
    # On attend que tout le monde ait fini d'écrire
    for p in workers:
        p.join()

    duration = (time.time() - start_time)/60
    print(f"\nTerminé ! {frame_count} images extraites en {duration:.2f} minutes.")

# ================= CONFIGURATION =================
VIDEO_PATH = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Final_videos/M01C02_final.mp4'
DOSSIER_SORTIE = '/Volumes/LaCie/BeeVid/Final_videos/Mc02'
PREFIXE = 'M01C02'

if __name__ == '__main__':
    # Protection obligatoire pour le multiprocessing sous Windows/macOS
    extract_frames_multicore(VIDEO_PATH, DOSSIER_SORTIE, PREFIXE)