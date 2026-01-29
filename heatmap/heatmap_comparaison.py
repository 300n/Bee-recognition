import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= CONFIGURATION GLOBALE =================
INPUT_IMAGE_PATH = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/images_crop/1297.png'
OUTPUT_DIR = '/Users/valentindaveau/2IA_S8/Mission_R&D/Bee-recognition/Output/heatmap'

# Paramètres de la grille (Grid Search)
# Lignes de la grille
R_VALUES = [40, 50, 60]
# Colonnes de la grille
THRESH_VALUES = [30, 50, 70]
# =========================================================

def generate_and_save_heatmap(img_gray, r, threshold_val, output_folder, base_name):
    """Génère une image heatmap unique et la sauvegarde sur le disque."""
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
    _, heatmap_thresh = cv2.threshold(heatmap_raw, threshold_val, 255, cv2.THRESH_TOZERO)
    heatmap_color = cv2.applyColorMap(heatmap_thresh, cv2.COLORMAP_JET)
    
    # 3. Overlay et Sauvegarde
    img_bgr_net = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    final_overlay = cv2.addWeighted(img_bgr_net, 0.6, heatmap_color, 0.4, 0)
    
    filename = f"heatmap_r={r}_t_={threshold_val}_{base_name}.png"
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, final_overlay)
    return filename

def create_summary_grid(output_folder, base_name, r_vals, t_vals):
    """Relit les images générées et crée une figure matplotlib 3x3."""
    print("\n--- Création de la grille récapitulative ---")
    
    # Création de la figure et des sous-plots (3 lignes, 3 colonnes)
    # figsize gère la taille totale de l'image en pouces
    fig, axes = plt.subplots(nrows=len(r_vals), ncols=len(t_vals), figsize=(18, 18))
    
    # Titre global de la figure
    fig.suptitle(f"Grid Search : Rayon (r) vs Seuil (t)\nImage : {base_name}.png", fontsize=22, fontweight='bold', y=0.95)

    # On parcourt la grille : i = index ligne (r), j = index colonne (t)
    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            # Reconstruction du nom de fichier cible
            filename = f"heatmap_r={r}_t_={t}_{base_name}.png"
            filepath = os.path.join(output_folder, filename)
            
            # Lecture de l'image
            if os.path.exists(filepath):
                # OpenCV lit en BGR, Matplotlib affiche en RGB. Conversion nécessaire.
                img_bgr = cv2.imread(filepath)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Sélection du sous-plot courant
                ax = axes[i, j]
                ax.imshow(img_rgb)
                
                # Titre du sous-plot avec les paramètres
                ax.set_title(f"r = {r} | Seuil = {t}", fontsize=14, pad=10)
                
                # Suppression des axes (ticks) pour la propreté
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Ajout d'étiquettes pour les lignes et colonnes sur les bords
                if j == 0: ax.set_ylabel(f"Rayon r={r}", fontsize=16, fontweight='bold', labelpad=10)
                if i == 0: ax.set_xlabel(f"Seuil t={t}", fontsize=16, fontweight='bold', labelpad=10)
                if i == 0: ax.xaxis.set_label_position('top') 

            else:
                print(f"Attention : Image manquante pour la grille : {filename}")

    # Ajustement automatique des espacements
    plt.tight_layout()
    # Petit ajustement pour laisser de la place au titre principal
    plt.subplots_adjust(top=0.90)
    
    # Sauvegarde de l'image finale
    summary_path = os.path.join(output_folder, f"SUMMARY_GRID_{base_name}.png")
    # dpi=150 assure une bonne résolution pour zoomer
    plt.savefig(summary_path, dpi=150) 
    print(f" Image récapitulative sauvegardée : {summary_path}")
    # plt.show() # Décommentez si vous voulez voir la fenêtre s'ouvrir aussi

# ================= MAIN =================
if __name__ == "__main__":
    # 1. Préparation
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier créé : {OUTPUT_DIR}")

    img_net = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if img_net is None:
        print(f"ERREUR CRITIQUE : Impossible de charger l'image : {INPUT_IMAGE_PATH}")
    else:
        base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
        
        print("--- Étape 1 : Génération des 9 images individuelles ---")
        # 2. Boucles pour générer les 9 images
        for r in R_VALUES:
            for t in THRESH_VALUES:
                fname = generate_and_save_heatmap(img_net, r, t, OUTPUT_DIR, base_name)
                print(f"Généré : {fname}")
                
        # 3. Appel de la fonction pour créer la grille finale
        create_summary_grid(OUTPUT_DIR, base_name, R_VALUES, THRESH_VALUES)