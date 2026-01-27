"""
Pipeline AMÉLIORÉ de détection et mise en évidence des abeilles
Version optimisée pour distinguer les abeilles du fond (rayons de cire)

Améliorations par rapport à la version initiale:
1. Détection basée sur l'intensité (abeilles = zones sombres)
2. Filtrage par taille adapté aux dimensions réelles des abeilles
3. Analyse de texture pour éliminer la structure hexagonale
4. Détection de mouvement entre frames consécutifs
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage


class ImprovedBeeDetectionPipeline:
    """
    Pipeline amélioré pour la détection des abeilles.
    Optimisé pour les images infrarouges de ruches.
    """

    def __init__(self, image_width=None, image_height=None):
        # Paramètres CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Estimation de la taille d'une abeille en pixels
        # Une abeille fait environ 12-15mm, une cellule 5mm
        # Ratio abeille/cellule ≈ 2.5-3
        self.min_bee_area = 500  # Pixels carrés minimum
        self.max_bee_area = 15000  # Pixels carrés maximum

    def step1_intensity_based_segmentation(self, image):
        """
        Étape 1: Segmentation basée sur l'intensité
        Les abeilles apparaissent comme des zones SOMBRES sur le fond clair des rayons

        Technique: Seuillage d'Otsu inversé pour trouver les zones sombres
        """
        # Égalisation d'histogramme adaptative
        enhanced = self.clahe.apply(image)

        # Flou pour réduire le bruit de la texture hexagonale
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

        # Seuillage d'Otsu pour trouver le seuil optimal
        # On cherche les zones SOMBRES (abeilles) donc on inverse
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Alternative: seuil fixe basé sur l'histogramme
        # Les abeilles sont généralement dans les 30% les plus sombres
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
        cumsum = np.cumsum(hist.flatten())
        total_pixels = cumsum[-1]
        threshold_percentile = 0.35  # 35% les plus sombres
        threshold_value = np.searchsorted(cumsum, total_pixels * threshold_percentile)

        _, dark_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

        return enhanced, otsu_mask, dark_mask, blurred

    def step2_texture_analysis(self, image):
        """
        Étape 2: Analyse de texture pour éliminer la structure hexagonale
        Utilise la variance locale pour détecter les régions non-uniformes

        Les rayons ont une texture régulière (hexagonale)
        Les abeilles ont une texture plus variable
        """
        # Calcul de la variance locale
        kernel_size = 15
        mean = cv2.blur(image.astype(np.float64), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((image.astype(np.float64)) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        variance = np.sqrt(np.maximum(variance, 0))

        # Normaliser
        variance_norm = ((variance - variance.min()) / (variance.max() - variance.min() + 1e-6) * 255).astype(np.uint8)

        # Les zones à haute variance sont potentiellement des abeilles
        _, high_variance_mask = cv2.threshold(variance_norm, 50, 255, cv2.THRESH_BINARY)

        return variance_norm, high_variance_mask

    def step3_gradient_analysis(self, image):
        """
        Étape 3: Analyse du gradient pour détecter les bords des abeilles
        Les abeilles ont des contours plus marqués que les cellules
        """
        # Calcul du gradient avec Sobel
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalisation
        gradient_norm = ((gradient_magnitude / gradient_magnitude.max()) * 255).astype(np.uint8)

        # Seuillage pour garder les forts gradients
        _, strong_edges = cv2.threshold(gradient_norm, 30, 255, cv2.THRESH_BINARY)

        return gradient_norm, strong_edges

    def step4_combine_masks(self, dark_mask, high_variance_mask, strong_edges):
        """
        Étape 4: Combinaison intelligente des masques
        Une abeille doit être: sombre ET (haute variance OU forts bords)
        """
        # Dilatation des masques secondaires pour être plus inclusif
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        variance_dilated = cv2.dilate(high_variance_mask, kernel, iterations=2)
        edges_dilated = cv2.dilate(strong_edges, kernel, iterations=1)

        # Combinaison: zones sombres qui ont aussi une texture variable ou des bords forts
        texture_or_edges = cv2.bitwise_or(variance_dilated, edges_dilated)
        combined = cv2.bitwise_and(dark_mask, texture_or_edges)

        return combined

    def step5_morphological_refinement(self, mask):
        """
        Étape 5: Raffinement morphologique
        Nettoyage du masque pour éliminer le bruit et connecter les régions
        """
        # Élément structurant elliptique (forme d'abeille)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # Ouverture agressive pour éliminer le petit bruit
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium)

        # Fermeture pour connecter les parties d'une même abeille
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

        # Remplissage des trous
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255

        return filled

    def step6_contour_filtering(self, mask, original_image):
        """
        Étape 6: Filtrage des contours par critères géométriques
        Garde uniquement les régions qui ressemblent à des abeilles
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bees = []
        filtered_mask = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtrage par taille
            if not (self.min_bee_area < area < self.max_bee_area):
                continue

            # Calcul des caractéristiques géométriques
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Les abeilles ont un ratio d'aspect entre 0.4 et 2.5
            if not (0.4 < aspect_ratio < 2.5):
                continue

            # Calcul de la solidité (aire / aire convexe)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Les abeilles ont une solidité > 0.5
            if solidity < 0.5:
                continue

            # Calcul de l'étendue (aire / aire bbox)
            extent = area / (w * h) if (w * h) > 0 else 0

            # Les abeilles ont une étendue > 0.4
            if extent < 0.4:
                continue

            # Calcul de l'intensité moyenne dans la région
            mask_roi = np.zeros_like(original_image)
            cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            mean_intensity = cv2.mean(original_image, mask=mask_roi)[0]

            # Cette région passe tous les filtres - c'est probablement une abeille
            bees.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'mean_intensity': mean_intensity
            })

            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        return bees, filtered_mask

    def step7_visualize_results(self, original, bees, show_stats=True):
        """
        Étape 7: Visualisation des résultats
        """
        # Créer image couleur pour visualisation
        output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        # Couleurs pour différentes confiances
        color_high = (0, 255, 0)  # Vert - haute confiance
        color_med = (0, 255, 255)  # Jaune - moyenne confiance
        color_low = (0, 165, 255)  # Orange - basse confiance

        for i, bee in enumerate(bees):
            # Couleur basée sur la solidité (mesure de confiance)
            if bee['solidity'] > 0.8:
                color = color_high
            elif bee['solidity'] > 0.65:
                color = color_med
            else:
                color = color_low

            # Dessiner le contour
            cv2.drawContours(output, [bee['contour']], -1, color, 2)

            # Dessiner le centre
            cx, cy = bee['center']
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            # Numéro
            cv2.putText(output, str(i + 1), (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Stats si demandé
            if show_stats:
                x, y, w, h = bee['bbox']
                info = f"A:{bee['area']:.0f}"
                cv2.putText(output, info, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return output

    def step8_create_heatmap(self, shape, bees):
        """
        Étape 8: Création d'une carte de chaleur de densité
        """
        heatmap = np.zeros(shape, dtype=np.float32)

        for bee in bees:
            cx, cy = bee['center']
            sigma = np.sqrt(bee['area']) / 3

            # Gaussienne locale
            y_min = max(0, int(cy - 3 * sigma))
            y_max = min(shape[0], int(cy + 3 * sigma))
            x_min = max(0, int(cx - 3 * sigma))
            x_max = min(shape[1], int(cx + 3 * sigma))

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    heatmap[y, x] += np.exp(-dist ** 2 / (2 * sigma ** 2))

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    def run_pipeline(self, image, verbose=False):
        """
        Exécution complète du pipeline amélioré
        """
        results = {'original': image}

        # Étape 1: Segmentation par intensité
        enhanced, otsu_mask, dark_mask, blurred = self.step1_intensity_based_segmentation(image)
        results['enhanced'] = enhanced
        results['otsu_mask'] = otsu_mask
        results['dark_mask'] = dark_mask
        results['blurred'] = blurred

        # Étape 2: Analyse de texture
        variance, high_variance_mask = self.step2_texture_analysis(blurred)
        results['variance'] = variance
        results['high_variance_mask'] = high_variance_mask

        # Étape 3: Analyse du gradient
        gradient, strong_edges = self.step3_gradient_analysis(blurred)
        results['gradient'] = gradient
        results['strong_edges'] = strong_edges

        # Étape 4: Combinaison des masques
        combined = self.step4_combine_masks(dark_mask, high_variance_mask, strong_edges)
        results['combined_mask'] = combined

        # Étape 5: Raffinement morphologique
        refined = self.step5_morphological_refinement(combined)
        results['refined_mask'] = refined

        # Étape 6: Filtrage des contours
        bees, filtered_mask = self.step6_contour_filtering(refined, enhanced)
        results['bees'] = bees
        results['filtered_mask'] = filtered_mask
        results['n_bees'] = len(bees)

        # Étape 7: Visualisation
        visualization = self.step7_visualize_results(image, bees)
        results['visualization'] = visualization

        # Étape 8: Heatmap
        heatmap = self.step8_create_heatmap(image.shape, bees)
        results['heatmap'] = heatmap

        # Overlay final
        original_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_color, 0.6, heatmap, 0.4, 0)
        results['overlay'] = overlay

        if verbose:
            print(f"  Détections: {len(bees)} abeilles")
            if bees:
                areas = [b['area'] for b in bees]
                print(f"  Aire moyenne: {np.mean(areas):.0f} px², min: {np.min(areas):.0f}, max: {np.max(areas):.0f}")

        return results


def create_visualization_improved(results, title="Pipeline amélioré"):
    """
    Crée une visualisation en grille des résultats
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"{title} - {results['n_bees']} abeilles détectées", fontsize=16)

    images_to_show = [
        ('original', 'Image originale'),
        ('enhanced', 'CLAHE amélioré'),
        ('dark_mask', 'Masque zones sombres'),
        ('variance', 'Variance locale'),
        ('high_variance_mask', 'Haute variance'),
        ('gradient', 'Gradient'),
        ('strong_edges', 'Contours forts'),
        ('combined_mask', 'Masque combiné'),
        ('refined_mask', 'Masque raffiné'),
        ('filtered_mask', 'Masque filtré'),
        ('visualization', 'Détections'),
        ('overlay', 'Superposition finale')
    ]

    for ax, (key, label) in zip(axes.flat, images_to_show):
        if key in results and isinstance(results[key], np.ndarray):
            img = results[key]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(label, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig


def process_images_improved(data_dir, output_dir):
    """
    Traite toutes les images avec le pipeline amélioré
    """
    pipeline = ImprovedBeeDetectionPipeline()
    data_path = Path(data_dir)
    output_path = Path(output_dir) / "improved"
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(data_path.glob("*.png"))
    print(f"Traitement de {len(image_files)} images avec le pipeline amélioré...")

    all_results = []

    for img_file in image_files:
        print(f"\n  - {img_file.name}...")

        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        results = pipeline.run_pipeline(image, verbose=True)

        # Sauvegarder la visualisation
        fig = create_visualization_improved(results, f"Détection - {img_file.name}")
        fig.savefig(str(output_path / f"{img_file.stem}_improved.png"), dpi=150)
        plt.close(fig)

        # Sauvegarder l'image avec détections
        cv2.imwrite(str(output_path / f"{img_file.stem}_detection.png"), results['visualization'])

        all_results.append({
            'filename': img_file.name,
            'n_bees': results['n_bees'],
            'bees': results['bees']
        })

    return all_results


def detect_motion_between_frames(frame1, frame2, min_area=500):
    """
    Détection de mouvement entre deux frames consécutifs
    Utile pour identifier les abeilles en mouvement
    """
    # Différence absolue
    diff = cv2.absdiff(frame1, frame2)

    # Seuillage
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    # Trouver les régions de mouvement
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            moving_regions.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area
            })

    return motion_mask, moving_regions


def analyze_motion_sequence(data_dir, output_dir):
    """
    Analyse le mouvement sur une séquence d'images
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir) / "motion"
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(data_path.glob("*.png"))

    if len(image_files) < 2:
        print("Pas assez d'images pour l'analyse de mouvement")
        return

    print(f"\nAnalyse de mouvement sur {len(image_files)} images...")

    # Charger toutes les images
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    # Analyser les différences consécutives
    motion_summary = np.zeros_like(images[0], dtype=np.float32)

    for i in range(len(images) - 1):
        motion_mask, regions = detect_motion_between_frames(images[i], images[i + 1])
        motion_summary += motion_mask.astype(np.float32)
        print(f"  Frame {i}->{i + 1}: {len(regions)} régions en mouvement")

    # Normaliser et visualiser
    motion_summary = (motion_summary / motion_summary.max() * 255).astype(np.uint8)
    motion_heatmap = cv2.applyColorMap(motion_summary, cv2.COLORMAP_HOT)

    # Sauvegarder
    cv2.imwrite(str(output_path / "motion_summary.png"), motion_heatmap)

    # Créer une visualisation combinée
    original_color = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(original_color, 0.5, motion_heatmap, 0.5, 0)
    cv2.imwrite(str(output_path / "motion_overlay.png"), combined)

    print(f"\nRésultats sauvegardés dans {output_path}")

    return motion_summary


if __name__ == "__main__":
    DATA_DIR = "/Users/noledge/Downloads/B E E/data"
    OUTPUT_DIR = "/Users/noledge/Downloads/B E E/output"

    # Pipeline amélioré de détection
    results = process_images_improved(DATA_DIR, OUTPUT_DIR)

    # Analyse de mouvement
    analyze_motion_sequence(DATA_DIR, OUTPUT_DIR)

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    for r in results:
        print(f"{r['filename']}: {r['n_bees']} abeilles détectées")

    total = sum(r['n_bees'] for r in results)
    avg = total / len(results) if results else 0
    print(f"\nTotal: {total} détections sur {len(results)} images")
    print(f"Moyenne: {avg:.1f} abeilles par image")
