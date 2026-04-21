"""
Pipeline de détection des abeilles par reconstruction morphologique

Principe:
- La reconstruction morphologique permet d'extraire les structures
  qui correspondent à un certain pattern (ici le honeycomb régulier)
- En soustrayant cette reconstruction de l'original, on obtient
  les éléments qui ne font pas partie du pattern = les abeilles

Techniques:
1. Opening by reconstruction: élimine les objets qui ne s'ouvrent pas
2. Closing by reconstruction: remplit les trous qui ne se ferment pas
3. H-maxima/H-minima transform: supprime les maxima/minima locaux
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import reconstruction


class MorphologicalBeeDetection:
    """
    Détection par reconstruction morphologique.
    """

    def __init__(self):
        self.roi_mask = None
        self.cell_size = 12  # Taille approximative des cellules

    def detect_roi(self, image):
        """Détecter la région honeycomb - exclure cadre et éléments non-honeycomb"""
        h, w = image.shape
        mask = np.ones_like(image, dtype=np.uint8) * 255

        # Exclure le bas (cadre noir et barre métallique)
        mask[int(h * 0.62):, :] = 0

        # Exclure les bords latéraux (cadre)
        mask[:, :15] = 0
        mask[:, w-15:] = 0

        # Exclure le haut
        mask[:15, :] = 0

        # Exclure LEDs (points très brillants)
        _, bright = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        bright_dilated = cv2.dilate(bright, kernel)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright_dilated))

        # Exclure zones très sombres (ombres, trous)
        _, dark = cv2.threshold(image, 35, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, dark)

        # Exclure les zones uniformément grises (cadre métallique)
        # Le honeycomb a de la texture, le métal est uniforme
        # Utiliser variance locale (plus rapide que generic_filter)
        blur_size = 15
        mean = cv2.blur(image.astype(np.float32), (blur_size, blur_size))
        sq_mean = cv2.blur(image.astype(np.float32) ** 2, (blur_size, blur_size))
        local_var = np.sqrt(np.clip(sq_mean - mean ** 2, 0, None))
        _, texture_mask = cv2.threshold(local_var.astype(np.uint8), 8, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, texture_mask)

        # Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self.roi_mask = mask
        return mask

    def opening_by_reconstruction(self, image, kernel_size):
        """
        Opening by reconstruction:
        - Érode l'image
        - Reconstruit par dilatation sous contrainte de l'original
        Cela préserve les structures qui survivent à l'érosion
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Érosion
        eroded = cv2.erode(image, kernel)

        # Reconstruction par dilatation
        # La reconstruction préserve les formes de l'original
        marker = eroded.astype(np.float64)
        mask = image.astype(np.float64)

        reconstructed = reconstruction(marker, mask, method='dilation')

        return reconstructed.astype(np.uint8)

    def closing_by_reconstruction(self, image, kernel_size):
        """
        Closing by reconstruction:
        - Dilate l'image
        - Reconstruit par érosion sous contrainte de l'original
        Cela remplit les trous tout en préservant la forme
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Dilatation
        dilated = cv2.dilate(image, kernel)

        # Inversion pour reconstruire par érosion
        # (reconstruction ne fait que dilatation, donc on inverse)
        marker = (255 - dilated).astype(np.float64)
        mask = (255 - image).astype(np.float64)

        reconstructed = reconstruction(marker, mask, method='dilation')

        return (255 - reconstructed).astype(np.uint8)

    def regional_minima(self, image, h=10):
        """
        H-minima transform: supprime les minima locaux de profondeur < h
        Le honeycomb a des minima réguliers (les cellules)
        Les abeilles créent des minima irréguliers
        """
        # Ajouter h à l'image
        marker = np.clip(image.astype(np.int32) + h, 0, 255).astype(np.uint8)

        # Reconstruction
        reconstructed = self.opening_by_reconstruction(marker, 3)

        # Les minima sont là où reconstructed == image + h
        minima = reconstructed - image

        return minima

    def extract_honeycomb_pattern(self, image, roi_mask):
        """
        Extraire le pattern honeycomb régulier.
        Utilise une série de reconstructions morphologiques.
        """
        # Prétraitement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Appliquer le masque
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)

        # 1. Opening by reconstruction avec un élément de la taille d'une cellule
        # Cela préserve les structures régulières (cellules)
        opened = self.opening_by_reconstruction(enhanced, self.cell_size)

        # 2. Closing by reconstruction
        # Remplit les petits trous tout en préservant les grandes structures
        closed = self.closing_by_reconstruction(opened, self.cell_size // 2)

        # Le pattern honeycomb est préservé
        honeycomb_pattern = closed

        return honeycomb_pattern, enhanced

    def detect_anomalies(self, enhanced, honeycomb_pattern, roi_mask):
        """
        Détecter les anomalies en comparant l'image originale
        avec le pattern honeycomb extrait.
        """
        # Différence: là où l'original diffère du pattern
        diff = cv2.absdiff(enhanced, honeycomb_pattern)

        # Amplifier la différence
        diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Appliquer le masque
        diff_enhanced = cv2.bitwise_and(diff_enhanced, diff_enhanced, mask=roi_mask)

        # Les abeilles sont généralement plus sombres que le pattern
        # Zones où l'original est plus sombre que le pattern reconstruit
        darker = honeycomb_pattern.astype(np.int32) - enhanced.astype(np.int32)
        darker = np.clip(darker, 0, 255).astype(np.uint8)
        darker = cv2.bitwise_and(darker, darker, mask=roi_mask)

        # Combiner: différence + zones sombres
        combined = cv2.addWeighted(diff_enhanced, 0.5, darker, 0.5, 0)

        return combined, diff_enhanced, darker

    def regional_texture_analysis(self, image, roi_mask):
        """
        Analyse de texture locale.
        Les abeilles ont une texture différente du honeycomb.
        """
        # Variance locale
        kernel_size = self.cell_size * 2 + 1
        mean = cv2.blur(image.astype(np.float32), (kernel_size, kernel_size))
        sq_mean = cv2.blur(image.astype(np.float32) ** 2, (kernel_size, kernel_size))
        variance = sq_mean - mean ** 2
        variance = np.clip(variance, 0, None)

        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        variance_norm = cv2.bitwise_and(variance_norm, variance_norm, mask=roi_mask)

        return variance_norm

    def segment_bees(self, anomaly_score, roi_mask):
        """Segmenter les abeilles"""
        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(
            anomaly_score, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.cell_size * 4 + 1, -3
        )

        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # Morphologie
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_medium)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
        filled = ndimage.binary_fill_holes(opened).astype(np.uint8) * 255

        return filled

    def detect_bees(self, binary_mask, original, roi_mask):
        """Détecter les contours"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bees = []
        cell_area = self.cell_size ** 2

        min_area = cell_area * 1.5
        max_area = cell_area * 50

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                cx, cy = x + w // 2, y + h // 2
                if cy >= roi_mask.shape[0] or cx >= roi_mask.shape[1]:
                    continue
                if roi_mask[cy, cx] == 0:
                    continue

                if 0.2 < aspect_ratio < 5.0:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    extent = area / (w * h) if w * h > 0 else 0

                    if solidity > 0.25 and extent > 0.2:
                        confidence = min(1.0, area / (cell_area * 6)) * solidity

                        bees.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (cx, cy),
                            'confidence': confidence
                        })

        bees.sort(key=lambda b: b['area'], reverse=True)
        return bees

    def visualize(self, original, bees, roi_mask):
        """Visualiser"""
        output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        output[roi_mask == 0] = output[roi_mask == 0] // 2

        for i, bee in enumerate(bees):
            conf = bee['confidence']
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255) if conf > 0.3 else (0, 165, 255)

            cv2.drawContours(output, [bee['contour']], -1, color, 2)
            cx, cy = bee['center']
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(output, str(i + 1), (cx + 5, cy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return output

    def run(self, image, save_dir=None):
        """Pipeline complet"""
        results = {'original': image}

        # 1. ROI
        roi_mask = self.detect_roi(image)
        results['roi_mask'] = roi_mask

        # 2. Extraire le pattern honeycomb
        honeycomb, enhanced = self.extract_honeycomb_pattern(image, roi_mask)
        results['enhanced'] = enhanced
        results['honeycomb'] = honeycomb

        # 3. Détecter les anomalies
        anomaly, diff, darker = self.detect_anomalies(enhanced, honeycomb, roi_mask)
        results['anomaly'] = anomaly
        results['diff'] = diff
        results['darker'] = darker

        # 4. Analyse de texture
        texture = self.regional_texture_analysis(enhanced, roi_mask)
        results['texture'] = texture

        # 5. Combiner anomalie et texture
        final_score = cv2.addWeighted(anomaly, 0.6, texture, 0.4, 0)
        results['final_score'] = final_score

        # 6. Segmenter
        binary = self.segment_bees(final_score, roi_mask)
        results['binary'] = binary

        # 7. Détecter
        bees = self.detect_bees(binary, image, roi_mask)
        results['bees'] = bees
        print(f"    Abeilles: {len(bees)}")

        # 8. Visualiser
        output = self.visualize(image, bees, roi_mask)
        results['output'] = output

        # Sauvegarder
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for name, img in results.items():
                if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                    cv2.imwrite(str(Path(save_dir) / f"{name}.png"), img)

        return results


def create_grid(results, title):
    """Grille"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=14)

    items = [
        ('original', 'Original'),
        ('roi_mask', 'ROI'),
        ('enhanced', 'Améliorée'),
        ('honeycomb', 'Pattern Honeycomb (reconstruit)'),
        ('diff', 'Différence'),
        ('darker', 'Zones plus sombres'),
        ('texture', 'Texture locale'),
        ('anomaly', 'Score anomalie'),
        ('final_score', 'Score final'),
        ('binary', 'Segmentation'),
        ('output', 'Détections'),
        ('original', 'Original (ref)')
    ]

    for ax, (key, label) in zip(axes.flat, items):
        if key in results and isinstance(results[key], np.ndarray):
            img = results[key]
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    DATA_DIR = "/Users/noledge/Downloads/B E E/data"
    OUTPUT_DIR = "/Users/noledge/Downloads/B E E/output/morpho"

    pipeline = MorphologicalBeeDetection()
    data_path = Path(DATA_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(data_path.glob("*.png"))
    print(f"Traitement de {len(image_files)} images par reconstruction morphologique...")

    all_results = []

    for img_file in image_files:
        print(f"\n  {img_file.name}...")

        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        results = pipeline.run(image, str(output_path / img_file.stem))

        fig = create_grid(results, f"Morpho Detection - {img_file.name}")
        fig.savefig(str(output_path / f"{img_file.stem}_grid.png"), dpi=150)
        plt.close(fig)

        all_results.append({
            'file': img_file.name,
            'count': len(results['bees'])
        })

    print("\n" + "=" * 50)
    print("RÉSUMÉ - RECONSTRUCTION MORPHOLOGIQUE")
    print("=" * 50)
    for r in all_results:
        print(f"{r['file']}: {r['count']} abeilles")

    total = sum(r['count'] for r in all_results)
    print(f"\nTotal: {total} | Moyenne: {total/len(all_results):.1f} par image")


if __name__ == "__main__":
    main()
