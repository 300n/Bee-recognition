"""
Pipeline de détection des abeilles par filtres de Gabor

Le honeycomb a une structure hexagonale avec des bords dans 3 directions:
0°, 60°, 120° (et leurs opposés 180°, 240°, 300°)

Approche:
1. Appliquer des filtres de Gabor aux 3 orientations du honeycomb
2. Les zones qui répondent fortement aux 3 orientations = honeycomb régulier
3. Les zones qui ne répondent pas = anomalies (abeilles)
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage


class GaborHoneycombDetection:
    """
    Détection par analyse Gabor du pattern hexagonal.
    """

    def __init__(self):
        self.roi_mask = None
        self.gabor_kernels = []
        self.cell_wavelength = None

    def detect_roi(self, image):
        """Détecter la région honeycomb"""
        h, w = image.shape
        mask = np.ones_like(image, dtype=np.uint8) * 255

        # Exclure le bas (cadre noir)
        mask[int(h * 0.68):, :] = 0

        # Exclure les zones très brillantes (LEDs)
        _, bright = cv2.threshold(image, 235, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        bright_dilated = cv2.dilate(bright, kernel)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright_dilated))

        # Exclure les zones très sombres
        _, dark = cv2.threshold(image, 35, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, dark)

        # Nettoyage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self.roi_mask = mask
        return mask

    def estimate_cell_wavelength(self, image, roi_mask):
        """
        Estimer la longueur d'onde des cellules (distance entre centres)
        par analyse FFT
        """
        # Masquer et préparer l'image
        mean_val = np.mean(image[roi_mask > 0])
        masked = np.where(roi_mask > 0, image, mean_val).astype(np.float32)

        # FFT
        f = np.fft.fft2(masked)
        f_shift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(f_shift))

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Profil radial moyen
        max_radius = min(crow, ccol)
        radial = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        for i in range(rows):
            for j in range(cols):
                r = int(np.sqrt((i - crow)**2 + (j - ccol)**2))
                if r < max_radius:
                    radial[r] += magnitude[i, j]
                    counts[r] += 1

        radial = np.divide(radial, counts, where=counts > 0)

        # Trouver le premier pic significatif (après les basses fréquences)
        # Le pic correspond à la fréquence du pattern hexagonal
        start = 20  # Ignorer les très basses fréquences
        peak_idx = start + np.argmax(radial[start:start+100])

        # Convertir en longueur d'onde (pixels par cycle)
        if peak_idx > 0:
            wavelength = rows / peak_idx
        else:
            wavelength = 15  # Valeur par défaut

        self.cell_wavelength = max(8, min(30, wavelength))
        return self.cell_wavelength, magnitude

    def create_gabor_bank(self):
        """
        Créer une banque de filtres de Gabor aux orientations du honeycomb
        Les hexagones ont des bords à 0°, 60°, 120°
        """
        self.gabor_kernels = []

        # Paramètres du filtre
        wavelength = self.cell_wavelength if self.cell_wavelength else 15
        sigma = wavelength * 0.5
        kernel_size = int(wavelength * 2) | 1  # Assurer impair

        # 3 orientations pour l'hexagone (0°, 60°, 120°)
        # En radians: 0, π/3, 2π/3
        orientations = [0, np.pi/3, 2*np.pi/3]

        for theta in orientations:
            # Créer le filtre de Gabor
            kernel = cv2.getGaborKernel(
                ksize=(kernel_size, kernel_size),
                sigma=sigma,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,  # Aspect ratio
                psi=0  # Phase
            )
            # Normaliser
            kernel = kernel - kernel.mean()
            self.gabor_kernels.append((kernel, theta))

        return self.gabor_kernels

    def apply_gabor_filters(self, image, roi_mask):
        """
        Appliquer les filtres de Gabor et combiner les réponses
        """
        # Prétraitement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Appliquer chaque filtre
        responses = []
        for kernel, theta in self.gabor_kernels:
            response = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
            response = np.abs(response)  # Magnitude de la réponse
            responses.append(response)

        # Combiner les réponses - les zones honeycomb répondent aux 3 orientations
        # Prendre le minimum des 3 réponses: élevé seulement si les 3 sont élevées
        combined_min = np.minimum.reduce(responses)

        # Aussi calculer le maximum pour voir les bords individuels
        combined_max = np.maximum.reduce(responses)

        # Et la moyenne
        combined_mean = np.mean(responses, axis=0)

        # Normaliser
        combined_min = cv2.normalize(combined_min, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        combined_max = cv2.normalize(combined_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        combined_mean = cv2.normalize(combined_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Appliquer ROI
        combined_min = cv2.bitwise_and(combined_min, combined_min, mask=roi_mask)
        combined_max = cv2.bitwise_and(combined_max, combined_max, mask=roi_mask)
        combined_mean = cv2.bitwise_and(combined_mean, combined_mean, mask=roi_mask)

        return responses, combined_min, combined_max, combined_mean, enhanced

    def detect_anomalies(self, image, gabor_responses, roi_mask):
        """
        Détecter les anomalies (zones qui ne correspondent pas au pattern)

        Les abeilles:
        - Ont une texture différente du honeycomb
        - Ne répondent pas uniformément aux 3 orientations
        - Créent des "ombres" ou des zones plus sombres
        """
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)

        # Méthode 1: Variance des réponses Gabor
        # Le honeycomb a des réponses similaires dans les 3 directions
        # Les abeilles créent des asymétries
        responses_stack = np.stack(gabor_responses, axis=0)
        variance = np.var(responses_stack, axis=0)
        variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Méthode 2: Écart par rapport à la réponse attendue
        # Calculer la moyenne locale de la réponse Gabor
        mean_response = np.mean(responses_stack, axis=0)
        kernel_size = int(self.cell_wavelength * 3) | 1
        local_mean = cv2.blur(mean_response, (kernel_size, kernel_size))
        deviation = np.abs(mean_response - local_mean)
        deviation_norm = cv2.normalize(deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Méthode 3: Zones sombres de l'image originale
        # Les abeilles sont généralement plus sombres que le honeycomb vide
        local_mean_orig = cv2.blur(enhanced.astype(np.float32), (kernel_size, kernel_size))
        dark_deviation = local_mean_orig - enhanced.astype(np.float32)
        dark_deviation = np.clip(dark_deviation, 0, None)
        dark_norm = cv2.normalize(dark_deviation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Combiner les méthodes
        # Pondération: zones sombres ET (variance élevée OU déviation)
        anomaly_score = cv2.addWeighted(dark_norm, 0.5, variance_norm, 0.3, 0)
        anomaly_score = cv2.addWeighted(anomaly_score, 0.8, deviation_norm, 0.2, 0)

        # Appliquer ROI
        anomaly_score = cv2.bitwise_and(anomaly_score, anomaly_score, mask=roi_mask)

        return anomaly_score, variance_norm, dark_norm

    def segment_bees(self, anomaly_score, roi_mask):
        """Segmenter les abeilles à partir du score d'anomalie"""
        # Seuillage adaptatif
        thresh = cv2.adaptiveThreshold(
            anomaly_score, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(self.cell_wavelength * 4) | 1, -2
        )

        # Appliquer ROI
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # Morphologie
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Fermeture pour connecter
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_medium)

        # Ouverture pour nettoyer
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)

        # Remplir les trous
        filled = ndimage.binary_fill_holes(opened).astype(np.uint8) * 255

        return filled

    def detect_contours(self, binary_mask, original, roi_mask):
        """Détecter les contours des abeilles"""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bees = []
        cell_area = self.cell_wavelength ** 2 if self.cell_wavelength else 225

        # Une abeille fait 2-6 cellules
        min_area = cell_area * 1.5
        max_area = cell_area * 50

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                cx, cy = x + w // 2, y + h // 2
                if roi_mask[cy, cx] == 0:
                    continue

                if 0.2 < aspect_ratio < 5.0:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    extent = area / (w * h) if w * h > 0 else 0

                    if solidity > 0.3 and extent > 0.25:
                        mask = np.zeros(original.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_intensity = cv2.mean(original, mask=mask)[0]

                        # Score basé sur la taille et la forme
                        size_score = min(1.0, area / (cell_area * 8))
                        shape_score = solidity * extent
                        confidence = size_score * 0.4 + shape_score * 0.6

                        bees.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'extent': extent,
                            'center': (cx, cy),
                            'mean_intensity': mean_intensity,
                            'confidence': confidence
                        })

        bees.sort(key=lambda b: b['area'], reverse=True)
        return bees

    def visualize(self, original, bees, roi_mask):
        """Visualiser les résultats"""
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

        # 2. Estimer la longueur d'onde
        wavelength, spectrum = self.estimate_cell_wavelength(image, roi_mask)
        results['spectrum'] = (spectrum / spectrum.max() * 255).astype(np.uint8)
        print(f"    Longueur d'onde: {wavelength:.1f} px")

        # 3. Créer les filtres de Gabor
        self.create_gabor_bank()

        # 4. Appliquer les filtres
        responses, combined_min, combined_max, combined_mean, enhanced = \
            self.apply_gabor_filters(image, roi_mask)
        results['enhanced'] = enhanced
        results['gabor_min'] = combined_min
        results['gabor_max'] = combined_max
        results['gabor_mean'] = combined_mean

        # 5. Détecter les anomalies
        anomaly_score, variance, dark = self.detect_anomalies(image, responses, roi_mask)
        results['anomaly'] = anomaly_score
        results['variance'] = variance
        results['dark'] = dark

        # 6. Segmenter
        binary = self.segment_bees(anomaly_score, roi_mask)
        results['binary'] = binary

        # 7. Détecter les abeilles
        bees = self.detect_contours(binary, image, roi_mask)
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
    """Grille de visualisation"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=14)

    items = [
        ('original', 'Original'),
        ('roi_mask', 'ROI'),
        ('enhanced', 'Améliorée'),
        ('gabor_min', 'Gabor Min (hex pattern)'),
        ('gabor_max', 'Gabor Max (tous bords)'),
        ('variance', 'Variance Gabor'),
        ('dark', 'Zones sombres'),
        ('anomaly', 'Score anomalie'),
        ('binary', 'Segmentation'),
        ('output', 'Détections'),
        ('spectrum', 'Spectre FFT'),
        ('gabor_mean', 'Gabor Moyen')
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
    OUTPUT_DIR = "/Users/noledge/Downloads/B E E/output/gabor"

    pipeline = GaborHoneycombDetection()
    data_path = Path(DATA_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(data_path.glob("*.png"))
    print(f"Traitement de {len(image_files)} images avec filtres de Gabor...")

    all_results = []

    for img_file in image_files:
        print(f"\n  {img_file.name}...")

        image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        results = pipeline.run(image, str(output_path / img_file.stem))

        fig = create_grid(results, f"Gabor Detection - {img_file.name}")
        fig.savefig(str(output_path / f"{img_file.stem}_grid.png"), dpi=150)
        plt.close(fig)

        all_results.append({
            'file': img_file.name,
            'count': len(results['bees']),
            'wavelength': pipeline.cell_wavelength
        })

    print("\n" + "=" * 50)
    print("RÉSUMÉ - FILTRES DE GABOR")
    print("=" * 50)
    for r in all_results:
        print(f"{r['file']}: {r['count']} abeilles (λ={r['wavelength']:.1f}px)")

    total = sum(r['count'] for r in all_results)
    print(f"\nTotal: {total} | Moyenne: {total/len(all_results):.1f} par image")


if __name__ == "__main__":
    main()
