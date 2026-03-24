# Contexte : Pipeline de détection et suppression d'alvéoles sur images de cadres de ruche

## Objectif

Détecter automatiquement les alvéoles (cellules hexagonales) sur des photos de cadres de ruche prises par une caméra embarquée, puis les "noircir" (blackout) pour ne conserver que les éléments non-alvéolaires (abeilles, zones vides, etc.). L'objectif final est de faciliter l'analyse de l'activité sur le cadre en supprimant le motif régulier des alvéoles.

## Problèmes à résoudre

Les images présentent plusieurs difficultés :

1. **Gradient de luminosité** : la caméra filme en angle rasant, créant un gradient gauche→droite (la gauche est ~2x plus sombre que la droite).
2. **Zones d'abeilles** : les abeilles recouvrent certaines alvéoles et ne doivent PAS être supprimées.
3. **Inversion de polarité du contraste** : dans les zones bien éclairées (droite), les alvéoles sont claires avec des murs sombres. Dans la zone extrême gauche (angle rasant), c'est l'inverse — les murs deviennent plus clairs que l'intérieur des cellules. Un seuillage classique ne fonctionne donc pas partout.
4. **Variation de taille** : les alvéoles apparaissent légèrement plus petites en perspective vers la gauche.

## Pipeline développé (12 étapes)

### Phase 1 : Correction de luminosité (étapes 1-4)

**Principe** : correction itérative adaptative qui booste les zones sombres tout en préservant les zones d'abeilles.

```python
# Pour chaque itération :
# 1. Calculer la moyenne locale (GaussianBlur 101x101)
# 2. Calculer le coefficient de variation local (CV = std/mean)
# 3. CV bas (<0.30) + zone sombre → ombre de perspective → à corriger
# 4. CV haut (>0.30) → texture d'abeilles → à préserver
# 5. Boost proportionnel à l'écart avec le P75 de luminosité
# 6. Lisser le boost (GaussianBlur 101x101)
```

- **3 premières itérations** : corrigent principalement la zone centre (boost max ~3.8x)
- **17 itérations suivantes** : la correction se propage vers l'extrême gauche (boost max ~5.7x)
- **Total : 20 itérations** suffisent (convergence confirmée)

La fonction clé :

```python
def adaptive_correction(img):
    img_f = img.astype(np.float64)
    local_mean = cv2.GaussianBlur(img_f, (101, 101), 0)
    local_std = np.sqrt(np.clip(
        cv2.GaussianBlur(img_f**2, (51, 51), 0) -
        cv2.GaussianBlur(img_f, (51, 51), 0)**2, 0, None))
    local_cv = np.where(local_mean > 5, local_std / local_mean, 0)
    target = np.percentile(local_mean, 75)
    dark_thresh = np.percentile(local_mean, 50)
    needs = (local_mean < dark_thresh) & (local_cv <= 0.30)
    ratio = np.clip(1.0 - (local_mean / target), 0, 1)
    boost = np.where(needs, 1.0 + ratio * 1.5, 1.0)
    boost = cv2.GaussianBlur(boost, (101, 101), 0)
    return np.clip(img_f * boost, 0, 255).astype(np.uint8), boost
```

### Phase 2 : Détection des alvéoles (étapes 5-9)

Deux pipelines complémentaires tournent en parallèle :

#### Pipeline A — Détection standard (seuillage adaptatif)

```python
def detect_cells(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blurred)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 2)
    cells = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(cells, cv2.MORPH_OPEN, kernel, iterations=1)
```

Fonctionne bien sur ~70% de l'image (zones à polarité normale).

#### Pipeline B — Normalisation locale z-score (agnostique à la polarité)

```python
def detect_localnorm(img, k=15):
    img_f = img.astype(np.float64)
    local_mean = cv2.GaussianBlur(img_f, (k, k), 0)
    local_std = np.sqrt(np.clip(
        cv2.GaussianBlur(img_f**2, (k, k), 0) - local_mean**2, 0, None))
    local_std[local_std < 1] = 1
    normalized = np.clip((img_f - local_mean) / local_std * 50 + 128, 0, 255).astype(np.uint8)
    return normalized, detect_cells(normalized)
```

**C'est la découverte clé du projet.** Au lieu de regarder si un pixel est clair ou sombre (dépendant de la polarité), on regarde s'il **s'écarte de son voisinage immédiat**. Que les murs soient plus clairs ou plus sombres que l'intérieur, le z-score les distingue. k=15 et k=21 sont les valeurs optimales.

#### Extraction et filtrage des contours

```python
def extract_contours(binary, min_area=30, max_area=2000, min_circ=0.25):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours
            if min_area < cv2.contourArea(c) < max_area
            and cv2.arcLength(c, True) > 0
            and 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True)**2 > min_circ]
```

#### Calibration du filtre d'aire

Les seuils d'aire sont calibrés automatiquement sur la moitié droite de l'image (bien éclairée, détection fiable) :

```python
right_contours = extract_contours(detect_cells(cropped[:, w//2:]))
right_areas = np.array([cv2.contourArea(c) for c in right_contours])
FLOOR = np.percentile(right_areas, 70)    # ~414 px²
CEILING = np.percentile(right_areas, 97.5)  # ~646 px²
```

#### Fusion itérative

Les deux pipelines tournent sur 21 passes successives de l'image (chaque passe ajoute une correction de luminosité supplémentaire). Les résultats sont fusionnés avec dédoublonnage spatial :

```python
all_found = []; existing = set()

def add_unique(contours):
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            key = (int(M["m10"]/M["m00"])//8, int(M["m01"]/M["m00"])//8)
            if key not in existing:
                existing.add(key)
                all_found.append(c)

for it in range(21):
    # Standard
    add_unique(filter(detect_cells(current)))
    # Z-score k=15 et k=21
    add_unique(filter(detect_localnorm(current, 15)))
    add_unique(filter(detect_localnorm(current, 21)))
    # Correction supplémentaire
    current = adaptive_correction(current)
```

### Phase 3 : Blackout (étapes 10-12)

```python
mask = np.zeros_like(cropped)
cv2.drawContours(mask, all_found, -1, 255, cv2.FILLED)
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
blackout = corrected_img.copy()
blackout[mask > 0] = 0
```

Le blackout est appliqué sur l'**image corrigée en luminosité** (pas l'originale), car les corrections de luminosité sont pertinentes et utiles pour l'analyse ultérieure.

## Résultats chiffrés

| Métrique | Valeur |
|---|---|
| Baseline (détection simple, 1 passe) | 678 cellules |
| Pipeline complet (fusion itérative) | **1104 cellules** |
| Zone gauche (<400px) baseline | 49 cellules |
| Zone gauche (<400px) pipeline | **171 cellules** (x3.5) |
| % image noircie | **36.4%** |
| Filtre d'aire | [414, 646] px² |

## Paramètres importants

| Paramètre | Valeur | Rôle |
|---|---|---|
| Crop | `img[35:1130, 5:1995]` | Zone utile du cadre |
| Itérations correction | 20 | Convergence confirmée |
| Boost strength | 1.5 | Multiplicateur du ratio d'obscurité |
| CV seuil abeilles | 0.30 | Au-dessus = texture abeilles |
| CLAHE clipLimit | 3.0 | Rehaussement de contraste |
| CLAHE tileGrid | (8, 8) | Taille des tuiles |
| Seuil adaptatif blockSize | 21 | Taille voisinage |
| Seuil adaptatif C | 2 | Constante soustraite |
| Z-score k | 15 et 21 | Rayon de normalisation locale |
| Circularité min | 0.25 | Rejette les formes très allongées |
| Dédoublonnage | grille 8x8 px | Évite les doublons spatiaux |
| Dilatation masque | ellipse 5x5, 2 iter | Marge de sécurité autour des cellules |

## Approches testées et écartées

- **Inversion manuelle des zones** : nécessite un œil humain pour identifier les zones, pas scalable.
- **Détection par gradient morphologique** : ne donne que ~60 cellules, le réseau de murs est trop fragmenté.
- **Laplacien** : 14-60 cellules selon les paramètres, trop bruité.
- **Canny edges** : ~20 cellules, les bords ne forment pas de régions fermées exploitables.
- **Exclusion de zones (lock pixels après détection)** : convergence prématurée si trop fréquent. Optimal à 1 exclusion toutes les 5 itérations (940 cells) mais la fusion sans exclusion fait mieux (1104).
- **Carte de polarité (white/black top-hat)** : la zone gauche n'est pas franchement "inversée" mais plutôt à très faible contraste — la normalisation z-score résout mieux le problème.

## Structure des fichiers

```
pipeline_v3.py          # Script principal — génère tout
step_01_original.png    # Image croppée
step_02_heatmap_centre.png   # Heatmap boost 3 iter
step_03_corrected_centre.png # Image après 3 iter
step_04_heatmap_gauche.png   # Heatmap boost 20 iter
step_05_corrected_final.png  # Image corrigée finale
step_06_detection_brute.png  # Tous les contours bruts
step_07_filtre_aire.png      # Vert/rouge/bleu par taille
step_08_zscore.png           # Image normalisée z-score
step_09_zscore_detect.png    # Détections sur z-score
step_10_fusion.png           # Fusion finale
step_11_mask.png             # Masque binaire
step_12_blackout.png         # Résultat final
pipeline_complet.png         # Récapitulatif visuel 12 étapes
```

## Dépendances

```
opencv-python (cv2)
numpy
matplotlib (uniquement pour la visualisation)
```

## Points d'attention pour le développement futur

1. **Le crop est codé en dur** (`img[35:1130, 5:1995]`) — à adapter si la caméra bouge.
2. **La calibration d'aire se fait sur la moitié droite** — suppose que cette zone est toujours bien éclairée.
3. **20 itérations** c'est conservateur, 10-15 suffisent probablement (les gains sont marginaux après 10).
4. **Le pipeline prend ~30s** par image (21 passes × 3 détections) — optimisable en réduisant les passes ou en parallélisant.
5. **Les 36.4% de blackout** laissent ~60% de l'image non couverte — principalement les zones d'abeilles, les bords du cadre, et les zones à contraste trop faible (extrême gauche en haut).
