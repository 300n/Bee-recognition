# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for beehive frame image analysis. Camera captures grayscale images of honeycomb frames from a raking angle, creating a strong left→right luminosity gradient (~2× darker on the left). The core challenge: detect and separate bees from the hexagonal honeycomb background pattern for activity analysis.

## Running Scripts

All scripts are standalone Python files — run directly:

```bash
python pattern_v3.py          # Main honeycomb suppression pipeline (latest)
python bee_detection_improved.py
python heatmap/heatmap_final.py
python local/bee_enhance_pipeline.py
```

No build system, test runner, or package manager. Dependencies: `opencv-python`, `numpy`, `matplotlib`, `scipy`.

## Architecture & Pipeline Evolution

The work has evolved through several approaches, all stored as flat Python scripts:

### Primary Pipeline (pattern_v3.py)
The most complete solution. 12-step pipeline:
1. **Brightness correction** (20 adaptive iterations using coefficient of variation to distinguish bee texture from perspective shadows)
2. **Honeycomb cell detection** — two parallel branches:
   - Pipeline A: CLAHE + adaptive threshold (works on ~70% of image)
   - Pipeline B: **Local z-score normalization** — the key discovery; polarity-agnostic detection that works where contrast inverts on the left edge
3. **Blackout**: detected cells are zeroed out, leaving only bees and empty zones

Full algorithm context with parameters: `local/CONTEXT_HONEYCOMB_PIPELINE.md`

### Heatmap Approach (heatmap/)
FFT-based blur detection: low-frequency components (large blobs = bees) are isolated via circular mask in frequency domain. Files: `heatmap_final.py`, `heatmap_CLAHE.py`, `heatmap_DoG.py`, `heatmap_wavelet.py`, `heatmap_bm3d.py` — comparative experiments.

### Statistical Model Approach (pattern_v1.py → v3.py)
Builds a background model from N aligned frames (ECC alignment via `cv2.findTransformECC`), computes per-pixel percentile background + std deviation map. Bees detected as z-score outliers against the background model.

### Local Enhancement (local/)
Post-detection scripts that enhance individual bee crops:
- `bee_enhance_pipeline.py` — SNN filter + multi-scale Laplacian + Multi-Scale Retinex
- `bee_behaviour_analysis.py` — rose diagram + local alignment from heading keypoints
- `bee_flow_chains.py`, `draw_bee_directions.py` — optical flow / motion chains

### Denoising Experiments (heatmap/debruitage*.py, autres/)
Butterworth filter, temporal variance/std deviation, BM3D, wavelet denoising — comparing noise reduction methods before bee detection.

## Image Data

- `images1_crop/` — Camera 1 frames (M01C01_*.png), ~1920×1200 grayscale
- `images2_crop/` — Camera 2 frames (M01C02_*.png)
- `images2_64/` — Downsampled 64px versions
- `abeilles_extraites/` — Individual bee crops extracted via `extraction_abeilles.py` from Roboflow JSON predictions (`predictions.json`)
- Output goes to `Output/` subdirectories

## Key Technical Parameters

Crop region (hardcoded to camera position): `img[35:1130, 5:1995]`

Critical thresholds from `local/CONTEXT_HONEYCOMB_PIPELINE.md`:
- CV threshold separating bee texture from shadow: `0.30`
- Z-score normalization kernel sizes: `k=15` and `k=21`
- Cell area filter: `[414, 646] px²` (auto-calibrated on right half of image)
- ECC alignment scale: `0.2` (downscaled for speed)

## Naming Conventions

- `MC01` / `MC02` suffix = camera identifier (Caméra 1, Caméra 2)
- `inter_NN_*.png` = intermediate step outputs from the 12-step pipeline
- `pipeline_output/` = runtime output directory (not committed)
