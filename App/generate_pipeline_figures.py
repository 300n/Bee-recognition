"""
generate_pipeline_figures.py
----------------------------
Generates before/after figures for each of the 9 unique processing steps
(3 contrast + 6 filters) and a summary grid, saved to figures/ for the paper.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_transforms import (
    contrast_original, contrast_clahe, contrast_retinex,
    filter_bilateral, filter_nlm, filter_bm3d,
    filter_snn, filter_kramer_bruckner, filter_epoaf,
    apply_pipeline, PIPELINE_NAMES, pipeline_label,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Pick a representative image
TEST_IMG = Path(__file__).parent / "datasets/bee_yolo/images/test/M01C02_004098-3_png.rf.mpWxokd8MKwwiFJuaDX8.png"

img_bgr = cv2.imread(str(TEST_IMG))
assert img_bgr is not None, f"Cannot read {TEST_IMG}"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Contrast steps ────────────────────────────────────────────────────────────
CONTRAST = [
    ("Original (no-op)",         "original",  contrast_original(img_rgb)),
    ("CLAHE",                    "clahe",     contrast_clahe(img_rgb)),
    ("Multi-Scale Retinex (MSR)","retinex",   contrast_retinex(img_rgb)),
]

# ── Filter steps (applied on original contrast) ───────────────────────────────
FILTERS = [
    ("Bilateral",        "bilateral",       filter_bilateral(img_rgb)),
    ("NLM",              "nlm",             filter_nlm(img_rgb)),
    ("BM3D",             "bm3d",            filter_bm3d(img_rgb)),
    ("SNN",              "snn",             filter_snn(img_rgb)),
    ("Kramer-Brückner",  "kramer_bruckner", filter_kramer_bruckner(img_rgb)),
    ("EPOAF",            "epoaf",           filter_epoaf(img_rgb)),
]

def save_before_after(title, key, processed, original=img_rgb):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original);   axes[0].set_title("Original",  fontsize=10); axes[0].axis("off")
    axes[1].imshow(processed);  axes[1].set_title(title,        fontsize=10); axes[1].axis("off")
    fig.tight_layout(pad=0.5)
    out = FIGURES_DIR / f"pipeline_{key}.pdf"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

print("Generating contrast before/after figures…")
for title, key, proc in CONTRAST:
    save_before_after(title, f"contrast_{key}", proc)

print("Generating filter before/after figures…")
for title, key, proc in FILTERS:
    save_before_after(title, f"filter_{key}", proc)

# ── Summary grid: original vs all 18 pipelines ranked ────────────────────────
# Try to load ranking from yolo_pipeline_results.json if available
import json
results_path = Path(__file__).parent / "outputs/yolo_pipeline_results.json"
if results_path.exists():
    with open(results_path) as f:
        yolo_res = json.load(f)
    ranked = sorted(
        [(k, v) for k, v in yolo_res.items()],
        key=lambda x: x[1].get("metrics", {}).get("test_pose_mAP50", 0),
        reverse=True,
    )
    print(f"\nRanking from {len(ranked)} completed pipelines:")
    for name, res in ranked:
        m = res.get("metrics", {})
        print(f"  {pipeline_label(name):30s}  Pose mAP50={m.get('test_pose_mAP50','?')}")
else:
    ranked = []
    print("\nNo yolo_pipeline_results.json yet — skipping ranking figure.")

# ── Compact 3×6 grid (contrast rows × filter cols) ───────────────────────────
print("\nGenerating pipeline grid figure…")
contrasts = ["original", "clahe", "retinex"]
filters   = ["bilateral", "nlm", "bm3d", "snn", "kramer_bruckner", "epoaf"]
C_LABELS  = ["Original", "CLAHE", "Retinex"]
F_LABELS  = ["Bilateral", "NLM", "BM3D", "SNN", "Kramer-\nBrückner", "EPOAF"]

fig, axes = plt.subplots(3, 6, figsize=(14, 7))
for ri, (c, cl) in enumerate(zip(contrasts, C_LABELS)):
    for ci, (e, el) in enumerate(zip(filters, F_LABELS)):
        name = f"{c}_{e}"
        proc = apply_pipeline(img_rgb, name)
        ax   = axes[ri, ci]
        ax.imshow(proc)
        ax.axis("off")
        if ri == 0:
            ax.set_title(el, fontsize=8, pad=2)
        if ci == 0:
            ax.set_ylabel(cl, fontsize=8, rotation=90, labelpad=4)

fig.suptitle("All 18 preprocessing pipelines (rows: contrast step, cols: filter step)",
             fontsize=10, y=1.01)
fig.tight_layout(pad=0.3)
out_grid = FIGURES_DIR / "pipeline_grid.pdf"
fig.savefig(str(out_grid), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out_grid.name}")

# ── Bar chart: Pose mAP50 per pipeline (if data available) ───────────────────
if ranked:
    print("Generating ranking bar chart…")
    names  = [pipeline_label(k) for k, _ in ranked]
    scores = [v.get("metrics", {}).get("test_pose_mAP50", 0) for _, v in ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    colors = ["#00e5b0" if i == 0 else "#4b9cd3" for i in range(len(names))]
    bars = ax.barh(names[::-1], [s*100 for s in scores[::-1]], color=colors[::-1], height=0.65)
    ax.set_xlabel("Test Pose mAP50 (%)", fontsize=10)
    ax.set_title("YOLOv8-pose: Pose mAP50 by preprocessing pipeline", fontsize=11)
    ax.axvline(scores[0]*100, color="#00e5b0", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, score in zip(bars, [s*100 for s in scores[::-1]]):
        ax.text(score + 0.3, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}%", va="center", fontsize=7.5)
    fig.tight_layout()
    out_bar = FIGURES_DIR / "pipeline_ranking.pdf"
    fig.savefig(str(out_bar), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_bar.name}")

print("\nDone. Figures in:", FIGURES_DIR)
