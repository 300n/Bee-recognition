"""
predict_on_images.py
--------------------
Runs the ResNet18 model on every test-split image and draws, on the original
full-resolution image:
  • bounding box of each annotated bee (blue)
  • ground-truth keypoints              (green dots)
  • predicted keypoints                 (red dots)
  • predicted class label               (text above bbox)

Output: outputs/test_image_predictions/  — one PNG per test image.
A single composite grid is also saved to outputs/test_predictions_grid.png.

Usage:
    python predict_on_images.py [--cols N]   (default: 4 images per row)
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import _load_coco, _build_image_map, _group_annotations, _split_image_ids
from models import build_model

# ── Colours (BGR for cv2) ──────────────────────────────────────────────────
C_BBOX   = (220, 100,  30)   # blue-ish
C_GT_KP  = ( 50, 200,  50)   # green
C_PR_KP  = ( 50,  50, 220)   # red
C_TEXT   = (255, 255, 255)   # white
NORM_MEAN = np.array(config.NORM_MEAN, dtype=np.float32)
NORM_STD  = np.array(config.NORM_STD,  dtype=np.float32)


# ── helpers ────────────────────────────────────────────────────────────────

def _parse_bbox(raw):
    return [float(v) for v in raw]


def _crop_padded(img, x, y, w, h, pad=config.BBOX_PADDING):
    ih, iw = img.shape[:2]
    x1 = max(0,    x - pad * w)
    y1 = max(0,    y - pad * h)
    x2 = min(iw,   x + (1 + pad) * w)
    y2 = min(ih,   y + (1 + pad) * h)
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return img, 0.0, 0.0, float(iw), float(ih)
    return crop, x1, y1, x2 - x1, y2 - y1


def _prepare_crop(crop_bgr):
    """BGR crop → normalised CHW tensor."""
    crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (config.CROP_SIZE, config.CROP_SIZE))
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - NORM_MEAN) / NORM_STD
    return torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)


def _draw_keypoint(img, px, py, color, radius=6):
    cv2.circle(img, (int(px), int(py)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(px), int(py)), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_bbox(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)


def _put_label(img, text, x, y, color=C_TEXT, bg=(40, 40, 40)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (int(x), int(y) - th - 4), (int(x) + tw + 4, int(y) + 2), bg, -1)
    cv2.putText(img, text, (int(x) + 2, int(y) - 2), font, scale, color, thick, cv2.LINE_AA)


# ── main ───────────────────────────────────────────────────────────────────

def run(cols: int = 4):
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────
    ckpt_path = config.OUTPUT_DIR / "resnet18_best.pt"
    model = build_model("resnet18", config.NUM_CLASSES, config.MAX_KEYPOINTS)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    if hasattr(model, "unfreeze_backbone"):
        model.unfreeze_backbone()
    print(f"Loaded ResNet18 from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")

    # ── Load COCO annotations ────────────────────────────────────────────
    data      = _load_coco(config.ANNOTATION_FILE)
    image_map = _build_image_map(data["images"])
    ann_groups = _group_annotations(data["annotations"])
    all_ids   = [img["id"] for img in data["images"]]
    _, _, test_ids = _split_image_ids(all_ids, seed=config.RANDOM_SEED)
    print(f"Test images: {len(test_ids)}")

    out_dir = config.OUTPUT_DIR / "test_image_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []   # (img_rgb, filename) for the grid

    for img_id in sorted(test_ids):
        img_info = image_map[img_id]
        fname    = img_info["file_name"]
        img_path = config.DATA_ROOT / fname

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [skip] cannot read {img_path}")
            continue

        canvas = img_bgr.copy()
        anns   = ann_groups.get(img_id, [])

        for ann in anns:
            if ann["category_id"] not in config.CATEGORY_MAP:
                continue
            x, y, w, h = _parse_bbox(ann["bbox"])
            pad = config.BBOX_PADDING
            x1_b = max(0, x - pad * w);  y1_b = max(0, y - pad * h)
            x2_b = min(img_bgr.shape[1], x + (1 + pad) * w)
            y2_b = min(img_bgr.shape[0], y + (1 + pad) * h)

            # ── GT keypoints ─────────────────────────────────────────
            kp_raw   = ann.get("keypoints", [])
            n_kp     = len(kp_raw) // 3
            gt_kp_px = []
            for i in range(min(n_kp, config.MAX_KEYPOINTS)):
                kx = float(kp_raw[i * 3])
                ky = float(kp_raw[i * 3 + 1])
                v  = float(kp_raw[i * 3 + 2])
                if v > 0:
                    gt_kp_px.append((kx, ky))
                    _draw_keypoint(canvas, kx, ky, C_GT_KP, radius=5)

            # ── Run model on padded crop ─────────────────────────────
            crop_bgr, cx1, cy1, cw, ch = _crop_padded(img_bgr, x, y, w, h)
            tensor = _prepare_crop(crop_bgr).to(device)

            with torch.no_grad():
                pred_cls, pred_kp, pred_vis = model(tensor)

            pred_label = pred_cls[0].argmax().item()
            pred_kp_norm = pred_kp[0].cpu().numpy()    # (K, 2) in [0,1]
            pred_vis_prob = torch.sigmoid(pred_vis[0]).cpu().numpy()

            # Convert normalised crop coords → full image pixels
            for i in range(config.MAX_KEYPOINTS):
                if pred_vis_prob[i] < 0.5:
                    continue
                px = cx1 + pred_kp_norm[i, 0] * cw
                py = cy1 + pred_kp_norm[i, 1] * ch
                _draw_keypoint(canvas, px, py, C_PR_KP, radius=5)

            # ── Bounding box + label ─────────────────────────────────
            _draw_bbox(canvas, x1_b, y1_b, x2_b, y2_b, C_BBOX)
            cls_name = config.CLASS_NAMES[pred_label]
            conf     = torch.softmax(pred_cls[0], dim=0)[pred_label].item()
            label_str = f"{cls_name} {conf:.0%}"
            _put_label(canvas, label_str, x1_b, y1_b)

        # Save individual image
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), canvas)
        rendered.append((cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), fname))

    print(f"Saved {len(rendered)} annotated images → {out_dir}")

    # ── Composite grid ───────────────────────────────────────────────────
    n = len(rendered)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten()

    for idx, (img_rgb, fname) in enumerate(rendered):
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(fname.split(".")[0][-16:], fontsize=6)
        axes[idx].axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=np.array(C_GT_KP[::-1]) / 255, label="GT keypoints"),
        Patch(facecolor=np.array(C_PR_KP[::-1]) / 255, label="Predicted keypoints"),
        Patch(facecolor=np.array(C_BBOX[::-1])  / 255, label="Bounding box"),
    ]
    fig.legend(handles=legend, loc="lower right", fontsize=9)
    fig.suptitle("ResNet18 — Predictions on Test Images", fontsize=14, y=1.005)
    plt.tight_layout()

    grid_path = config.OUTPUT_DIR / "test_predictions_grid.png"
    plt.savefig(grid_path, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"Saved grid → {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()
    run(cols=args.cols)
