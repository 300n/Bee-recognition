"""
krcnn_predict.py
----------------
Load a trained Keypoint R-CNN checkpoint and run inference on:
  • every image in the test split   (--mode test)
  • a single arbitrary image file   (--image path/to/img.png)

Outputs
  outputs/krcnn_test_predictions/   one PNG per test image
  outputs/krcnn_grid.png            composite grid

Colour convention
  bbox     blue outline
  KP 0     red
  KP 1     amber
  KP 2     sky-blue
  label    white text on dark background

Usage:
    python krcnn_predict.py                          # run on test split
    python krcnn_predict.py --image my_photo.png     # single image
    python krcnn_predict.py --score 0.4              # lower detection threshold
"""

import argparse, math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from krcnn_dataset import (
    BeeKeypointRCNNDataset, collate_fn,
    _load_coco, _build_image_map, _split_image_ids,
)
from krcnn_train import build_krcnn, NUM_CLASSES, NUM_KEYPOINTS, CLASS_NAMES

CKPT_PATH = config.OUTPUT_DIR / "krcnn_best.pt"

KP_COLORS_BGR = [
    ( 50,  50, 220),   # red
    ( 30, 160, 240),   # amber
    (220, 160,  40),   # sky-blue
]
BBOX_COLOR  = (200,  80,  20)   # blue outline
TEXT_BG     = ( 30,  30,  30)
TEXT_FG     = (255, 255, 255)


# ── Load model ─────────────────────────────────────────────────────────────

def load_model(device):
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = build_krcnn(
        num_classes=ckpt.get("num_classes", NUM_CLASSES),
        num_keypoints=ckpt.get("num_keypoints", NUM_KEYPOINTS),
        min_size=ckpt.get("min_size", 256),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded KRCNN  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model


# ── Drawing ─────────────────────────────────────────────────────────────────

def draw_predictions(img_bgr: np.ndarray, prediction: dict,
                     score_threshold: float = 0.5) -> np.ndarray:
    out = img_bgr.copy()
    boxes  = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    kps    = prediction["keypoints"].cpu().numpy()          # (N, K, 3)
    kp_sc  = prediction.get("keypoints_scores", None)
    if kp_sc is not None:
        kp_sc = kp_sc.cpu().numpy()                          # (N, K)

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls = int(labels[i])
        cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, 2, cv2.LINE_AA)

        # Label
        label_str = f"{cls_name}  {scores[i]:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label_str, font, 0.45, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+6, y1), TEXT_BG, -1)
        cv2.putText(out, label_str, (x1+3, y1-3), font, 0.45, TEXT_FG, 1, cv2.LINE_AA)

        # Keypoints
        for k in range(kps.shape[1]):
            vis = kps[i, k, 2]
            if vis < 0.5:
                continue
            kconf = kp_sc[i, k] if kp_sc is not None else 1.0
            if kconf < 0.1:
                continue
            px, py = int(kps[i, k, 0]), int(kps[i, k, 1])
            col = KP_COLORS_BGR[k % len(KP_COLORS_BGR)]
            cv2.circle(out, (px, py), 7, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), 6, col,     -1, cv2.LINE_AA)
            cv2.putText(out, str(k), (px+6, py-4), font, 0.4, col, 1, cv2.LINE_AA)

    return out


# ── Inference on test split ────────────────────────────────────────────────

@torch.no_grad()
def predict_test_set(model, device, score_threshold, cols=4):
    data    = _load_coco(config.ANNOTATION_FILE)
    all_ids = [img["id"] for img in data["images"]]
    _, _, test_ids = _split_image_ids(all_ids, seed=config.RANDOM_SEED)
    test_ds = BeeKeypointRCNNDataset(
        config.ANNOTATION_FILE, config.DATA_ROOT, test_ids, augment=False
    )
    loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn, num_workers=0)

    out_dir = config.OUTPUT_DIR / "krcnn_test_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for images, targets in loader:
        img_t   = images[0]
        img_id  = targets[0]["image_id"].item()
        img_info = _build_image_map(data["images"])[img_id]
        fname    = img_info["file_name"]

        preds = model([img_t.to(device)])[0]

        # Convert tensor back to BGR for drawing
        img_np  = (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        annotated = draw_predictions(img_bgr, preds, score_threshold)

        out_path = out_dir / fname
        cv2.imwrite(str(out_path), annotated)
        rendered.append((cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), fname))

    print(f"Saved {len(rendered)} images → {out_dir}")
    _make_grid(rendered, cols, config.OUTPUT_DIR / "krcnn_grid.png")


# ── Inference on a single file ─────────────────────────────────────────────

@torch.no_grad()
def predict_single_file(model, device, image_path, score_threshold):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t   = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1)

    preds = model([img_t.to(device)])[0]
    annotated = draw_predictions(img_bgr, preds, score_threshold)

    out_path = config.OUTPUT_DIR / f"krcnn_{Path(image_path).name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved → {out_path}")

    n_det = int((preds["scores"] >= score_threshold).sum())
    print(f"Detected {n_det} bee(s)")
    for i in range(len(preds["boxes"])):
        if preds["scores"][i] < score_threshold:
            continue
        cls  = int(preds["labels"][i])
        sc   = float(preds["scores"][i])
        print(f"  [{i}] {CLASS_NAMES[cls]}  {sc:.1%}  "
              f"box={preds['boxes'][i].int().tolist()}")


# ── Grid helper ────────────────────────────────────────────────────────────

def _make_grid(rendered, cols, out_path):
    n    = len(rendered)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = np.array(axes).flatten()
    for idx, (img_rgb, fname) in enumerate(rendered):
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(fname[-20:], fontsize=6)
        axes[idx].axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=[0.2, 0.2, 0.86], label="KP 0"),
        Patch(facecolor=[0.94, 0.63, 0.12], label="KP 1"),
        Patch(facecolor=[0.16, 0.63, 0.86], label="KP 2"),
    ], loc="lower right", fontsize=9)
    fig.suptitle("Keypoint R-CNN — Test predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"Grid → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   type=str,   default=None)
    parser.add_argument("--score",   type=float, default=0.5)
    parser.add_argument("--cols",    type=int,   default=4)
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    model = load_model(device)

    if args.image:
        predict_single_file(model, device, args.image, args.score)
    else:
        predict_test_set(model, device, args.score, cols=args.cols)
