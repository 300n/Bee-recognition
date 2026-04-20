"""
evaluate.py
-----------
Loads saved checkpoints and produces:
  1. Per-model metrics table on the test set
  2. Visual comparison of keypoint predictions vs ground truth
  3. Training curves (loss over epochs)

Usage:
    python evaluate.py [--model MODEL_NAME] [--visualise N]

    --model     : evaluate only one model (default: all)
    --visualise : render N random test samples with predictions (default 8)
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import get_datasets, compute_class_weights
from losses import BeeKeypointLoss
from metrics import MetricAccumulator
from models import build_model

COLORS = {
    "pred": (255, 80,  80),   # red
    "gt":   (80,  200, 80),   # green
    "bbox": (80,  80,  255),  # blue
}


# ── Load one model ─────────────────────────────────────────────────────────

def load_model(model_name: str, device):
    ckpt_path = config.OUTPUT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model(model_name, config.NUM_CLASSES, config.MAX_KEYPOINTS)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    if hasattr(model, "unfreeze_backbone"):
        model.unfreeze_backbone()
    return model, ckpt.get("epoch", "?"), ckpt.get("val_loss", "?")


# ── Metrics table ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, test_loader, criterion, device):
    total_loss = 0.0
    acc = MetricAccumulator()
    n = 0
    for batch in test_loader:
        images  = batch["image"].to(device)
        gt_kp   = batch["keypoints"].to(device)
        gt_vis  = batch["visibility"].to(device)
        gt_cls  = batch["label"].to(device)
        bbox_wh = batch["bbox_wh"]

        pred_cls, pred_kp, pred_vis = model(images)
        loss, _ = criterion(pred_cls, pred_kp, pred_vis, gt_cls, gt_kp, gt_vis)
        total_loss += loss.item()
        acc.update(pred_cls, pred_kp, gt_cls, gt_kp, gt_vis, bbox_wh)
        n += 1

    return total_loss / max(n, 1), acc.compute(config.PCK_THRESHOLDS)


# ── Visualisation ──────────────────────────────────────────────────────────

NORM_MEAN = np.array(config.NORM_MEAN, dtype=np.float32)
NORM_STD  = np.array(config.NORM_STD,  dtype=np.float32)


def _denorm(tensor):
    """Convert normalised CHW tensor → HWC uint8 numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * NORM_STD + NORM_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_keypoints(img, kp_norm, vis, color, radius=5):
    h, w = img.shape[:2]
    out = img.copy()
    for i in range(len(kp_norm)):
        if vis[i] < 0.5:
            continue
        px = int(kp_norm[i, 0] * (w - 1))
        py = int(kp_norm[i, 1] * (h - 1))
        cv2.circle(out, (px, py), radius, color, -1)
        cv2.putText(out, str(i), (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def visualise_predictions(model, test_ds, model_name: str, n_samples: int, device):
    indices = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for ax_idx, sample_idx in enumerate(indices):
        sample = test_ds[sample_idx]
        image_t = sample["image"].unsqueeze(0).to(device)
        gt_kp   = sample["keypoints"].numpy()
        gt_vis  = sample["visibility"].numpy()
        gt_cls  = sample["label"].item()

        pred_cls, pred_kp, pred_vis = model(image_t)
        pred_kp_np  = pred_kp[0].cpu().numpy()
        pred_vis_np = torch.sigmoid(pred_vis[0]).cpu().numpy()
        pred_cls_idx = pred_cls[0].argmax().item()

        img_rgb = _denorm(sample["image"])
        img_rgb = _draw_keypoints(img_rgb, gt_kp,      gt_vis,              COLORS["gt"])
        img_rgb = _draw_keypoints(img_rgb, pred_kp_np, pred_vis_np > 0.5,   COLORS["pred"])

        axes[ax_idx].imshow(img_rgb)
        gt_name   = config.CLASS_NAMES[gt_cls]
        pred_name = config.CLASS_NAMES[pred_cls_idx]
        match = "✓" if gt_cls == pred_cls_idx else "✗"
        axes[ax_idx].set_title(
            f"GT: {gt_name}\nPred: {pred_name} {match}",
            fontsize=8,
            color="green" if gt_cls == pred_cls_idx else "red",
        )
        axes[ax_idx].axis("off")

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(COLORS["gt"])  / 255, label="GT keypoints"),
        Patch(facecolor=np.array(COLORS["pred"])/ 255, label="Predicted keypoints"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.suptitle(f"Keypoint Predictions – {model_name}", fontsize=13, y=1.01)

    out_path = config.OUTPUT_DIR / f"{model_name}_predictions.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualisation → {out_path}")


# ── Training curve ─────────────────────────────────────────────────────────

def plot_training_curves(model_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in model_names:
        hist_path = config.OUTPUT_DIR / f"{name}_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], label=name)
        axes[1].plot(epochs, h["val_loss"],   label=name)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out = config.OUTPUT_DIR / "training_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved training curves → {out}")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--visualise",  type=int, default=8)
    args = parser.parse_args()

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    print("\nLoading dataset …")
    _, _, test_ds = get_datasets(config.ANNOTATION_FILE, config.DATA_ROOT)
    # We need class_weights only to instantiate the criterion
    from dataset import _load_coco, _build_image_map, _group_annotations, _split_image_ids
    data       = _load_coco(config.ANNOTATION_FILE)
    all_ids    = [img["id"] for img in data["images"]]
    train_ids, _, _ = _split_image_ids(all_ids, seed=config.RANDOM_SEED)
    from dataset import BeeKeypointDataset
    _tmp = BeeKeypointDataset.__new__(BeeKeypointDataset)
    _tmp.samples = [
        (None, ann) for ann in data["annotations"]
        if ann["image_id"] in train_ids and ann["category_id"] in config.CATEGORY_MAP
    ]
    class_weights = compute_class_weights(_tmp)

    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    criterion = BeeKeypointLoss(class_weights=class_weights.to(device))

    model_names = [args.model] if args.model else list(config.MODELS.keys())
    results = []

    for name in model_names:
        try:
            model, best_epoch, best_val = load_model(name, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        test_loss, metrics = evaluate_model(model, test_loader, criterion, device)

        row = {"model": name, "params_M": n_params / 1e6,
               "best_epoch": best_epoch, "test_loss": test_loss, **metrics}
        results.append(row)
        print(f"\n  {name} (epoch {best_epoch}, {n_params/1e6:.2f}M params)")
        for k, v in {**{"test_loss": test_loss}, **metrics}.items():
            print(f"    {k:>12}: {v:.4f}")

        if args.visualise > 0:
            visualise_predictions(model, test_ds, name, args.visualise, device)

    # Comparison table
    if results:
        print(f"\n{'─'*95}")
        print(f"{'Model':<20} {'Params':>8} {'Epoch':>6} {'Loss':>7} {'Acc':>7} "
              f"{'PCK@05':>7} {'PCK@10':>7} {'PCK@20':>7} {'NME':>7} {'OKS':>7}")
        print(f"{'─'*95}")
        for r in results:
            print(
                f"{r['model']:<20} {r['params_M']:>7.2f}M {str(r['best_epoch']):>6} "
                f"{r['test_loss']:>7.4f} {r['accuracy']:>7.3f} "
                f"{r.get('pck@05',0):>7.3f} {r.get('pck@10',0):>7.3f} "
                f"{r.get('pck@20',0):>7.3f} {r['nme']:>7.4f} {r['oks']:>7.4f}"
            )
        print(f"{'─'*95}")

    plot_training_curves(model_names)


if __name__ == "__main__":
    main()
