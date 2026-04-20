"""
krcnn_train.py
--------------
Fine-tune Keypoint R-CNN (ResNet-50 + FPN backbone, pretrained on COCO)
for bee detection + keypoint estimation.

Changes vs. the pretrained model
  • box_predictor  → 3 classes (background / cleaning bee / normal bee)
  • keypoint_predictor → 3 keypoints  (vs 17 on COCO)

Training strategy
  • Epochs 1-3  : backbone frozen, only heads train
  • Epoch 4+    : entire network fine-tuned with 10× lower LR on backbone

Usage:
    python krcnn_train.py
    python krcnn_train.py --epochs 60 --batch 2
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

import config
from krcnn_dataset import get_krcnn_datasets, collate_fn

# ── Constants ──────────────────────────────────────────────────────────────
NUM_CLASSES    = 3          # 0=bg, 1=cleaning bee, 2=normal bee
NUM_KEYPOINTS  = config.MAX_KEYPOINTS   # 3
WARMUP_EPOCHS  = 3
CKPT_PATH      = config.OUTPUT_DIR / "krcnn_best.pt"
HIST_PATH      = config.OUTPUT_DIR / "krcnn_history.json"
CLASS_NAMES    = ["__background__"] + config.CLASS_NAMES


# ── Device ─────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model ──────────────────────────────────────────────────────────────────
def build_krcnn(num_classes=NUM_CLASSES, num_keypoints=NUM_KEYPOINTS, min_size=256):
    """Load pretrained KeypointRCNN and replace the two task-specific heads."""
    model = keypointrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=min_size, max_size=min_size,
        # Our images have ≤15 bees — drastically reduce RPN/NMS overhead
        rpn_pre_nms_top_n_train=256, rpn_post_nms_top_n_train=128,
        rpn_pre_nms_top_n_test=128,  rpn_post_nms_top_n_test=64,
        box_detections_per_img=20,
    )

    # ── Replace box predictor ────────────────────────────────────────
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ── Replace keypoint predictor ───────────────────────────────────
    # Keep the keypoint_head (conv feature extractor) — only swap final layer
    in_ch = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_ch, num_keypoints)

    return model


def freeze_backbone(model):
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = True


# ── Training helpers ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total = 0.0; n = 0
    loss_breakdown = {}

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total += loss.item()
        for k, v in loss_dict.items():
            loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v.item()
        n += 1

    avg = {k: v/max(n,1) for k, v in loss_breakdown.items()}
    return total / max(n, 1), avg


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run in train mode to get losses (the standard approach for detection models),
    then switch back.  Also collect detection stats for a quick mAP proxy.
    """
    model.train()
    total = 0.0; n = 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
        n += 1
    model.eval()
    return total / max(n, 1)


# ── Main training loop ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--lr",       type=float, default=5e-4)
    parser.add_argument("--patience", type=int,   default=12)
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading dataset …")
    train_ds, val_ds, _ = get_krcnn_datasets(config.ANNOTATION_FILE, config.DATA_ROOT)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"  Batches/epoch: {len(train_loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_krcnn()
    model.to(device)
    freeze_backbone(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params — total: {total/1e6:.2f}M   trainable: {trainable/1e6:.2f}M")

    # ── Optimiser ─────────────────────────────────────────────────────
    def make_optimizer(model):
        backbone_ids = {id(p) for n, p in model.named_parameters() if "backbone" in n}
        head_params     = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
        groups = [{"params": head_params, "lr": args.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": args.lr * 0.1})
        return torch.optim.AdamW(groups, weight_decay=1e-4)

    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0

    print(f"{'='*62}")
    print(f"  Training Keypoint R-CNN  ({args.epochs} epochs max)")
    print(f"{'='*62}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Backbone stays frozen — pretrained COCO features are sufficient
        # (full backbone fine-tune stalls on MPS with 59M params)

        train_loss, breakdown = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # Format loss breakdown
        bd_str = "  ".join(f"{k.replace('loss_','')[:6]}={v:.3f}"
                           for k, v in breakdown.items())
        print(
            f"  Ep {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"[{bd_str}]  lr={lr_now:.1e}  {elapsed:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "val_loss":    val_loss,
                "num_classes": NUM_CLASSES,
                "num_keypoints": NUM_KEYPOINTS,
                "min_size": 256,
            }, CKPT_PATH)
            print(f"    ✓ Saved checkpoint  (val_loss={val_loss:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save history
    with open(HIST_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val_loss: {best_val:.4f}  → {CKPT_PATH}")
    return best_val


if __name__ == "__main__":
    main()
