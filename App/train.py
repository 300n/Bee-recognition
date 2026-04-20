"""
train.py
--------
Trains each model in config.MODELS, evaluates on the validation set,
saves the best checkpoint, and prints a final comparison table on the test set.

Usage:
    python train.py [--model MODEL_NAME]

    --model : train only one specific model (optional; default: all models)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force unbuffered stdout so progress prints immediately to the terminal
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import get_datasets, compute_class_weights
from losses import BeeKeypointLoss
from metrics import MetricAccumulator
from models import build_model


# ── Device selection ───────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── DataLoader helpers ─────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    """Over-sample minority class (cleaning bees) to correct imbalance."""
    labels = [config.CATEGORY_MAP[ann["category_id"]] for _, ann in dataset.samples]
    from collections import Counter
    counts = Counter(labels)
    n_total = sum(counts.values())
    class_weight = {c: n_total / (len(counts) * cnt) for c, cnt in counts.items()}
    weights = [class_weight[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


def get_loaders(train_ds, val_ds, test_ds):
    sampler = make_weighted_sampler(train_ds)
    # pin_memory not supported on MPS; use persistent_workers to amortise spawn cost
    pin = torch.cuda.is_available()
    pw  = config.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=config.NUM_WORKERS,
        pin_memory=pin, drop_last=True,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=pin, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=pin, persistent_workers=pw,
    )
    return train_loader, val_loader, test_loader


# ── LR scheduler ──────────────────────────────────────────────────────────

def make_scheduler(optimizer, n_batches_per_epoch):
    if config.LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LR,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=n_batches_per_epoch,
            pct_start=config.WARMUP_EPOCHS / config.NUM_EPOCHS,
            anneal_strategy="cos",
        )
    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# ── Train one epoch ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        images    = batch["image"].to(device)
        gt_kp     = batch["keypoints"].to(device)
        gt_vis    = batch["visibility"].to(device)
        gt_cls    = batch["label"].to(device)

        pred_cls, pred_kp, pred_vis = model(images)
        loss, _ = criterion(pred_cls, pred_kp, pred_vis, gt_cls, gt_kp, gt_vis)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ── Evaluate one epoch ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    accumulator = MetricAccumulator()
    n = 0

    for batch in loader:
        images  = batch["image"].to(device)
        gt_kp   = batch["keypoints"].to(device)
        gt_vis  = batch["visibility"].to(device)
        gt_cls  = batch["label"].to(device)
        bbox_wh = batch["bbox_wh"]

        pred_cls, pred_kp, pred_vis = model(images)
        loss, _ = criterion(pred_cls, pred_kp, pred_vis, gt_cls, gt_kp, gt_vis)

        total_loss += loss.item()
        accumulator.update(pred_cls, pred_kp, gt_cls, gt_kp, gt_vis, bbox_wh)
        n += 1

    metrics = accumulator.compute(pck_thresholds=config.PCK_THRESHOLDS)
    return total_loss / max(n, 1), metrics


# ── Full training loop for one model ──────────────────────────────────────

def train_one_model(
    model_name: str,
    train_loader, val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model = build_model(model_name, config.NUM_CLASSES, config.MAX_KEYPOINTS)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    criterion = BeeKeypointLoss(class_weights=class_weights.to(device))

    # Separate learning rates: small for pretrained backbone, larger for head
    backbone_params, head_params = [], []
    has_backbone = hasattr(model, "backbone")
    if has_backbone:
        backbone_ids = set(id(p) for p in model.backbone.parameters())
        for p in model.parameters():
            (backbone_params if id(p) in backbone_ids else head_params).append(p)
    else:
        head_params = list(model.parameters())

    param_groups = [{"params": head_params, "lr": config.LR}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.LR * 0.1})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, len(train_loader))

    ckpt_path  = config.OUTPUT_DIR / f"{model_name}_best.pt"
    history    = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_val   = float("inf")
    patience   = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if has_backbone and epoch == config.WARMUP_EPOCHS + 1:
            model.unfreeze_backbone()
            print(f"  [epoch {epoch}] Backbone unfrozen")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={val_metrics['accuracy']:.3f}  "
            f"pck@10={val_metrics.get('pck@10', 0):.3f}  "
            f"nme={val_metrics['nme']:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save history
    with open(config.OUTPUT_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"model_name": model_name, "best_val_loss": best_val, "history": history}


# ── Final test evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def test_all_models(test_loader, class_weights, device):
    print(f"\n{'='*60}")
    print("  FINAL TEST SET EVALUATION")
    print(f"{'='*60}")
    criterion = BeeKeypointLoss(class_weights=class_weights.to(device))

    results = []
    for model_name in config.MODELS:
        ckpt_path = config.OUTPUT_DIR / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {model_name} — checkpoint not found")
            continue

        model = build_model(model_name, config.NUM_CLASSES, config.MAX_KEYPOINTS)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        # Unfreeze to count all params for fair comparison
        if hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()

        test_loss, metrics = evaluate(model, test_loader, criterion, device)
        n_params = sum(p.numel() for p in model.parameters())

        row = {
            "model":      model_name,
            "params_M":   n_params / 1e6,
            "test_loss":  test_loss,
            **metrics,
        }
        results.append(row)
        print(f"  {model_name}  loss={test_loss:.4f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # Comparison table
    print(f"\n{'─'*90}")
    header = f"{'Model':<20} {'Params':>8} {'Loss':>7} {'Acc':>7} {'PCK@05':>7} {'PCK@10':>7} {'PCK@20':>7} {'NME':>7} {'OKS':>7}"
    print(header)
    print(f"{'─'*90}")
    for r in results:
        print(
            f"{r['model']:<20} {r['params_M']:>7.2f}M {r['test_loss']:>7.4f} "
            f"{r['accuracy']:>7.3f} {r.get('pck@05',0):>7.3f} "
            f"{r.get('pck@10',0):>7.3f} {r.get('pck@20',0):>7.3f} "
            f"{r['nme']:>7.4f} {r['oks']:>7.4f}"
        )
    print(f"{'─'*90}")

    # Save
    with open(config.OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Train only this model (default: all)")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    print("\nLoading dataset …")
    train_ds, val_ds, test_ds = get_datasets(
        coco_json=config.ANNOTATION_FILE,
        image_dir=config.DATA_ROOT,
    )
    class_weights = compute_class_weights(train_ds)
    train_loader, val_loader, test_loader = get_loaders(train_ds, val_ds, test_ds)

    print(f"  Batches per epoch: {len(train_loader)}")

    # ── Train ─────────────────────────────────────────────────────────
    models_to_train = [args.model] if args.model else list(config.MODELS.keys())
    for model_name in models_to_train:
        if model_name not in config.MODELS:
            print(f"Unknown model '{model_name}'. Choices: {list(config.MODELS.keys())}")
            sys.exit(1)
        train_one_model(model_name, train_loader, val_loader, class_weights, device)

    # ── Test ──────────────────────────────────────────────────────────
    test_all_models(test_loader, class_weights, device)


if __name__ == "__main__":
    main()
