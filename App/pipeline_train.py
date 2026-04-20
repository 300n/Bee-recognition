"""
pipeline_train.py
-----------------
Train Keypoint R-CNN on each of the 15 preprocessing pipelines and evaluate
on the held-out test set.  Results are written incrementally to:
    outputs/pipeline_results.json

Usage:
    python pipeline_train.py                         # all 15, sequential
    python pipeline_train.py --pipeline clahe_nlm    # single pipeline
    python pipeline_train.py --epochs 30 --patience 10
    python pipeline_train.py --resume                # skip already-done pipelines
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from krcnn_dataset import get_krcnn_datasets, collate_fn, BeeKeypointRCNNDataset, \
    _load_coco, _split_image_ids, _build_image_map
from krcnn_train import build_krcnn, NUM_CLASSES, NUM_KEYPOINTS
from pipeline_transforms import PIPELINE_NAMES, apply_pipeline, pipeline_label

RESULTS_PATH = config.OUTPUT_DIR / "pipeline_results.json"
STATUS_PATH  = config.OUTPUT_DIR / "pipeline_status.json"
CKPT_DIR     = config.OUTPUT_DIR / "pipelines"


def _write_status(status: dict):
    """Atomically write the live status file (app polls this)."""
    tmp = STATUS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f)
    tmp.replace(STATUS_PATH)


# ── Augmented dataset wrapper ─────────────────────────────────────────────────

class PipelineBeeDataset(BeeKeypointRCNNDataset):
    """Wraps BeeKeypointRCNNDataset and applies a preprocessing pipeline to
    every image before returning it."""
    def __init__(self, pipeline_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_name = pipeline_name

    def __getitem__(self, idx):
        img_t, target = super().__getitem__(idx)
        # img_t is float32 (C,H,W) in [0,1] — convert to uint8 RGB, apply, back
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_np = apply_pipeline(img_np, self.pipeline_name)
        img_t  = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img_t, target


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def evaluate_test(model, test_ds, device, score_t=0.5):
    """Return dict of detection / classification / keypoint metrics."""
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    model.eval()
    tp = fp = fn = 0
    iou_list, kp_errors = [], []
    cls_correct = cls_total = 0
    kp_total_gt = 0

    for images, targets in loader:
        img_t  = images[0].to(device)
        target = targets[0]
        preds  = model([img_t])[0]

        gt_boxes  = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()
        gt_kps    = target["keypoints"].numpy()  # (N,K,3)

        sc   = preds["scores"].cpu().numpy()
        keep = sc >= score_t
        pb   = preds["boxes"][keep].cpu().numpy()
        pl   = preds["labels"][keep].cpu().numpy()
        pk   = preds["keypoints"][keep].cpu().numpy()

        matched_gt = set()
        for pi in range(len(pb)):
            best_iou, best_gi = 0.0, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                iou = _box_iou(pb[pi], gt_boxes[gi])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5:
                matched_gt.add(best_gi)
                tp += 1
                iou_list.append(best_iou)
                if pl[pi] == gt_labels[best_gi]:
                    cls_correct += 1
                cls_total += 1
                for k in range(min(gt_kps.shape[1], pk.shape[1])):
                    if gt_kps[best_gi, k, 2] >= 1:
                        kp_total_gt += 1
                        if pk[pi, k, 2] > 0.5:
                            err = float(np.linalg.norm(
                                pk[pi, k, :2] - gt_kps[best_gi, k, :2]))
                            kp_errors.append(err)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    rec    = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    cls_acc  = cls_correct / cls_total  if cls_total  else 0.0
    pck5  = float(np.mean(np.array(kp_errors) <= 5))  if kp_errors else 0.0
    pck10 = float(np.mean(np.array(kp_errors) <= 10)) if kp_errors else 0.0
    kp_mean   = float(np.mean(kp_errors))   if kp_errors else 0.0
    kp_median = float(np.median(kp_errors)) if kp_errors else 0.0

    return {
        "precision":   round(prec,     4),
        "recall":      round(rec,      4),
        "f1":          round(f1,       4),
        "mean_iou":    round(mean_iou, 4),
        "cls_accuracy":round(cls_acc,  4),
        "kp_mean_err": round(kp_mean,  3),
        "kp_median_err":round(kp_median,3),
        "pck5":        round(pck5,     4),
        "pck10":       round(pck10,    4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_pipeline(pipeline_name: str, args, device,
                   pipeline_index: int = 1, total_pipelines: int = 18,
                   completed_labels: list = None):
    """Train one pipeline variant.  Returns metrics dict."""
    print(f"\n{'='*64}")
    print(f"  Pipeline: {pipeline_label(pipeline_name)}  ({pipeline_name})")
    print(f"{'='*64}")

    ckpt_dir = CKPT_DIR / pipeline_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "krcnn_best.pt"

    # ── Datasets ─────────────────────────────────────────────────────
    data     = _load_coco(config.ANNOTATION_FILE)
    all_ids  = [img["id"] for img in data["images"]]
    train_ids, val_ids, test_ids = _split_image_ids(all_ids, seed=config.RANDOM_SEED)

    train_ds = PipelineBeeDataset(pipeline_name, config.ANNOTATION_FILE,
                                   config.DATA_ROOT, train_ids, augment=True)
    val_ds   = PipelineBeeDataset(pipeline_name, config.ANNOTATION_FILE,
                                   config.DATA_ROOT, val_ids,   augment=False)
    test_ds  = PipelineBeeDataset(pipeline_name, config.ANNOTATION_FILE,
                                   config.DATA_ROOT, test_ids,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── Model ────────────────────────────────────────────────────────
    model = build_krcnn(num_classes=NUM_CLASSES, num_keypoints=NUM_KEYPOINTS,
                        min_size=256)
    model.to(device)
    # Freeze backbone (same strategy as main training)
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

    # ── Optimiser ────────────────────────────────────────────────────
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Training loop ─────────────────────────────────────────────────
    best_val   = float("inf")
    patience   = 0
    best_epoch = 0
    history    = {"train_loss": [], "val_loss": []}
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0; n = 0
        for images, targets in train_loader:
            imgs = [img.to(device) for img in images]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item(); n += 1

        train_loss = total_loss / max(n, 1)

        # Validate
        model.train()
        vtotal = 0.0; vn = 0
        with torch.no_grad():
            for images, targets in val_loader:
                imgs = [img.to(device) for img in images]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
                vloss = sum(model(imgs, tgts).values()).item()
                vtotal += vloss; vn += 1
        val_loss = vtotal / max(vn, 1)
        model.eval()

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch_s = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        epochs_left = args.epochs - epoch
        eta_s = int(avg_epoch_s * epochs_left)

        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.0f}s")

        new_best = val_loss < best_val
        if new_best:
            best_val   = val_loss
            best_epoch = epoch
            patience   = 0
            torch.save({
                "model_state":   model.state_dict(),
                "epoch":         epoch,
                "val_loss":      val_loss,
                "num_classes":   NUM_CLASSES,
                "num_keypoints": NUM_KEYPOINTS,
                "min_size":      256,
                "pipeline":      pipeline_name,
            }, ckpt_path)
            print(f"    ✓ Saved  (val={val_loss:.4f})")
        else:
            patience += 1

        # ── Write live status ─────────────────────────────────────────
        _write_status({
            "status":           "training",
            "pipeline":         pipeline_name,
            "pipeline_label":   pipeline_label(pipeline_name),
            "pipeline_index":   pipeline_index,
            "total_pipelines":  total_pipelines,
            "overall_progress": round((pipeline_index - 1 + epoch / args.epochs)
                                      / total_pipelines * 100, 1),
            "epoch":            epoch,
            "total_epochs":     args.epochs,
            "train_loss":       round(train_loss, 4),
            "val_loss":         round(val_loss,   4),
            "best_val_loss":    round(best_val,   4),
            "best_epoch":       best_epoch,
            "patience":         patience,
            "patience_limit":   args.patience,
            "new_best":         new_best,
            "eta_s":            eta_s,
            "completed":        completed_labels or [],
        })

        if patience >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Evaluate ─────────────────────────────────────────────────────
    print("  Evaluating on test set …")
    # Reload best checkpoint
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_test(model, test_ds, device)

    result = {
        "pipeline":      pipeline_name,
        "label":         pipeline_label(pipeline_name),
        "best_epoch":    best_epoch,
        "best_val_loss": round(best_val, 4),
        "history":       history,
        "metrics":       metrics,
    }
    print(f"  F1={metrics['f1']:.3f}  "
          f"Recall={metrics['recall']:.3f}  "
          f"PCK@10={metrics['pck10']:.3f}  "
          f"KP err={metrics['kp_mean_err']:.2f}px")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, default=None,
                        help="Run a single named pipeline (default: all 18)")
    parser.add_argument("--epochs",   type=int,   default=40)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--lr",       type=float, default=5e-4)
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--resume",   action="store_true",
                        help="Skip pipelines that already have a result entry")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # Load existing results if resuming
    all_results: dict = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        all_results = {r["pipeline"]: r for r in saved}
        print(f"Loaded {len(all_results)} existing results.")

    # Which pipelines to run
    if args.pipeline:
        if args.pipeline not in PIPELINE_NAMES:
            print(f"Unknown pipeline '{args.pipeline}'. "
                  f"Options: {PIPELINE_NAMES}")
            sys.exit(1)
        todo = [args.pipeline]
    else:
        todo = PIPELINE_NAMES

    if args.resume:
        todo = [p for p in todo if p not in all_results]
        print(f"Resuming: {len(todo)} pipeline(s) remaining.")

    total     = len(todo)
    completed = [all_results[p]["label"] for p in PIPELINE_NAMES if p in all_results]

    for i, name in enumerate(todo, 1):
        print(f"\n[{i}/{total}] Starting {name} …")
        t_start = time.time()
        result  = train_pipeline(name, args, device,
                                 pipeline_index=len(all_results) + 1,
                                 total_pipelines=18,
                                 completed_labels=completed)
        result["wall_time_s"] = round(time.time() - t_start)
        all_results[name] = result
        completed.append(result["label"])

        # Save after every pipeline so partial runs are preserved
        with open(RESULTS_PATH, "w") as f:
            json.dump(list(all_results.values()), f, indent=2)
        print(f"  Saved → {RESULTS_PATH}")

    _write_status({
        "status":          "done",
        "total_pipelines": 18,
        "completed":       completed,
        "overall_progress": 100,
    })

    # ── Final summary table ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  {'Pipeline':<35} {'F1':>6} {'Rec':>6} {'Prec':>6} "
          f"{'Cls':>6} {'PCK10':>6} {'KPErr':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    rows = sorted(all_results.values(), key=lambda r: -r["metrics"]["f1"])
    for r in rows:
        m = r["metrics"]
        print(f"  {r['label']:<35} {m['f1']:>6.3f} {m['recall']:>6.3f} "
              f"{m['precision']:>6.3f} {m['cls_accuracy']:>6.3f} "
              f"{m['pck10']:>6.3f} {m['kp_mean_err']:>7.2f}px")
    print(f"{'='*80}")
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
