"""
yolo_train.py
-------------
Train a single YOLOv8-pose model on the bee keypoint dataset.
Run yolo_convert.py first to generate datasets/bee_yolo/ and bee_pose.yaml.

Usage:
    python3 yolo_train.py [--epochs 50] [--model yolov8n-pose.pt] [--device mps]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

import config

YAML_PATH  = Path("bee_pose.yaml")
OUTPUT_DIR = config.OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8n-pose.pt",
                        help="YOLOv8 pose model weights (downloaded automatically)")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--imgsz",   type=int,   default=448)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--device",  default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--workers", type=int,   default=4)
    parser.add_argument("--patience",type=int,   default=20)
    args = parser.parse_args()

    if not YAML_PATH.exists():
        raise FileNotFoundError("bee_pose.yaml not found — run yolo_convert.py first")

    print(f"Device : {args.device}")
    print(f"Model  : {args.model}")
    print(f"Epochs : {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}")

    model = YOLO(args.model)

    t0 = time.time()
    results = model.train(
        data        = str(YAML_PATH.resolve()),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        workers     = args.workers,
        patience    = args.patience,
        project     = str((OUTPUT_DIR / "yolo_runs").resolve()),
        name        = "bee_baseline",
        exist_ok    = True,
        verbose     = True,
        # Augmentation — keep mild to preserve keypoint validity
        fliplr      = 0.5,
        flipud      = 0.0,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.3,
        hsv_h       = 0.015,
        hsv_s       = 0.4,
        hsv_v       = 0.3,
        mosaic      = 0.5,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save summary metrics
    run_dir = (OUTPUT_DIR / "yolo_runs" / "bee_baseline").resolve()
    metrics_file = run_dir / "results.csv"
    summary = {
        "model":   args.model,
        "epochs":  args.epochs,
        "elapsed_s": round(elapsed, 1),
    }
    if metrics_file.exists():
        import csv
        with open(metrics_file) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            summary["final_metrics"] = {k.strip(): v.strip() for k, v in last.items()}

    out_path = OUTPUT_DIR / "yolo_baseline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {out_path}")

    # Validate on test set
    print("\nEvaluating on test split …")
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        m = YOLO(str(best_weights))
        val_results = m.val(data=str(YAML_PATH.resolve()), split="test",
                            device=args.device, verbose=True)
        print(val_results)


if __name__ == "__main__":
    main()
