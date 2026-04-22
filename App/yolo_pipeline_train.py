"""
yolo_pipeline_train.py
----------------------
Train YOLOv8-pose on all 18 preprocessing pipelines and collect metrics.

For each pipeline:
  1. Generates preprocessed images into a temp YOLO dataset
  2. Trains YOLOv8-pose from the same pretrained weights
  3. Evaluates on the shared test split
  4. Records metrics to outputs/pipeline_results.json
  5. Writes live status to outputs/pipeline_status.json (polled by the app)

Usage:
    python3 yolo_pipeline_train.py [--epochs 50] [--pipelines original_bilateral ...]
    python3 yolo_pipeline_train.py --resume          # skip already-done pipelines

Expect ~3-6 min per pipeline on MPS — 18 pipelines ≈ 1-2 hours.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from pipeline_transforms import PIPELINE_NAMES, apply_pipeline, pipeline_label

# ── Constants ─────────────────────────────────────────────────────────────────
YAML_TEMPLATE  = Path("bee_pose.yaml")
YOLO_DATASET   = Path("datasets/bee_yolo")
OUTPUT_DIR     = config.OUTPUT_DIR
RESULTS_FILE   = OUTPUT_DIR / "yolo_pipeline_results.json"
STATUS_FILE    = OUTPUT_DIR / "pipeline_status.json"
RUNS_DIR       = OUTPUT_DIR / "yolo_pipeline_runs"

OUTPUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


# ── Status writer ─────────────────────────────────────────────────────────────

def _write_status(data: dict):
    """Atomically write status JSON (temp-file rename)."""
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(STATUS_FILE)


# ── Dataset preparation ───────────────────────────────────────────────────────

def build_pipeline_dataset(pipeline_name: str, tmp_root: Path):
    """
    Copy images from the YOLO dataset through the named pipeline into tmp_root,
    then symlink label dirs from the original dataset.

    tmp_root/
        images/{train,val,test}/   ← preprocessed PNGs
        labels/{train,val,test}/   ← symlinks to original label dirs
    """
    for split in ("train", "val", "test"):
        # Preprocessed image dir
        out_img_dir = tmp_root / "images" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = YOLO_DATASET / "images" / split
        for img_path in sorted(src_img_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out_rgb = apply_pipeline(img_rgb, pipeline_name)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_dir / img_path.name), out_bgr)

        # Symlink label dir (labels are coordinate-based — same for all pipelines)
        lbl_link = tmp_root / "labels" / split
        lbl_link.parent.mkdir(parents=True, exist_ok=True)
        if not lbl_link.exists():
            src_lbl = (YOLO_DATASET / "labels" / split).resolve()
            os.symlink(src_lbl, lbl_link)


def write_pipeline_yaml(tmp_root: Path) -> Path:
    """Write a bee_pose.yaml pointing to tmp_root."""
    original = YAML_TEMPLATE.read_text()
    # Replace the path line
    lines = []
    for line in original.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {tmp_root.resolve()}")
        else:
            lines.append(line)
    yaml_path = tmp_root / "bee_pose.yaml"
    yaml_path.write_text("\n".join(lines))
    return yaml_path


# ── Metrics extraction ────────────────────────────────────────────────────────

def extract_metrics(results_csv: Path) -> dict:
    """Read last row of Ultralytics results.csv and return a flat metrics dict."""
    if not results_csv.exists():
        return {}
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    # Standardise keys we care about
    out = {}
    for k, v in last.items():
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            out[k] = v
    return out


# ── Single pipeline run ───────────────────────────────────────────────────────

def train_pipeline(pipeline_name: str, args, device: str,
                   pipeline_index: int, total: int,
                   completed: list) -> dict:
    """Preprocess → train → evaluate one pipeline. Returns metrics dict."""
    label = pipeline_label(pipeline_name)
    run_name = f"pipeline_{pipeline_name}"
    run_dir  = (RUNS_DIR / run_name).resolve()

    print(f"\n{'='*60}")
    print(f"[{pipeline_index}/{total}] {label}  ({pipeline_name})")
    print(f"{'='*60}")

    _write_status({
        "status": "preprocessing",
        "pipeline": pipeline_name,
        "pipeline_label": label,
        "pipeline_index": pipeline_index,
        "total_pipelines": total,
        "overall_progress": round((pipeline_index - 1) / total * 100, 1),
        "epoch": 0,
        "total_epochs": args.epochs,
        "completed": completed,
    })

    # Build pipeline dataset in a temporary directory
    with tempfile.TemporaryDirectory(prefix=f"bee_yolo_{pipeline_name}_") as tmp_str:
        tmp_root = Path(tmp_str)
        t_prep = time.time()
        build_pipeline_dataset(pipeline_name, tmp_root)
        yaml_path = write_pipeline_yaml(tmp_root)
        print(f"  Preprocessing done in {time.time()-t_prep:.1f}s")

        _write_status({
            "status": "training",
            "pipeline": pipeline_name,
            "pipeline_label": label,
            "pipeline_index": pipeline_index,
            "total_pipelines": total,
            "overall_progress": round((pipeline_index - 1) / total * 100, 1),
            "epoch": 0,
            "total_epochs": args.epochs,
            "completed": completed,
        })

        model = YOLO(args.base_model)
        t_train = time.time()
        _epoch_state = {"epoch": 0, "t0": t_train}

        def on_epoch_end(trainer):
            ep = trainer.epoch + 1
            total_ep = trainer.epochs
            metrics_now = trainer.metrics or {}
            box_map50 = float(metrics_now.get("metrics/mAP50(B)", 0) or 0)
            pose_map50 = float(metrics_now.get("metrics/mAP50(P)", 0) or 0)
            elapsed = time.time() - _epoch_state["t0"]
            eta_s = int(elapsed / ep * (total_ep - ep)) if ep > 0 else 0
            _write_status({
                "status": "training",
                "pipeline": pipeline_name,
                "pipeline_label": label,
                "pipeline_index": pipeline_index,
                "total_pipelines": total,
                "overall_progress": round((pipeline_index - 1 + ep / total_ep) / total * 100, 1),
                "epoch": ep,
                "total_epochs": total_ep,
                "box_mAP50": round(box_map50, 4),
                "pose_mAP50": round(pose_map50, 4),
                "eta_s": eta_s,
                "completed": completed,
            })

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        train_results = model.train(
            data        = str(yaml_path.resolve()),
            epochs      = args.epochs,
            imgsz       = args.imgsz,
            batch       = args.batch,
            device      = device,
            workers     = 0,          # safer with temp dirs on macOS
            cache       = False,        # ram cache causes OOM on MPS unified memory (MacBook Air)
            patience    = 0,             # EarlyStopping disabled — val metrics unreliable on MPS
            val         = False,         # disable per-epoch val — NMS timeout on MPS corrupts metrics
            project     = str(run_dir.resolve()),
            name        = "run",
            exist_ok    = True,
            verbose     = False,
            amp         = False,      # AMP causes float16 underflow on MPS (Apple Silicon)
            # Mild augmentation
            fliplr      = 0.5,
            flipud      = 0.0,
            degrees     = 10.0,
            translate   = 0.1,
            scale       = 0.3,
            hsv_h       = 0.0,
            hsv_s       = 0.0,
            hsv_v       = 0.3,
            mosaic      = 0.5,
        )
        train_elapsed = time.time() - t_train
        print(f"  Training done in {train_elapsed/60:.1f} min")

        # Extract last-epoch metrics
        results_csv = run_dir / "run" / "results.csv"
        metrics = extract_metrics(results_csv)
        metrics["train_time_s"] = round(train_elapsed, 1)

        # Evaluate on test split — use last.pt (val=False → no best.pt)
        last_weights = run_dir / "run" / "weights" / "last.pt"
        best_weights = run_dir / "run" / "weights" / "best.pt"
        eval_weights = last_weights if last_weights.exists() else best_weights
        if eval_weights.exists():
            _write_status({
                "status": "evaluating",
                "pipeline": pipeline_name,
                "pipeline_label": label,
                "pipeline_index": pipeline_index,
                "total_pipelines": total,
                "overall_progress": round(pipeline_index / total * 100, 1),
                "epoch": args.epochs,
                "total_epochs": args.epochs,
                "completed": completed,
            })
            eval_model = YOLO(str(eval_weights))
            val_res = eval_model.val(
                data    = str(yaml_path.resolve()),
                split   = "test",
                device  = device,
                batch   = 8,
                verbose = False,
            )
            # Pull numeric results from the validator
            try:
                box = val_res.box
                pose = val_res.pose
                metrics["test_box_mAP50"]    = round(float(box.map50),  4)
                metrics["test_box_mAP50_95"] = round(float(box.map),    4)
                metrics["test_pose_mAP50"]   = round(float(pose.map50), 4)
                metrics["test_pose_mAP50_95"]= round(float(pose.map),   4)
            except Exception as e:
                print(f"  Warning: could not extract test metrics: {e}")

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",  default="yolov8n-pose.pt")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--imgsz",       type=int, default=448)
    parser.add_argument("--batch",       type=int, default=16)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--device",      default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--pipelines",   nargs="*", default=None,
                        help="Subset of pipeline names to run (default: all 18)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip pipelines already present in pipeline_results.json")
    args = parser.parse_args()

    if not YAML_TEMPLATE.exists():
        raise FileNotFoundError("bee_pose.yaml not found in App/")

    pipelines = args.pipelines or PIPELINE_NAMES

    # Load existing results for resume
    existing: dict = {}
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text())
        except json.JSONDecodeError:
            existing = {}

    if args.resume:
        pipelines = [p for p in pipelines if p not in existing]
        print(f"Resuming: {len(pipelines)} pipelines remaining "
              f"({len(existing)} already done)")

    total = len(pipelines)
    if total == 0:
        print("Nothing to do.")
        _write_status({"status": "done", "completed": list(existing.keys())})
        return

    print(f"Device     : {args.device}")
    print(f"Base model : {args.base_model}")
    print(f"Pipelines  : {total}  |  epochs: {args.epochs}  |  imgsz: {args.imgsz}")

    results = dict(existing)
    completed_labels = [pipeline_label(p) for p in existing]

    t_total = time.time()
    for i, pipeline_name in enumerate(pipelines, start=len(existing) + 1):
        metrics = train_pipeline(
            pipeline_name, args, args.device,
            pipeline_index=i, total=len(existing) + total,
            completed=completed_labels,
        )
        results[pipeline_name] = {
            "label":   pipeline_label(pipeline_name),
            "metrics": metrics,
        }
        completed_labels.append(pipeline_label(pipeline_name))

        # Persist results after every pipeline
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"  Saved → {RESULTS_FILE}")

    total_elapsed = time.time() - t_total
    print(f"\nAll {total} pipelines done in {total_elapsed/3600:.2f} h")

    _write_status({
        "status": "done",
        "completed": completed_labels,
        "total_elapsed_s": round(total_elapsed, 1),
    })
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
