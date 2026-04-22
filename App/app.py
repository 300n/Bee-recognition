"""
app.py — Flask dashboard for YOLOv8-pose bee keypoint detection.
Run: python app.py
Then open http://localhost:5001
"""

import base64
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)


# ── YOLO model loading ───────────────────────────────────────────────────────

def _find_best_yolo_weights() -> Path | None:
    """Return best.pt from the top-scoring pipeline run, or None."""
    results_file = config.OUTPUT_DIR / "yolo_pipeline_results.json"
    if results_file.exists():
        try:
            results = json.loads(results_file.read_text())
            if results:
                best_name = max(
                    results,
                    key=lambda k: results[k].get("metrics", {}).get("test_pose_mAP50", 0)
                )
                run_weights = (config.OUTPUT_DIR / "yolo_pipeline_runs"
                               / f"pipeline_{best_name}" / "run" / "weights")
                w = run_weights / "last.pt"
                if not w.exists():
                    w = run_weights / "best.pt"
                if w.exists():
                    print(f"  ✓ YOLOv8-pose best weights: pipeline={best_name}  path={w}")
                    return w
        except Exception as e:
            print(f"  Warning: could not load pipeline results: {e}")
    return None


def _load_yolo():
    from ultralytics import YOLO as _YOLO
    w = _find_best_yolo_weights()
    if w is None:
        print("  No trained weights found — YOLO inference will be unavailable.")
        return None
    return _YOLO(str(w))


print(f"Loading models on {DEVICE} …")
_yolo_model = _load_yolo()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encode_annotated(img_bgr: np.ndarray, max_side: int = 1600) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale   = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB


@app.route("/")
def index():
    return render_template("index.html", device=str(DEVICE))


# ── YOLO inference ───────────────────────────────────────────────────────────

@app.route("/predict_yolo", methods=["POST"])
def predict_yolo_route():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if _yolo_model is None:
        return jsonify({"error": "YOLO model not trained yet. Run: python3 yolo_pipeline_train.py"}), 503

    file    = request.files["image"]
    score_t = float(request.form.get("score", 0.4))

    pil_img = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(pil_img)
    H, W    = img_rgb.shape[:2]

    results = _yolo_model.predict(img_rgb, conf=score_t, verbose=False)
    r = results[0]

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()              if r.boxes     is not None else []
    confs      = r.boxes.conf.cpu().numpy()              if r.boxes     is not None else []
    clss       = r.boxes.cls.cpu().numpy().astype(int)   if r.boxes     is not None else []
    kps_xy     = r.keypoints.xy.cpu().numpy()            if r.keypoints is not None else []
    kps_conf   = r.keypoints.conf.cpu().numpy()          if (r.keypoints is not None
                                                            and r.keypoints.conf is not None) else None

    detections = []
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    KP_COLS_BGR = [(80, 80, 255), (30, 180, 255), (255, 180, 80)]

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        cls_idx = int(clss[i])
        kps = []
        for k in range(len(kps_xy[i])):
            kpx, kpy = float(kps_xy[i][k][0]), float(kps_xy[i][k][1])
            vis = float(kps_conf[i][k]) if kps_conf is not None else 1.0
            kps.append({"x": kpx, "y": kpy, "vis": vis, "visible": vis > 0.5})
        detections.append({
            "class_name": config.CLASS_NAMES[cls_idx] if cls_idx < len(config.CLASS_NAMES) else str(cls_idx),
            "class_idx":  cls_idx,
            "confidence": float(confs[i]),
            "bbox":       [float(x1), float(y1), float(x2), float(y2)],
            "keypoints":  kps,
        })

        color = (255, 180, 50) if cls_idx == 0 else (80, 200, 80)
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{config.CLASS_NAMES[cls_idx] if cls_idx < len(config.CLASS_NAMES) else cls_idx} {float(confs[i]):.0%}"
        cv2.putText(img_bgr, label, (int(x1), max(int(y1) - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for k, kp in enumerate(kps):
            if kp["visible"]:
                px, py = int(kp["x"]), int(kp["y"])
                col_bgr = KP_COLS_BGR[k % len(KP_COLS_BGR)]
                cv2.circle(img_bgr, (px, py), 5, (0, 0, 0), -1)
                cv2.circle(img_bgr, (px, py), 4, col_bgr,   -1)

    return jsonify({
        "mode":            "yolo",
        "detections":      detections,
        "annotated_image": _encode_annotated(img_bgr),
        "image_size":      [W, H],
        "n_detections":    len(detections),
    })


# ── Behavioural analysis ─────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    try:
        detections = json.loads(request.form.get("detections", "[]"))
    except Exception:
        return jsonify({"error": "Invalid detections JSON"}), 400

    file    = request.files["image"]
    pil_img = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(pil_img)

    try:
        import bee_analysis
        result = bee_analysis.analyse(img_rgb, detections)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


# ── Pipeline training status & results ───────────────────────────────────────

@app.route("/pipeline_status")
def pipeline_status():
    spath = config.OUTPUT_DIR / "pipeline_status.json"
    if not spath.exists():
        return jsonify({"status": "idle"})
    with open(spath) as f:
        return jsonify(json.load(f))


@app.route("/yolo_pipeline_results")
def yolo_pipeline_results():
    rpath = config.OUTPUT_DIR / "yolo_pipeline_results.json"
    if not rpath.exists():
        return jsonify({"results": {}, "status": "not_started"})
    with open(rpath) as f:
        results = json.load(f)
    return jsonify({"results": results, "status": "ok"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n  → Open http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
