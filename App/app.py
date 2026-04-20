"""
app.py  –  Flask web app for bee keypoint prediction
-----------------------------------------------------
Two prediction modes:

  1. Whole-image mode  – treats the uploaded image as a single bee crop
     (best for zoomed-in / already-cropped bee images)

  2. Grid-scan mode    – slides a window over the image, clusters nearby
     keypoints and renders all detections on the original image
     (best for full hive-camera frames like the training images)

Run:
    python app.py
Then open http://localhost:5000
"""

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as tv_ops
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from models import build_model

# ── Global model (loaded once at startup) ─────────────────────────────────
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
NORM_MEAN = np.array(config.NORM_MEAN, dtype=np.float32)
NORM_STD  = np.array(config.NORM_STD,  dtype=np.float32)

def _load_resnet18():
    from models import build_model
    m = build_model("resnet18", config.NUM_CLASSES, config.MAX_KEYPOINTS)
    ckpt = torch.load(config.OUTPUT_DIR / "resnet18_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state"])
    if hasattr(m, "unfreeze_backbone"):
        m.unfreeze_backbone()
    m.to(DEVICE).eval()
    print(f"  ✓ ResNet18  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return m, ckpt

def _load_krcnn():
    from krcnn_train import build_krcnn
    ckpt = torch.load(config.OUTPUT_DIR / "krcnn_best.pt", map_location=DEVICE)
    m = build_krcnn(num_classes=ckpt.get("num_classes", 3),
                    num_keypoints=ckpt.get("num_keypoints", 3))
    m.load_state_dict(ckpt["model_state"])
    m.to(DEVICE).eval()
    print(f"  ✓ KeypointRCNN  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return m, ckpt

# ── YOLO model ─────────────────────────────────────────────────────────────
_YOLO_WEIGHTS = (config.OUTPUT_DIR / "yolo_runs" / "bee_baseline" / "weights" / "best.pt")

def _load_yolo():
    from ultralytics import YOLO as _YOLO
    m = _YOLO(str(_YOLO_WEIGHTS))
    print(f"  ✓ YOLOv8-pose  weights={_YOLO_WEIGHTS}")
    return m

_yolo_available = _YOLO_WEIGHTS.exists()

# Load whichever checkpoint exists (prefer KRCNN)
_krcnn_available  = (config.OUTPUT_DIR / "krcnn_best.pt").exists()
_resnet_available = (config.OUTPUT_DIR / "resnet18_best.pt").exists()

print(f"Loading models on {DEVICE} …")
_krcnn_model,  _krcnn_ckpt  = (_load_krcnn()   if _krcnn_available  else (None, None))
_resnet_model, _resnet_ckpt = (_load_resnet18() if _resnet_available else (None, None))
_yolo_model                 = (_load_yolo()     if _yolo_available   else None)
_ckpt  = _krcnn_ckpt  or _resnet_ckpt
_model = _krcnn_model or _resnet_model

# ── Inference helpers ──────────────────────────────────────────────────────

def _preprocess(crop_rgb: np.ndarray) -> torch.Tensor:
    """Resize, normalise, return (1,3,H,W) tensor."""
    img = cv2.resize(crop_rgb, (config.CROP_SIZE, config.CROP_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def _infer(tensor: torch.Tensor):
    """Return (class_idx, confidence, kp_norm (K,2), vis_prob (K,))."""
    pred_cls, pred_kp, pred_vis = _model(tensor)
    probs      = torch.softmax(pred_cls[0], dim=0).cpu().numpy()
    class_idx  = int(probs.argmax())
    confidence = float(probs[class_idx])
    kp_norm    = pred_kp[0].cpu().numpy()          # (K, 2)
    vis_prob   = torch.sigmoid(pred_vis[0]).cpu().numpy()  # (K,)
    return class_idx, confidence, kp_norm, vis_prob


# ── Mode 1: single crop ────────────────────────────────────────────────────

def predict_single(img_rgb: np.ndarray) -> dict:
    """Run the model on the whole image as one bee crop."""
    h, w = img_rgb.shape[:2]
    tensor = _preprocess(img_rgb)
    class_idx, confidence, kp_norm, vis_prob = _infer(tensor)

    keypoints = []
    for i in range(config.MAX_KEYPOINTS):
        keypoints.append({
            "x":       float(kp_norm[i, 0] * w),
            "y":       float(kp_norm[i, 1] * h),
            "vis":     float(vis_prob[i]),
            "visible": bool(vis_prob[i] >= 0.5),
        })

    return {
        "mode":       "single",
        "class_name": config.CLASS_NAMES[class_idx],
        "class_idx":  class_idx,
        "confidence": confidence,
        "keypoints":  keypoints,
        "bbox":       {"x": 0, "y": 0, "w": w, "h": h},
    }


# ── Mode 2: sliding-window grid scan ──────────────────────────────────────

def predict_grid(img_rgb: np.ndarray, stride_ratio=0.4, scales=(1.0, 0.7, 0.5)) -> dict:
    """
    Slide windows at multiple scales over the image.
    Accept patches whose predicted keypoints have mean visibility ≥ threshold.
    Cluster nearby keypoints with a simple distance-based merge.
    """
    h, w = img_rgb.shape[:2]
    VIS_THRESH   = 0.60   # min mean visibility to accept a patch
    MERGE_DIST   = 0.12   # merge keypoints within 12% of image diagonal
    diag = np.sqrt(h**2 + w**2)

    candidates = []   # list of {"kp_abs": (K,2), "vis": (K,), "cls": int, "conf": float}

    for scale in scales:
        win = int(min(h, w) * scale)
        if win < 32:
            continue
        stride = max(1, int(win * stride_ratio))
        for y0 in range(0, h - win + 1, stride):
            for x0 in range(0, w - win + 1, stride):
                patch = img_rgb[y0:y0+win, x0:x0+win]
                tensor = _preprocess(patch)
                class_idx, confidence, kp_norm, vis_prob = _infer(tensor)

                mean_vis = float(vis_prob[vis_prob >= 0.5].mean()) if (vis_prob >= 0.5).any() else 0.0
                if mean_vis < VIS_THRESH:
                    continue

                # Convert to absolute image coordinates
                kp_abs = kp_norm.copy()
                kp_abs[:, 0] = kp_norm[:, 0] * win + x0
                kp_abs[:, 1] = kp_norm[:, 1] * win + y0

                candidates.append({
                    "kp_abs": kp_abs,
                    "vis":    vis_prob,
                    "cls":    class_idx,
                    "conf":   confidence,
                })

    if not candidates:
        # Fall back to single-crop if no confident patches found
        return predict_single(img_rgb)

    # ── Greedy merge: cluster by centroid distance ───────────────────────
    def centroid(c):
        vis_mask = c["vis"] >= 0.5
        if vis_mask.any():
            return c["kp_abs"][vis_mask].mean(axis=0)
        return c["kp_abs"].mean(axis=0)

    merged = []
    used   = [False] * len(candidates)
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        group = [c]
        used[i] = True
        ci = centroid(c)
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            if np.linalg.norm(centroid(candidates[j]) - ci) < MERGE_DIST * diag:
                group.append(candidates[j])
                used[j] = True
        # Best candidate = highest mean visibility
        best = max(group, key=lambda x: x["vis"].mean())
        merged.append(best)

    detections = []
    for det in merged:
        kps = []
        for i in range(config.MAX_KEYPOINTS):
            kps.append({
                "x":       float(det["kp_abs"][i, 0]),
                "y":       float(det["kp_abs"][i, 1]),
                "vis":     float(det["vis"][i]),
                "visible": bool(det["vis"][i] >= 0.5),
            })
        detections.append({
            "class_name": config.CLASS_NAMES[det["cls"]],
            "class_idx":  det["cls"],
            "confidence": det["conf"],
            "keypoints":  kps,
        })

    return {"mode": "grid", "detections": detections}


# ── Tiled KRCNN inference ──────────────────────────────────────────────────
# Sliding-window inference for large images.
#
# Strategy — centre-bias + 50 % overlap:
#   Each 512px tile is slid with a 256px stride (50 % overlap).
#   After inference, a detection is only *kept* from a tile if its bounding-box
#   centre falls inside the tile's "safe zone" — a 48px-inset inner rectangle
#   that excludes the edges.  Boundary tiles extend their safe zone to the
#   image edge so no bee is ever discarded there.
#
#   Why this works: with 50 % overlap every point in the image is always at
#   least 128px from the edge of one tile, so its centre will comfortably land
#   in that tile's safe zone.  A bee straddling the seam of two tiles has its
#   centre claimed by exactly one tile (the one in whose safe zone it falls),
#   and that tile contains the bee almost entirely — so confidence is high.
#
#   A final light NMS (IoU ≥ 0.30) cleans up the rare case where a bee centre
#   sits on the safe-zone boundary of two adjacent tiles.
_TILE_SIZE    = 512   # px — tile side length (good scale match for ~54 px bees)
_TILE_OVERLAP = 256   # px — 50 % overlap so every bee is fully inside ≥1 tile
_TILE_MARGIN  = 48    # px — safe-zone inset from each tile edge


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Return start positions that fully cover [0, total] with no partial tiles."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile, stride))
    if not starts or starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


@torch.no_grad()
def _krcnn_infer_tiled(img_rgb: np.ndarray, score_t: float) -> tuple[dict, int]:
    """
    Run KRCNN on img_rgb with automatic sliding-window tiling for large images.

    For images that fit in a single tile: one forward pass, no centre-bias needed.
    For larger images: 50 % overlapping tiles + centre-bias filtering per tile
    + final NMS to remove residual duplicates at safe-zone boundaries.

    Returns (preds_dict, n_tiles) where preds_dict contains CPU tensors with
    all coordinates in the original image space.
    """
    H, W = img_rgb.shape[:2]
    stride = _TILE_SIZE - _TILE_OVERLAP   # 256 px

    y_starts = _tile_starts(H, _TILE_SIZE, stride)
    x_starts = _tile_starts(W, _TILE_SIZE, stride)
    n_tiles  = len(y_starts) * len(x_starts)

    # ── Single-pass (image fits in one tile) ──────────────────────────
    if n_tiles == 1:
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        preds = _krcnn_model([img_t.to(DEVICE)])[0]
        return {k: v.cpu() for k, v in preds.items()}, 1

    # ── Sliding-window pass ───────────────────────────────────────────
    all_boxes, all_labels, all_scores, all_kps = [], [], [], []

    for y0 in y_starts:
        for x0 in x_starts:
            # Tiles are always exactly TILE_SIZE (ensured by _tile_starts)
            tile = img_rgb[y0:y0 + _TILE_SIZE, x0:x0 + _TILE_SIZE]

            img_t = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1)
            preds = _krcnn_model([img_t.to(DEVICE)])[0]

            scores = preds["scores"].cpu()
            keep   = scores >= score_t
            if not keep.any():
                continue

            boxes  = preds["boxes"][keep].cpu()     # (N,4) in tile coords
            labels = preds["labels"][keep].cpu()
            scores = scores[keep]
            kps    = preds["keypoints"][keep].cpu() # (N,K,3) in tile coords

            # Shift to original image coordinates
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            kps[:, :, 0]     += x0
            kps[:, :, 1]     += y0

            # ── Centre-bias filter ────────────────────────────────────
            # Compute safe zone for this tile in image coords.
            # Boundary tiles extend their safe zone all the way to the image
            # edge so we never discard a valid detection at the image border.
            sx_min = (x0 + _TILE_MARGIN) if x0 > 0        else 0
            sx_max = (x0 + _TILE_SIZE - _TILE_MARGIN) if (x0 + _TILE_SIZE) < W else W
            sy_min = (y0 + _TILE_MARGIN) if y0 > 0        else 0
            sy_max = (y0 + _TILE_SIZE - _TILE_MARGIN) if (y0 + _TILE_SIZE) < H else H

            # Bbox centre
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5

            in_safe = (cx >= sx_min) & (cx <= sx_max) & \
                      (cy >= sy_min) & (cy <= sy_max)

            if not in_safe.any():
                continue

            all_boxes.append(boxes[in_safe])
            all_labels.append(labels[in_safe])
            all_scores.append(scores[in_safe])
            all_kps.append(kps[in_safe])

    if not all_boxes:
        K = config.MAX_KEYPOINTS
        return {
            "boxes":     torch.zeros((0, 4)),
            "labels":    torch.zeros(0, dtype=torch.int64),
            "scores":    torch.zeros(0),
            "keypoints": torch.zeros((0, K, 3)),
        }, n_tiles

    boxes_all  = torch.cat(all_boxes)
    labels_all = torch.cat(all_labels)
    scores_all = torch.cat(all_scores)
    kps_all    = torch.cat(all_kps)

    # ── Light NMS — catches the rare safe-zone boundary overlap ──────
    keep_idx = tv_ops.nms(boxes_all.float(), scores_all.float(), iou_threshold=0.30)

    return {
        "boxes":     boxes_all[keep_idx],
        "labels":    labels_all[keep_idx],
        "scores":    scores_all[keep_idx],
        "keypoints": kps_all[keep_idx],
    }, n_tiles


def _encode_annotated(img_bgr: np.ndarray, max_side: int = 1600) -> str:
    """Resize if very large, then encode as base64 PNG."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale    = max_side / max(h, w)
        img_bgr  = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()


# ── Video job state ───────────────────────────────────────────────────────
VIDEO_JOB_DIR = config.OUTPUT_DIR / "video_jobs"
VIDEO_JOB_DIR.mkdir(parents=True, exist_ok=True)
_video_jobs: dict = {}   # job_id → state dict


def _purge_old_jobs(max_age_s: int = 3600):
    """Delete job folders older than max_age_s seconds."""
    now = time.time()
    for jid, job in list(_video_jobs.items()):
        if now - job.get("created", now) > max_age_s:
            try:
                import shutil
                shutil.rmtree(str(job["job_dir"]), ignore_errors=True)
            except Exception:
                pass
            _video_jobs.pop(jid, None)


def _make_timeline_plots(stats: list) -> dict:
    """
    Build time-series charts from per-frame stats.
    Returns { chart_name: base64_png }.
    """
    if len(stats) < 2:
        return {}

    times   = [s["time"] for s in stats]
    n_bees  = [s["n_bees"]    for s in stats]
    n_clean = [s["n_cleaning"] for s in stats]
    n_norm  = [s["n_normal"]  for s in stats]
    r_vals  = [s.get("rayleigh_r")  for s in stats]
    h_vals  = [s.get("mean_heading") for s in stats]

    def _b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    plots = {}

    # ── Bee counts over time ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#1a1d27")
    ax.set_facecolor("#1a1d27")
    ax.fill_between(times, n_norm,  color="#4a9eff", alpha=0.6, label="Normal")
    ax.fill_between(times, n_clean, color="#ff9955", alpha=0.8, label="Cleaning")
    ax.plot(times, n_bees, color="white", lw=1.2, label="Total")
    ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
    ax.set_ylabel("Bees detected", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#111", framealpha=0.5)
    ax.set_title("Bee counts over time", color="white", fontsize=10, pad=6)
    plt.tight_layout(pad=0.4)
    plots["bee_counts"] = _b64(fig)

    # ── Rayleigh r over time ─────────────────────────────────
    r_clean = [v for v in r_vals if v is not None]
    if r_clean:
        t_r = [times[i] for i, v in enumerate(r_vals) if v is not None]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        ax.axhline(0.7, color="#4caf72", lw=0.8, ls="--", alpha=0.6, label="strongly ordered")
        ax.axhline(0.4, color="#f5c842", lw=0.8, ls="--", alpha=0.6, label="moderately ordered")
        ax.plot(t_r, r_clean, color="#00e5b0", lw=1.5, label="Rayleigh r")
        ax.fill_between(t_r, r_clean, alpha=0.2, color="#00e5b0")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax.set_ylabel("Rayleigh r", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#444")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#111", framealpha=0.5)
        ax.set_title("Heading coherence (Rayleigh r) over time — 0=random, 1=all aligned",
                     color="white", fontsize=9, pad=6)
        plt.tight_layout(pad=0.4)
        plots["rayleigh"] = _b64(fig)

    # ── Mean heading over time (circular) ───────────────────
    h_clean = [v for v in h_vals if v is not None]
    if h_clean:
        t_h = [times[i] for i, v in enumerate(h_vals) if v is not None]
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        ax.scatter(t_h, h_clean, c=h_clean, cmap="hsv", s=18, alpha=0.85, vmin=-180, vmax=180)
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax.set_ylabel("Mean heading (°)", color="#aaa", fontsize=8)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_yticklabels(["←", "↓", "→", "↑", "←"], fontsize=9, color="#aaa")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#444")
        ax.set_title("Mean heading direction over time", color="white", fontsize=9, pad=6)
        plt.tight_layout(pad=0.4)
        plots["heading"] = _b64(fig)

    # ── Cleaning bee ratio over time ─────────────────────────
    ratio = [nc / nb if nb > 0 else 0 for nc, nb in zip(n_clean, n_bees)]
    if any(r > 0 for r in ratio):
        fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        ax.fill_between(times, ratio, color="#ff9955", alpha=0.7, label="Cleaning bee ratio")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax.set_ylabel("Fraction", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#444")
        ax.set_title("Cleaning bee fraction over time", color="white", fontsize=9, pad=6)
        plt.tight_layout(pad=0.4)
        plots["cleaning_ratio"] = _b64(fig)

    return plots


def _process_video_job(job_id: str, video_path: Path,
                       frame_step: int, score_t: float):
    """Background thread: annotate video frame-by-frame, collect stats."""
    job = _video_jobs[job_id]
    try:
        import bee_analysis

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Down-scale very large frames for output video (keep coords in full-res space)
        OUT_MAX = 1200
        scale   = min(OUT_MAX / max(W, H), 1.0)
        out_W   = int(W * scale)
        out_H   = int(H * scale)

        out_path = video_path.parent / "output.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps  = max(fps / frame_step, 2.0)   # output plays at a comfortable speed
        writer   = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_W, out_H))

        job.update({
            "total_frames": total_frames,
            "fps": fps,
            "video_info": {"W": W, "H": H, "fps": fps, "total_frames": total_frames},
        })

        from krcnn_predict import draw_predictions
        CLASS_NAMES_KRCNN = ["__background__"] + config.CLASS_NAMES

        frame_stats  = []
        key_frames   = []   # base64 thumbnails of representative frames
        KF_EVERY     = max(1, total_frames // frame_step // 8)  # ~8 key frames
        processed    = 0
        frame_idx    = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            preds, _  = _krcnn_infer_tiled(frame_rgb, score_t)

            annotated = draw_predictions(frame_bgr.copy(), preds, score_threshold=score_t)
            if scale < 1.0:
                annotated = cv2.resize(annotated, (out_W, out_H), interpolation=cv2.INTER_AREA)
            writer.write(annotated)

            # ── Per-frame stats ──────────────────────────────────────
            sc   = preds["scores"].numpy()
            lbl  = preds["labels"].numpy()
            kps  = preds["keypoints"].numpy()   # (N, K, 3)
            keep = sc >= score_t

            n_bees    = int(keep.sum())
            n_clean   = int((lbl[keep] == 1).sum()) if n_bees else 0

            stat: dict = {
                "frame":      frame_idx,
                "time":       round(frame_idx / fps, 2),
                "n_bees":     n_bees,
                "n_cleaning": n_clean,
                "n_normal":   n_bees - n_clean,
            }

            if n_bees >= 2:
                kps_kept = kps[keep]
                lbl_kept = lbl[keep]
                dets = []
                for i in range(n_bees):
                    if kps_kept[i, 0, 2] > 0.5 and kps_kept[i, 1, 2] > 0.5:
                        dets.append({
                            "keypoints": [
                                {"x": float(kps_kept[i, 0, 0]),
                                 "y": float(kps_kept[i, 0, 1]), "visible": True},
                                {"x": float(kps_kept[i, 1, 0]),
                                 "y": float(kps_kept[i, 1, 1]), "visible": True},
                            ],
                            "class_name": "cleaning bee" if lbl_kept[i] == 1 else "normal bee",
                        })
                bees = bee_analysis._extract_bees(dets)
                if len(bees) >= 2:
                    _, uv, _, valid = bee_analysis._build_arrays(bees)
                    uv = uv[valid]
                    if len(uv):
                        mx = float(np.mean(uv[:, 0]))
                        my = float(np.mean(uv[:, 1]))
                        stat["rayleigh_r"]   = round((mx**2 + my**2) ** 0.5, 3)
                        stat["mean_heading"] = round(float(np.degrees(np.arctan2(my, mx))), 1)

            frame_stats.append(stat)

            # ── Key frame thumbnail ───────────────────────────────────
            if processed % KF_EVERY == 0 and len(key_frames) < 10:
                thumb = cv2.resize(annotated, (320, int(320 * out_H / out_W)))
                _, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                key_frames.append({
                    "time":   stat["time"],
                    "n_bees": n_bees,
                    "img":    base64.b64encode(enc.tobytes()).decode(),
                })

            processed += 1
            job["current_frame"]   = frame_idx
            job["processed_frames"] = processed
            job["progress"]         = min(99, int(frame_idx / max(total_frames, 1) * 100))
            job["last_n_bees"]      = n_bees
            frame_idx += 1

        cap.release()
        writer.release()

        # ── Summary stats ────────────────────────────────────────────
        avg_bees    = round(float(np.mean([s["n_bees"] for s in frame_stats])), 1) if frame_stats else 0
        max_bees    = max((s["n_bees"] for s in frame_stats), default=0)
        r_vals      = [s["rayleigh_r"] for s in frame_stats if "rayleigh_r" in s]
        avg_r       = round(float(np.mean(r_vals)), 3) if r_vals else None
        total_clean = sum(s["n_cleaning"] for s in frame_stats)
        total_norm  = sum(s["n_normal"]   for s in frame_stats)

        plots = _make_timeline_plots(frame_stats)

        job.update({
            "status":       "done",
            "progress":     100,
            "message":      f"Processed {processed} frames",
            "output_path":  str(out_path),
            "frame_stats":  frame_stats,
            "key_frames":   key_frames,
            "plots":        plots,
            "summary": {
                "processed_frames": processed,
                "duration_s":       round(total_frames / fps, 1),
                "fps":              round(fps, 1),
                "frame_step":       frame_step,
                "original_size":    [W, H],
                "avg_bees":         avg_bees,
                "max_bees":         max_bees,
                "avg_rayleigh_r":   avg_r,
                "total_cleaning_detections": total_clean,
                "total_normal_detections":   total_norm,
            },
        })

    except Exception as exc:
        traceback.print_exc()
        job["status"]  = "error"
        job["message"] = str(exc)


# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 MB — allow large videos


@app.route("/")
def index():
    epoch    = _ckpt["epoch"]    if _ckpt else "—"
    val_loss = f"{_ckpt['val_loss']:.4f}" if _ckpt else "—"
    return render_template("index.html",
                           model_epoch=epoch,
                           val_loss=val_loss,
                           device=str(DEVICE))


@app.route("/predict", methods=["POST"])
def predict():
    """Crop-based ResNet18 endpoint (single bee / grid scan)."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if _resnet_model is None:
        return jsonify({"error": "ResNet18 model not loaded — train it first"}), 503

    file = request.files["image"]
    mode = request.form.get("mode", "single")

    pil_img = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(pil_img)

    result = predict_grid(img_rgb) if mode == "grid" else predict_single(img_rgb)
    annotated = _draw_result(img_rgb.copy(), result)
    _, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf.tobytes()).decode()
    return jsonify({**result, "annotated_image": b64})


@app.route("/predict_krcnn", methods=["POST"])
def predict_krcnn_route():
    """Full-image Keypoint R-CNN endpoint with automatic tiling for large images."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if _krcnn_model is None:
        return jsonify({"error": "Keypoint R-CNN not trained yet. Run: python krcnn_train.py"}), 503

    file    = request.files["image"]
    score_t = float(request.form.get("score", 0.5))

    pil_img = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(pil_img)
    H, W    = img_rgb.shape[:2]

    preds, n_tiles = _krcnn_infer_tiled(img_rgb, score_t)

    # Draw and encode (resize if very large)
    from krcnn_predict import draw_predictions
    CLASS_NAMES_KRCNN = ["__background__"] + config.CLASS_NAMES
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated = draw_predictions(img_bgr, preds, score_threshold=score_t)
    b64       = _encode_annotated(annotated)

    boxes  = preds["boxes"].tolist()
    labels = preds["labels"].tolist()
    scores = preds["scores"].tolist()
    kps    = preds["keypoints"].tolist()

    detections = []
    for i, sc in enumerate(scores):
        if sc < score_t:
            continue
        cls = labels[i]
        detections.append({
            "class_name": CLASS_NAMES_KRCNN[cls] if cls < len(CLASS_NAMES_KRCNN) else str(cls),
            "confidence": sc,
            "bbox":       boxes[i],
            "keypoints":  [{"x": kps[i][k][0], "y": kps[i][k][1],
                            "vis": kps[i][k][2], "visible": kps[i][k][2] > 0.5}
                           for k in range(len(kps[i]))],
        })

    return jsonify({
        "mode":            "krcnn",
        "detections":      detections,
        "annotated_image": b64,
        "image_size":      [W, H],
        "n_tiles":         n_tiles,
        "tile_size":       _TILE_SIZE if n_tiles > 1 else None,
    })


# ── Drawing helper ─────────────────────────────────────────────────────────
KP_COLORS = [
    (255,  80,  80),   # red   – keypoint 0
    (255, 180,  30),   # amber – keypoint 1
    ( 80, 180, 255),   # sky   – keypoint 2
]


def _draw_result(img_rgb: np.ndarray, result: dict) -> np.ndarray:
    out = img_rgb.copy()

    def draw_kp(kps, alpha=1.0):
        for i, kp in enumerate(kps):
            if not kp["visible"]:
                continue
            px, py = int(kp["x"]), int(kp["y"])
            col = KP_COLORS[i % len(KP_COLORS)]
            # Outer ring
            cv2.circle(out, (px, py), 9,  (0, 0, 0), -1)
            cv2.circle(out, (px, py), 8,  col,        -1)
            # Index number
            cv2.putText(out, str(i), (px + 7, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    def draw_label(text, x, y, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 6, y + 2), (30, 30, 30), -1)
        cv2.putText(out, text, (x + 3, y - 2), font, 0.55, color, 1, cv2.LINE_AA)

    if result["mode"] == "single":
        draw_kp(result["keypoints"])
        label = f"{result['class_name']}  {result['confidence']:.0%}"
        draw_label(label, 6, 22)

    else:  # grid
        for det in result.get("detections", []):
            draw_kp(det["keypoints"])
            # Small centroid marker
            vis_kps = [k for k in det["keypoints"] if k["visible"]]
            if vis_kps:
                cx = int(sum(k["x"] for k in vis_kps) / len(vis_kps))
                cy = int(sum(k["y"] for k in vis_kps) / len(vis_kps))
                label = f"{det['class_name']} {det['confidence']:.0%}"
                draw_label(label, max(0, cx - 30), max(16, cy - 12))

    return out


@app.route("/pipeline_status")
def pipeline_status():
    """Return live training status written by pipeline_train.py after each epoch."""
    spath = config.OUTPUT_DIR / "pipeline_status.json"
    if not spath.exists():
        return jsonify({"status": "idle"})
    with open(spath) as f:
        return jsonify(json.load(f))


@app.route("/pipeline_results")
def pipeline_results():
    """Return saved pipeline comparison results (pipeline_results.json)."""
    rpath = config.OUTPUT_DIR / "pipeline_results.json"
    if not rpath.exists():
        return jsonify({"results": [], "status": "not_started"})
    with open(rpath) as f:
        results = json.load(f)
    return jsonify({"results": results, "status": "ok"})


@app.route("/yolo_pipeline_results")
def yolo_pipeline_results():
    """Return YOLO pipeline comparison results (yolo_pipeline_results.json)."""
    rpath = config.OUTPUT_DIR / "yolo_pipeline_results.json"
    if not rpath.exists():
        return jsonify({"results": {}, "status": "not_started"})
    with open(rpath) as f:
        results = json.load(f)
    return jsonify({"results": results, "status": "ok"})


@app.route("/pipeline_charts")
def pipeline_charts():
    """
    Generate matplotlib evaluation charts from pipeline_results.json (KRCNN).
    Returns up to 3 base64 PNGs:
      1. F1 horizontal bar chart (all 18 pipelines, sorted)
      2. Precision-Recall scatter (one dot per pipeline, labeled)
      3. PR curves (one curve per pipeline, overlaid)
    """
    rpath = config.OUTPUT_DIR / "pipeline_results.json"
    if not rpath.exists():
        return jsonify({"error": "No KRCNN pipeline results yet."}), 404

    with open(rpath) as f:
        results = json.load(f)

    if not results:
        return jsonify({"error": "Pipeline results file is empty."}), 404

    # Filter to KRCNN results that have the richer metric set
    krcnn = [r for r in results if "f1" in r.get("metrics", {})]
    if not krcnn:
        return jsonify({"error": "No KRCNN metrics found (run pipeline_train.py first)."}), 404

    krcnn.sort(key=lambda r: r["metrics"].get("f1", 0), reverse=True)

    CONTRAST_COLORS = {"original": "#4a9eff", "clahe": "#7ed321", "retinex": "#f5a623"}
    FILTER_LABELS   = {
        "bilateral": "Bilateral", "nlm": "NLM", "bm3d": "BM3D",
        "snn": "SNN", "kramer_bruckner": "KB", "epoaf": "EPOAF",
    }

    def _label(r):
        parts = r["pipeline"].split("_", 1)
        return f"{parts[0].upper()} + {FILTER_LABELS.get(parts[1], parts[1]) if len(parts)>1 else ''}"

    def _bar_color(r):
        c = r["pipeline"].split("_", 1)[0]
        return CONTRAST_COLORS.get(c, "#aaaaaa")

    def _fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    charts = {}

    # ── Chart 1: F1 bar chart ─────────────────────────────────────────────────
    labels = [_label(r) for r in krcnn]
    f1s    = [r["metrics"]["f1"]        for r in krcnn]
    precs  = [r["metrics"]["precision"] for r in krcnn]
    recs   = [r["metrics"]["recall"]    for r in krcnn]
    aps    = [r["metrics"].get("ap", 0) for r in krcnn]
    colors = [_bar_color(r)             for r in krcnn]

    n = len(krcnn)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.42)))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#22263a")

    y = np.arange(n)
    bars = ax.barh(y, f1s, color=colors, alpha=0.85, height=0.55)
    ax.barh(y, aps, color=colors, alpha=0.30, height=0.55, label="AP")

    for bar, v in zip(bars, f1s):
        ax.text(min(v + 0.012, 0.98), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", ha="left", fontsize=8,
                color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color="white")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score", color="white", fontsize=9)
    ax.set_title("F1 (opaque) & AP (transparent) par pipeline — KRCNN",
                 color="white", fontsize=10, pad=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.xaxis.grid(True, alpha=0.2, color="white")
    ax.set_axisbelow(True)

    # Legend: contrast colors
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=k.upper())
              for k, c in CONTRAST_COLORS.items()]
    ax.legend(handles=legend, fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#22263a", edgecolor="#444",
              loc="lower right")
    fig.tight_layout()
    charts["bar_f1"] = _fig_to_b64(fig)

    # ── Chart 2: Precision-Recall scatter ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#22263a")

    for r, p, re, col in zip(krcnn, precs, recs, [_bar_color(r) for r in krcnn]):
        ax.scatter(re, p, color=col, s=80, alpha=0.85, zorder=3)
        ax.annotate(_label(r), (re, p), textcoords="offset points",
                    xytext=(5, 3), fontsize=6.5, color="white", alpha=0.85)

    # Iso-F1 curves
    x_iso = np.linspace(0.01, 1.0, 200)
    for iso_f1 in [0.3, 0.5, 0.7, 0.9]:
        y_iso = iso_f1 * x_iso / (2 * x_iso - iso_f1 + 1e-10)
        mask  = (y_iso >= 0) & (y_iso <= 1)
        ax.plot(x_iso[mask], y_iso[mask], "--", color="white", alpha=0.12, lw=0.8)
        idx = np.argmin(np.abs(x_iso[mask] - 0.85))
        ax.text(x_iso[mask][idx], y_iso[mask][idx] + 0.01,
                f"F1={iso_f1:.1f}", fontsize=6, color="white", alpha=0.35)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", color="white", fontsize=10)
    ax.set_ylabel("Precision", color="white", fontsize=10)
    ax.set_title("Precision vs Recall — 18 pipelines KRCNN", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(True, alpha=0.15, color="white")

    legend = [Patch(color=c, label=k.upper()) for k, c in CONTRAST_COLORS.items()]
    ax.legend(handles=legend, fontsize=8, framealpha=0.3,
              labelcolor="white", facecolor="#22263a", edgecolor="#444")
    fig.tight_layout()
    charts["scatter_pr"] = _fig_to_b64(fig)

    # ── Chart 3: PR curves (one per pipeline) ─────────────────────────────────
    has_curves = any("pr_curve" in r.get("metrics", {}) and
                     r["metrics"]["pr_curve"].get("recall")
                     for r in krcnn)
    if has_curves:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#22263a")

        alpha_step = max(0.25, 1.0 / max(len(krcnn), 1))
        for i, r in enumerate(krcnn):
            prc = r["metrics"].get("pr_curve", {})
            if not prc.get("recall"):
                continue
            rec_pts  = prc["recall"]
            prec_pts = prc["precision"]
            col      = _bar_color(r)
            lw       = 2.0 if i == 0 else 1.0
            alpha    = 1.0 if i == 0 else max(0.30, 0.85 - i * alpha_step * 0.5)
            ax.plot(rec_pts, prec_pts, color=col, lw=lw, alpha=alpha,
                    label=f"{_label(r)} AP={r['metrics'].get('ap', 0):.2f}")

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall", color="white", fontsize=10)
        ax.set_ylabel("Precision", color="white", fontsize=10)
        ax.set_title("Courbes Precision-Recall — 18 pipelines KRCNN\n"
                     "(IoU ≥ 0.5, matching global par score décroissant)",
                     color="white", fontsize=9)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.15, color="white")
        ax.legend(fontsize=6.5, framealpha=0.35, labelcolor="white",
                  facecolor="#1a1d27", edgecolor="#444",
                  ncol=2, loc="lower left")
        fig.tight_layout()
        charts["pr_curves"] = _fig_to_b64(fig)

    return jsonify(charts)


@app.route("/predict_yolo", methods=["POST"])
def predict_yolo_route():
    """YOLOv8-pose full-image inference endpoint."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if _yolo_model is None:
        return jsonify({"error": "YOLO model not trained yet. Run: python3 yolo_train.py"}), 503

    file    = request.files["image"]
    score_t = float(request.form.get("score", 0.4))

    pil_img = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(pil_img)
    H, W    = img_rgb.shape[:2]

    results = _yolo_model.predict(
        img_rgb, conf=score_t, verbose=False,
        device=str(DEVICE).replace("mps", "mps"),
    )
    r = results[0]

    detections = []
    img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()   if r.boxes is not None else []
    confs      = r.boxes.conf.cpu().numpy()    if r.boxes is not None else []
    clss       = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
    kps_xy     = r.keypoints.xy.cpu().numpy()  if r.keypoints is not None else []
    kps_conf   = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None

    CLASS_NAMES_YOLO = config.CLASS_NAMES   # ["cleaning bee", "normal bee"]

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        cls_idx = int(clss[i])
        kps = []
        for k in range(len(kps_xy[i])):
            kpx, kpy = float(kps_xy[i][k][0]), float(kps_xy[i][k][1])
            vis = float(kps_conf[i][k]) if kps_conf is not None else 1.0
            kps.append({"x": kpx, "y": kpy, "vis": vis, "visible": vis > 0.5})
        detections.append({
            "class_name": CLASS_NAMES_YOLO[cls_idx] if cls_idx < len(CLASS_NAMES_YOLO) else str(cls_idx),
            "class_idx":  cls_idx,
            "confidence": float(confs[i]),
            "bbox":       [float(x1), float(y1), float(x2), float(y2)],
            "keypoints":  kps,
        })

    # Draw detections on image
    KP_COLS_BGR = [(80, 80, 255), (30, 180, 255), (255, 180, 80)]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = (255, 180, 50) if det["class_idx"] == 0 else (80, 200, 80)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(img_bgr, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for k, kp in enumerate(det["keypoints"]):
            if kp["visible"]:
                px, py = int(kp["x"]), int(kp["y"])
                col_bgr = KP_COLS_BGR[k % len(KP_COLS_BGR)]
                cv2.circle(img_bgr, (px, py), 5, (0, 0, 0), -1)
                cv2.circle(img_bgr, (px, py), 4, col_bgr,   -1)

    b64 = _encode_annotated(img_bgr)
    return jsonify({
        "mode":            "yolo",
        "detections":      detections,
        "annotated_image": b64,
        "image_size":      [W, H],
        "n_detections":    len(detections),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Run all behavioural analyses on KRCNN detections."""
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


# ── Video routes ───────────────────────────────────────────────────────────

@app.route("/predict_video", methods=["POST"])
def predict_video():
    """Accept a video upload, start a background analysis job, return job_id."""
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400
    if _krcnn_model is None:
        return jsonify({"error": "Keypoint R-CNN not loaded — train it first"}), 503

    _purge_old_jobs()

    file       = request.files["video"]
    frame_step = max(1, min(int(request.form.get("frame_step", 10)), 120))
    score_t    = float(request.form.get("score", 0.5))

    job_id  = str(uuid.uuid4())[:8]
    job_dir = VIDEO_JOB_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext        = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        ext = ".mp4"
    video_path = job_dir / f"input{ext}"
    file.save(str(video_path))

    _video_jobs[job_id] = {
        "status":           "processing",
        "progress":         0,
        "message":          "Starting…",
        "created":          time.time(),
        "job_dir":          job_dir,
        "current_frame":    0,
        "processed_frames": 0,
        "last_n_bees":      0,
        "total_frames":     0,
        "fps":              0,
        "video_info":       {},
    }

    t = threading.Thread(
        target=_process_video_job,
        args=(job_id, video_path, frame_step, score_t),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/video_status/<job_id>")
def video_status(job_id):
    """Return lightweight job status (no heavy per-frame data)."""
    job = _video_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404

    # Fields safe to send every poll (skip large arrays)
    SKIP = {"job_dir", "frame_stats", "output_path"}
    payload = {k: v for k, v in job.items() if k not in SKIP}
    return jsonify(payload)


@app.route("/video_download/<job_id>")
def video_download(job_id):
    """Stream the annotated output video."""
    job = _video_jobs.get(job_id)
    if job is None or job.get("status") != "done":
        return jsonify({"error": "Job not ready"}), 404
    out_path = job.get("output_path")
    if not out_path or not Path(out_path).exists():
        return jsonify({"error": "Output file missing"}), 404
    return send_file(out_path, mimetype="video/mp4", as_attachment=True,
                     download_name="bee_analysis.mp4")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n  → Open http://localhost:{args.port}\n")
    app.run(debug=False, host=args.host, port=args.port)
