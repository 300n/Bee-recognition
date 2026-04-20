"""
Evaluation metrics for bee keypoint estimation.

PCK  – Percentage of Correct Keypoints
         A keypoint is correct if its L2 error < threshold × bbox_diag
NME  – Normalized Mean Error (mean L2 / bbox_diag)
OKS  – Object Keypoint Similarity (COCO-style, simplified)
Acc  – Top-1 classification accuracy
"""

from __future__ import annotations

import numpy as np
import torch


def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def compute_pck(
    pred_kp:  torch.Tensor,   # (B, K, 2) normalised [0,1]
    gt_kp:    torch.Tensor,   # (B, K, 2) normalised [0,1]
    gt_vis:   torch.Tensor,   # (B, K)    float 0/1
    bbox_wh:  torch.Tensor,   # (B, 2)    original bbox pixels
    threshold: float = 0.10,
    crop_size: int = 224,
) -> float:
    """
    PCK@threshold where the reference distance is bbox_diagonal.
    Coordinates are normalised; we convert back to pixel space using crop_size
    for a fair comparison across different scales.
    """
    pred_kp_np  = _to_np(pred_kp)    # (B, K, 2)
    gt_kp_np    = _to_np(gt_kp)
    gt_vis_np   = _to_np(gt_vis)     # (B, K)
    bbox_wh_np  = _to_np(bbox_wh)    # (B, 2)

    # Reference distance: bbox diagonal in normalised space
    bbox_diag = np.sqrt(
        (bbox_wh_np[:, 0] / crop_size) ** 2 +
        (bbox_wh_np[:, 1] / crop_size) ** 2
    )  # (B,)

    correct = 0
    total   = 0

    for b in range(len(pred_kp_np)):
        ref = bbox_diag[b]
        if ref < 1e-6:
            continue
        for k in range(pred_kp_np.shape[1]):
            if gt_vis_np[b, k] < 0.5:
                continue
            dist = np.linalg.norm(pred_kp_np[b, k] - gt_kp_np[b, k])
            total   += 1
            if dist < threshold * ref:
                correct += 1

    return (correct / total) if total > 0 else 0.0


def compute_nme(
    pred_kp: torch.Tensor,
    gt_kp:   torch.Tensor,
    gt_vis:  torch.Tensor,
    bbox_wh: torch.Tensor,
    crop_size: int = 224,
) -> float:
    """Normalised Mean Error: mean(L2_error / bbox_diag) over visible kps."""
    pred_kp_np = _to_np(pred_kp)
    gt_kp_np   = _to_np(gt_kp)
    gt_vis_np  = _to_np(gt_vis)
    bbox_wh_np = _to_np(bbox_wh)

    bbox_diag = np.sqrt(
        (bbox_wh_np[:, 0] / crop_size) ** 2 +
        (bbox_wh_np[:, 1] / crop_size) ** 2
    )

    errors = []
    for b in range(len(pred_kp_np)):
        ref = bbox_diag[b]
        if ref < 1e-6:
            continue
        for k in range(pred_kp_np.shape[1]):
            if gt_vis_np[b, k] < 0.5:
                continue
            dist = np.linalg.norm(pred_kp_np[b, k] - gt_kp_np[b, k])
            errors.append(dist / ref)

    return float(np.mean(errors)) if errors else 0.0


def compute_oks(
    pred_kp: torch.Tensor,
    gt_kp:   torch.Tensor,
    gt_vis:  torch.Tensor,
    bbox_wh: torch.Tensor,
    sigmas:  tuple[float, ...] | None = None,
    crop_size: int = 224,
) -> float:
    """
    Mean OKS (Object Keypoint Similarity) across all instances.
    OKS_i = Σ_k [ exp(-d²_k / 2s²_k A) * v_k ] / Σ_k v_k
    where A = bbox_area, s_k = per-keypoint sigma.
    """
    pred_kp_np = _to_np(pred_kp)
    gt_kp_np   = _to_np(gt_kp)
    gt_vis_np  = _to_np(gt_vis)
    bbox_wh_np = _to_np(bbox_wh)

    K = pred_kp_np.shape[1]
    if sigmas is None:
        sigmas = tuple([0.1] * K)   # equal sigmas (generic keypoints)

    s2 = (2 * np.array(sigmas) ** 2)    # (K,)

    oks_scores = []
    for b in range(len(pred_kp_np)):
        w, h = bbox_wh_np[b]
        area = (w / crop_size) * (h / crop_size)
        if area < 1e-10:
            continue
        num = 0.0
        den = 0.0
        for k in range(K):
            if gt_vis_np[b, k] < 0.5:
                continue
            d2 = np.sum((pred_kp_np[b, k] - gt_kp_np[b, k]) ** 2)
            e  = np.exp(-d2 / (s2[k] * area))
            num += e
            den += 1.0
        if den > 0:
            oks_scores.append(num / den)

    return float(np.mean(oks_scores)) if oks_scores else 0.0


def compute_accuracy(
    pred_cls: torch.Tensor,   # (B, C) logits
    gt_cls:   torch.Tensor,   # (B,)   long
) -> float:
    pred_labels = pred_cls.argmax(dim=1)
    return (pred_labels == gt_cls).float().mean().item()


class MetricAccumulator:
    """Accumulate batched predictions for epoch-level metric computation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred_cls:  list[torch.Tensor] = []
        self._gt_cls:    list[torch.Tensor] = []
        self._pred_kp:   list[torch.Tensor] = []
        self._gt_kp:     list[torch.Tensor] = []
        self._gt_vis:    list[torch.Tensor] = []
        self._bbox_wh:   list[torch.Tensor] = []

    def update(self, pred_cls, pred_kp, gt_cls, gt_kp, gt_vis, bbox_wh):
        self._pred_cls.append(pred_cls.detach().cpu())
        self._gt_cls.append(gt_cls.detach().cpu())
        self._pred_kp.append(pred_kp.detach().cpu())
        self._gt_kp.append(gt_kp.detach().cpu())
        self._gt_vis.append(gt_vis.detach().cpu())
        self._bbox_wh.append(bbox_wh.detach().cpu())

    def compute(self, pck_thresholds=(0.05, 0.10, 0.20)):
        pred_cls = torch.cat(self._pred_cls)
        gt_cls   = torch.cat(self._gt_cls)
        pred_kp  = torch.cat(self._pred_kp)
        gt_kp    = torch.cat(self._gt_kp)
        gt_vis   = torch.cat(self._gt_vis)
        bbox_wh  = torch.cat(self._bbox_wh)

        results = {
            "accuracy": compute_accuracy(pred_cls, gt_cls),
            "nme":      compute_nme(pred_kp, gt_kp, gt_vis, bbox_wh),
            "oks":      compute_oks(pred_kp, gt_kp, gt_vis, bbox_wh),
        }
        for thr in pck_thresholds:
            key = f"pck@{int(thr*100):02d}"
            results[key] = compute_pck(pred_kp, gt_kp, gt_vis, bbox_wh, threshold=thr)

        return results
