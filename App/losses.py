"""
BeeKeypointLoss
---------------
Multi-task loss combining:
  - Cross-entropy for bee classification (cleaning vs normal)
  - MSE on visible keypoint coordinates (normalised to [0, 1])
  - BCE on keypoint visibility

Total loss = W_CLS * cls + W_KP * kp + W_VIS * vis

Class weights can be passed to handle the severe imbalance (51 vs 2670).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BeeKeypointLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        pred_cls:  torch.Tensor,   # (B, num_classes)
        pred_kp:   torch.Tensor,   # (B, K, 2)  in [0,1]
        pred_vis:  torch.Tensor,   # (B, K)     logits
        gt_cls:    torch.Tensor,   # (B,)       long
        gt_kp:     torch.Tensor,   # (B, K, 2)  in [0,1]
        gt_vis:    torch.Tensor,   # (B, K)     float 0/1
    ):
        # ── Classification loss ─────────────────────────────────────────
        cls_loss = F.cross_entropy(
            pred_cls, gt_cls,
            weight=self.class_weights
        )

        # ── Keypoint coordinate loss (only on visible kps) ──────────────
        # gt_vis: (B, K) → mask (B, K, 1)
        mask = gt_vis.unsqueeze(-1)                  # (B, K, 1)
        kp_diff = (pred_kp - gt_kp) ** 2            # (B, K, 2)
        kp_loss_per = (kp_diff * mask).sum()         # sum over visible
        n_visible = mask.sum().clamp(min=1.0)
        kp_loss = kp_loss_per / n_visible

        # ── Visibility loss ──────────────────────────────────────────────
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis, gt_vis)

        total = (
            config.W_CLS * cls_loss
            + config.W_KP  * kp_loss
            + config.W_VIS * vis_loss
        )

        return total, {
            "total": total.item(),
            "cls":   cls_loss.item(),
            "kp":    kp_loss.item(),
            "vis":   vis_loss.item(),
        }
