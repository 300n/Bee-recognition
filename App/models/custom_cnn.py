"""
CustomCNN
---------
Lightweight convolutional architecture trained from scratch.

Architecture:
  5 × ConvBlock (3×3 conv, BN, ReLU, 2×2 MaxPool)  →  feature map
  Global Average Pooling
  Shared FC (512 → 256)
  ┌──── Classification head (256 → num_classes)
  ├──── Keypoint head       (256 → num_keypoints × 2)  [sigmoid → [0,1]]
  └──── Visibility head     (256 → num_keypoints)

Input: (B, 3, 224, 224)
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, num_keypoints: int = 3):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 224 → 112 → 56 → 28 → 14 → 7
        self.encoder = nn.Sequential(
            _conv_block(3,   32),
            _conv_block(32,  64),
            _conv_block(64,  128),
            _conv_block(128, 256),
            _conv_block(256, 512),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 512, 1, 1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        self.cls_head = nn.Linear(256, num_classes)
        self.kp_head  = nn.Linear(256, num_keypoints * 2)
        self.vis_head = nn.Linear(256, num_keypoints)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.gap(feat)
        feat = self.shared(feat)

        cls_logits = self.cls_head(feat)                          # (B, C)
        kp = torch.sigmoid(self.kp_head(feat))                   # (B, K*2) in [0,1]
        kp = kp.view(-1, self.num_keypoints, 2)                  # (B, K, 2)
        vis_logits = self.vis_head(feat)                          # (B, K)

        return cls_logits, kp, vis_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
