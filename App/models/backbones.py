"""
BackbonePoseModel
-----------------
Pretrained backbone (ResNet18 / MobileNetV3-Small / EfficientNet-B0)
with a lightweight multi-task head for bee classification + keypoint estimation.

Fine-tuning strategy
--------------------
- Backbone body is initially frozen; only the head trains for WARMUP_EPOCHS.
  (Callers are responsible for calling .unfreeze_backbone() after warmup.)
- This avoids destroying ImageNet features on a small dataset.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def _make_head(in_features: int, num_classes: int, num_keypoints: int):
    """Shared MLP trunk + three task-specific output branches."""
    shared = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )
    cls_head = nn.Linear(256, num_classes)
    kp_head  = nn.Linear(256, num_keypoints * 2)
    vis_head = nn.Linear(256, num_keypoints)
    return shared, cls_head, kp_head, vis_head


class BackbonePoseModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        num_classes: int = 2,
        num_keypoints: int = 3,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone_name = backbone_name

        weights = "DEFAULT" if pretrained else None

        # ── Build backbone & extract feature dim ──────────────────────
        if backbone_name == "resnet18":
            base = tvm.resnet18(weights=weights)
            in_features = base.fc.in_features
            # Remove final FC
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            # → output (B, 512, 1, 1) after AdaptiveAvgPool already inside

        elif backbone_name == "mobilenet_v3":
            base = tvm.mobilenet_v3_small(weights=weights)
            # The classifier[0] tells us the in_features
            in_features = base.classifier[0].in_features
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone_name == "efficientnet_b0":
            base = tvm.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_name_str = backbone_name
        self.in_features = in_features

        # ── Head ──────────────────────────────────────────────────────
        self.shared, self.cls_head, self.kp_head, self.vis_head = \
            _make_head(in_features, num_classes, num_keypoints)

        self._init_head()
        self.freeze_backbone()   # starts frozen

    def _init_head(self):
        for m in [self.shared, self.cls_head, self.kp_head, self.vis_head]:
            for layer in m.modules() if hasattr(m, 'modules') else [m]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _extract_features(self, x):
        if self.backbone_name_str == "resnet18":
            feat = self.backbone(x)          # (B, 512, 1, 1)
            feat = feat.flatten(1)           # (B, 512)
        else:
            feat = self.backbone(x)          # (B, C, H', W')
            feat = self.pool(feat)           # (B, C, 1, 1)
            feat = feat.flatten(1)           # (B, C)
        return feat

    def forward(self, x):
        feat = self._extract_features(x)
        feat = self.shared(feat)

        cls_logits = self.cls_head(feat)                          # (B, C)
        kp = torch.sigmoid(self.kp_head(feat))                   # (B, K*2)
        kp = kp.view(-1, self.num_keypoints, 2)                  # (B, K, 2)
        vis_logits = self.vis_head(feat)                          # (B, K)

        return cls_logits, kp, vis_logits

    def count_parameters(self, trainable_only=True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )
