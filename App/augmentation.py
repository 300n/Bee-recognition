"""
Augmentation pipelines using albumentations.

Keypoints are passed in `xy` format (absolute pixel coords in the crop).
Albumentations automatically transforms them together with the image.

Speed note: all ops run in DataLoader workers (num_workers>0), so the
pipeline should be fast. We keep the most impactful transforms and skip
the heavy ones (Affine, GaussNoise) in favour of lighter alternatives.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def get_train_transforms(crop_size: int = 128) -> A.Compose:
    """Augmentation pipeline for the training set."""
    return A.Compose(
        [
            # ── Geometric ────────────────────────────────────────────────
            A.Resize(crop_size, crop_size),
            A.HorizontalFlip(p=config.AUG_HFLIP_P),
            A.VerticalFlip(p=config.AUG_VFLIP_P),
            A.Rotate(
                limit=config.AUG_ROTATE_LIMIT,
                border_mode=0,
                p=config.AUG_ROTATE_P,
            ),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.10, 0.10),
                rotate=0,              # rotation already handled above
                border_mode=0,
                p=0.4,
            ),
            # ── Photometric ──────────────────────────────────────────────
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
                p=config.AUG_COLOR_JITTER_P,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=config.AUG_BLUR_P),
            A.ToGray(p=0.05),
            # ── Regularisation ───────────────────────────────────────────
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(6, 20),
                hole_width_range=(6, 20),
                fill=0,
                p=config.AUG_ERASING_P,
            ),
            # ── Normalise & convert to tensor ────────────────────────────
            A.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
            angle_in_degrees=True,
        ),
    )


def get_val_transforms(crop_size: int = 128) -> None:
    """Val/test: handled inline in dataset (resize → normalise → tensor)."""
    return None
