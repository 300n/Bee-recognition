"""
BeeKeypointDataset
------------------
Loads the COCO-format annotation file, extracts per-bee crops, and returns
tensors ready for training.

Each sample:
    image   : (3, H, W) float32, normalised
    keypoints: (MAX_KP, 2) float32, coords in [0,1] relative to the crop
    visibility: (MAX_KP,) float32, 1=visible, 0=absent/not-visible
    label   : int, 0=cleaning bee, 1=normal bee
    bbox_wh : (2,) float32, original bbox (w, h) in pixels – used for PCK metric
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config


# ── helpers ────────────────────────────────────────────────────────────────

def _parse_bbox(bbox_raw):
    """Return (x, y, w, h) as floats; COCO exports width/height as strings."""
    return [float(v) for v in bbox_raw]


def _load_coco(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def _build_image_map(images):
    return {img["id"]: img for img in images}


def _group_annotations(annotations):
    """Group annotation dicts by image_id."""
    groups: dict[int, list] = {}
    for ann in annotations:
        groups.setdefault(ann["image_id"], []).append(ann)
    return groups


def _split_image_ids(image_ids, seed=42):
    """Deterministic 70/15/15 split at the image level."""
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    train_ids = set(ids[:n_train])
    val_ids   = set(ids[n_train:n_train + n_val])
    test_ids  = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


# ── main dataset class ─────────────────────────────────────────────────────

class BeeKeypointDataset(Dataset):
    """
    One sample per annotated bee instance (crop-based approach).

    Parameters
    ----------
    coco_json   : path to _annotations.coco.json
    image_dir   : directory containing the PNG files
    image_ids   : subset of image IDs to use (train / val / test)
    transform   : albumentations Compose pipeline (receives image + keypoints)
    """

    def __init__(
        self,
        coco_json: Path,
        image_dir: Path,
        image_ids: set[int],
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform

        data = _load_coco(coco_json)
        image_map = _build_image_map(data["images"])
        ann_groups = _group_annotations(data["annotations"])

        self.samples = []
        for img_id in image_ids:
            if img_id not in image_map:
                continue
            img_info = image_map[img_id]
            anns = ann_groups.get(img_id, [])
            for ann in anns:
                # Only keep valid categories
                if ann["category_id"] not in config.CATEGORY_MAP:
                    continue
                # Require at least one visible keypoint
                kp_raw = ann.get("keypoints", [])
                if len(kp_raw) < 3:      # need at least 1 triplet
                    continue
                self.samples.append((img_info, ann))

        print(f"  → {len(self.samples)} instances from {len(image_ids)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, ann = self.samples[idx]

        # ── Load image ──────────────────────────────────────────────────
        img_path = self.image_dir / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # ── Parse bbox & extract padded crop ────────────────────────────
        x, y, w, h = _parse_bbox(ann["bbox"])
        pad = config.BBOX_PADDING
        x1 = max(0.0, x - pad * w)
        y1 = max(0.0, y - pad * h)
        x2 = min(float(img_w), x + (1 + pad) * w)
        y2 = min(float(img_h), y + (1 + pad) * h)
        crop_w = x2 - x1
        crop_h = y2 - y1

        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            # Fallback: use full image
            crop = img
            x1, y1, crop_w, crop_h = 0.0, 0.0, float(img_w), float(img_h)

        # ── Parse keypoints ─────────────────────────────────────────────
        kp_raw = ann.get("keypoints", [])
        n_kp_in_ann = len(kp_raw) // 3

        # Normalise to crop space  → [0, 1]
        kp_coords = np.zeros((config.MAX_KEYPOINTS, 2), dtype=np.float32)
        vis       = np.zeros(config.MAX_KEYPOINTS,     dtype=np.float32)

        for i in range(min(n_kp_in_ann, config.MAX_KEYPOINTS)):
            kx = float(kp_raw[i * 3])
            ky = float(kp_raw[i * 3 + 1])
            v  = float(kp_raw[i * 3 + 2])
            if v > 0:   # 1=not-visible but labelled, 2=visible
                kp_coords[i, 0] = np.clip((kx - x1) / crop_w, 0.0, 1.0)
                kp_coords[i, 1] = np.clip((ky - y1) / crop_h, 0.0, 1.0)
                vis[i] = 1.0

        # ── Albumentations ──────────────────────────────────────────────
        # Pass keypoints as list of (x_abs, y_abs) in the crop
        if self.transform is not None:
            # Convert back to absolute crop pixels for albumentations
            kp_abs = [
                (float(kp_coords[i, 0] * (crop.shape[1] - 1)),
                 float(kp_coords[i, 1] * (crop.shape[0] - 1)))
                for i in range(config.MAX_KEYPOINTS)
            ]
            result = self.transform(image=crop, keypoints=kp_abs)
            crop = result["image"]          # already a tensor (C, H, W)
            aug_kp = result["keypoints"]    # list of (x, y) in augmented crop

            # Re-normalise augmented keypoints; keep visibility mask
            aug_h, aug_w = (config.CROP_SIZE, config.CROP_SIZE)
            for i in range(config.MAX_KEYPOINTS):
                if i < len(aug_kp) and vis[i] > 0:
                    ax, ay = aug_kp[i][0], aug_kp[i][1]
                    kp_coords[i, 0] = np.clip(ax / (aug_w - 1), 0.0, 1.0)
                    kp_coords[i, 1] = np.clip(ay / (aug_h - 1), 0.0, 1.0)
                elif vis[i] > 0:
                    # Keypoint moved out of frame during augmentation
                    vis[i] = 0.0
                    kp_coords[i] = 0.0
        else:
            # Validation / test: just resize + normalise
            crop = cv2.resize(crop, (config.CROP_SIZE, config.CROP_SIZE))
            crop = crop.astype(np.float32) / 255.0
            mean = np.array(config.NORM_MEAN, dtype=np.float32)
            std  = np.array(config.NORM_STD,  dtype=np.float32)
            crop = (crop - mean) / std
            crop = torch.from_numpy(crop.transpose(2, 0, 1))  # CHW

        label    = config.CATEGORY_MAP[ann["category_id"]]
        bbox_wh  = np.array([w, h], dtype=np.float32)

        return {
            "image":      crop,                                     # (3,H,W)
            "keypoints":  torch.from_numpy(kp_coords),             # (MAX_KP, 2)
            "visibility": torch.from_numpy(vis),                   # (MAX_KP,)
            "label":      torch.tensor(label, dtype=torch.long),   # scalar
            "bbox_wh":    torch.from_numpy(bbox_wh),               # (2,)
        }


# ── Factory functions ──────────────────────────────────────────────────────

def get_datasets(coco_json: Path, image_dir: Path):
    """Return (train_ds, val_ds, test_ds) with correct transforms."""
    from augmentation import get_train_transforms, get_val_transforms

    data = _load_coco(coco_json)
    all_ids = [img["id"] for img in data["images"]]
    train_ids, val_ids, test_ids = _split_image_ids(all_ids, seed=config.RANDOM_SEED)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test images")

    train_ds = BeeKeypointDataset(
        coco_json, image_dir, train_ids,
        transform=get_train_transforms(config.CROP_SIZE)
    )
    val_ds = BeeKeypointDataset(
        coco_json, image_dir, val_ids,
        transform=None
    )
    test_ds = BeeKeypointDataset(
        coco_json, image_dir, test_ids,
        transform=None
    )
    return train_ds, val_ds, test_ds


def compute_class_weights(dataset: BeeKeypointDataset) -> torch.Tensor:
    """Return per-class weights to handle the cleaning-bee / normal-bee imbalance."""
    counts = torch.zeros(config.NUM_CLASSES)
    for _, ann in dataset.samples:
        counts[config.CATEGORY_MAP[ann["category_id"]]] += 1
    total = counts.sum()
    weights = total / (config.NUM_CLASSES * counts)
    print(f"Class counts: {counts.tolist()}  weights: {weights.tolist()}")
    return weights
