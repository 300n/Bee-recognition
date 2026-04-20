"""
krcnn_dataset.py
----------------
Dataset for Keypoint R-CNN.  Returns (image_tensor, target_dict) per image
(not per instance), which is what torchvision detection models expect.

image_tensor : (3, H, W) float32 in [0, 1]  — NO ImageNet normalisation
               (the model's GeneralizedRCNNTransform does it internally)

target_dict  :
  boxes      (N, 4)    float32  [x1, y1, x2, y2]  absolute pixel coords
  labels     (N,)      int64    1=cleaning bee  2=normal bee  (0=background)
  keypoints  (N, K, 3) float32  [x, y, visibility]  v∈{0,1,2}
  image_id   ()        int64
  area       (N,)      float32
  iscrowd    (N,)      int64

Augmentation (training only):
  - random horizontal flip  (boxes + keypoints transformed)
  - random vertical flip
  - colour jitter
  - random 90° rotation
"""

from __future__ import annotations
import json, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import config

MAX_KP = config.MAX_KEYPOINTS   # 3


# ── COCO loader helpers ────────────────────────────────────────────────────

def _load_coco(path):
    with open(path) as f:
        return json.load(f)

def _build_image_map(images):
    return {img["id"]: img for img in images}

def _group_annotations(annotations):
    groups: dict[int, list] = {}
    for a in annotations:
        groups.setdefault(a["image_id"], []).append(a)
    return groups

def _split_image_ids(ids, seed=42):
    rng = random.Random(seed)
    ids = list(ids); rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    return (set(ids[:n_train]),
            set(ids[n_train:n_train+n_val]),
            set(ids[n_train+n_val:]))


# ── Augmentation helpers ───────────────────────────────────────────────────

def _hflip(img_t, boxes, keypoints, W):
    """Horizontal flip for a (C,H,W) tensor + boxes + keypoints."""
    img_t = img_t.flip(-1)
    if len(boxes):
        x1 = W - boxes[:, 2]
        x2 = W - boxes[:, 0]
        boxes = torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=1)
        kp = keypoints.clone()
        kp[:, :, 0] = W - kp[:, :, 0]
        keypoints = kp
    return img_t, boxes, keypoints

def _vflip(img_t, boxes, keypoints, H):
    img_t = img_t.flip(-2)
    if len(boxes):
        y1 = H - boxes[:, 3]
        y2 = H - boxes[:, 1]
        boxes = torch.stack([boxes[:, 0], y1, boxes[:, 2], y2], dim=1)
        kp = keypoints.clone()
        kp[:, :, 1] = H - kp[:, :, 1]
        keypoints = kp
    return img_t, boxes, keypoints

def _rot90(img_t, boxes, keypoints, H, W, k):
    """k ∈ {1,2,3} — 90°, 180°, 270° counter-clockwise."""
    img_t = torch.rot90(img_t, k, dims=[-2, -1])
    if not len(boxes):
        return img_t, boxes, keypoints
    kp = keypoints.clone()
    bx = boxes.clone()
    for _ in range(k):
        # (x,y) → (y, H-x)  under 90° CCW
        old_x_kp = kp[:, :, 0].clone()
        old_y_kp = kp[:, :, 1].clone()
        kp[:, :, 0] = old_y_kp
        kp[:, :, 1] = W - old_x_kp

        old_x1 = bx[:, 0].clone(); old_y1 = bx[:, 1].clone()
        old_x2 = bx[:, 2].clone(); old_y2 = bx[:, 3].clone()
        # new box corners after 90° CCW
        new_x1 = old_y1;      new_y1 = W - old_x2
        new_x2 = old_y2;      new_y2 = W - old_x1
        bx = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
        H, W = W, H   # swap dims
    return img_t, bx, kp


# ── Dataset ────────────────────────────────────────────────────────────────

class BeeKeypointRCNNDataset(Dataset):
    def __init__(self, coco_json, image_dir, image_ids, augment=False):
        self.image_dir = Path(image_dir)
        self.augment   = augment
        data = _load_coco(coco_json)
        image_map  = _build_image_map(data["images"])
        ann_groups = _group_annotations(data["annotations"])

        self.samples = []   # (img_info, [ann, ...])
        for img_id in image_ids:
            if img_id not in image_map:
                continue
            img_info = image_map[img_id]
            anns = [a for a in ann_groups.get(img_id, [])
                    if a["category_id"] in (1, 2)]
            if not anns:
                continue
            self.samples.append((img_info, anns))

        print(f"  → {len(self.samples)} images  "
              f"({sum(len(s[1]) for s in self.samples)} instances)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_info, anns = self.samples[idx]
        img_path = self.image_dir / img_info["file_name"]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # Convert to float tensor [0,1]
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0
                                 ).permute(2, 0, 1)   # (3,H,W)

        # ── Build target arrays ──────────────────────────────────────
        boxes_list = []
        labels_list = []
        kp_list = []
        areas_list = []

        for ann in anns:
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            x1 = max(0.0, x);          y1 = max(0.0, y)
            x2 = min(float(W), x+bw);  y2 = min(float(H), y+bh)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(ann["category_id"])   # 1 or 2
            areas_list.append(float(ann.get("area", bw * bh)))

            # Keypoints: pad to MAX_KP
            kp_raw = ann.get("keypoints", [])
            n_kp = len(kp_raw) // 3
            kps = np.zeros((MAX_KP, 3), dtype=np.float32)
            for i in range(min(n_kp, MAX_KP)):
                kps[i, 0] = float(kp_raw[i*3])
                kps[i, 1] = float(kp_raw[i*3 + 1])
                kps[i, 2] = float(kp_raw[i*3 + 2])   # 0/1/2
            kp_list.append(kps)

        if not boxes_list:
            # Empty image — return dummy target (will be skipped by collate)
            target = {
                "boxes":     torch.zeros((0, 4), dtype=torch.float32),
                "labels":    torch.zeros(0, dtype=torch.int64),
                "keypoints": torch.zeros((0, MAX_KP, 3), dtype=torch.float32),
                "image_id":  torch.tensor(img_info["id"], dtype=torch.int64),
                "area":      torch.zeros(0, dtype=torch.float32),
                "iscrowd":   torch.zeros(0, dtype=torch.int64),
            }
            return img_t, target

        boxes     = torch.tensor(boxes_list,  dtype=torch.float32)
        labels    = torch.tensor(labels_list, dtype=torch.int64)
        keypoints = torch.tensor(np.stack(kp_list), dtype=torch.float32)
        areas     = torch.tensor(areas_list, dtype=torch.float32)
        iscrowd   = torch.zeros(len(boxes_list), dtype=torch.int64)

        # ── Augmentation ─────────────────────────────────────────────
        if self.augment:
            if random.random() < 0.5:
                img_t, boxes, keypoints = _hflip(img_t, boxes, keypoints, W)
            if random.random() < 0.2:
                img_t, boxes, keypoints = _vflip(img_t, boxes, keypoints, H)
            k = random.choice([0, 0, 0, 1, 2, 3])   # mostly no rotation
            if k > 0:
                img_t, boxes, keypoints = _rot90(img_t, boxes, keypoints, H, W, k)
            # Colour jitter (image only)
            if random.random() < 0.7:
                img_t = TF.adjust_brightness(img_t, 1 + random.uniform(-0.3, 0.3))
                img_t = TF.adjust_contrast(img_t,   1 + random.uniform(-0.3, 0.3))
                img_t = TF.adjust_saturation(img_t, 1 + random.uniform(-0.2, 0.2))
            if random.random() < 0.3:
                img_t = TF.gaussian_blur(img_t, kernel_size=5)
            img_t = img_t.clamp(0.0, 1.0)

            # Clamp boxes after geometric transforms
            _, aug_h, aug_w = img_t.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, aug_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, aug_h)
            # Remove degenerate boxes
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes     = boxes[keep]
            labels    = labels[keep]
            keypoints = keypoints[keep]
            areas     = areas[keep]
            iscrowd   = iscrowd[keep]

        target = {
            "boxes":     boxes,
            "labels":    labels,
            "keypoints": keypoints,
            "image_id":  torch.tensor(img_info["id"], dtype=torch.int64),
            "area":      areas,
            "iscrowd":   iscrowd,
        }
        return img_t, target


def collate_fn(batch):
    """Variable number of instances per image → keep as list."""
    return tuple(zip(*batch))


def get_krcnn_datasets(coco_json, image_dir):
    data = _load_coco(coco_json)
    all_ids = [img["id"] for img in data["images"]]
    train_ids, val_ids, test_ids = _split_image_ids(all_ids, seed=config.RANDOM_SEED)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test images")
    train_ds = BeeKeypointRCNNDataset(coco_json, image_dir, train_ids, augment=True)
    val_ds   = BeeKeypointRCNNDataset(coco_json, image_dir, val_ids,   augment=False)
    test_ds  = BeeKeypointRCNNDataset(coco_json, image_dir, test_ids,  augment=False)
    return train_ds, val_ds, test_ds
