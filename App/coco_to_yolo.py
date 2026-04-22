"""
coco_to_yolo.py
---------------
Converts the Roboflow COCO keypoint dataset to YOLO pose format and
splits it into train / val / test sets (80 / 10 / 10 %).

Input : App/Bees R-D.coco/train/_annotations.coco.json  +  images
Output: App/datasets/bee_yolo/   (replaces existing dataset)

YOLO pose label format per line:
  class cx cy w h  kp0_x kp0_y kp0_v  kp1_x kp1_y kp1_v  kp2_x kp2_y kp2_v
  (all coordinates normalised to [0, 1])

Category mapping (COCO id → YOLO class):
  1 (cleaning bee) → 0
  2 (normal bee)   → 1
"""

import json
import random
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
COCO_DIR   = Path("Bees R-D.coco/train")
COCO_JSON  = COCO_DIR / "_annotations.coco.json"
OUT_DIR    = Path("datasets/bee_yolo")

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED  = 42

# COCO category_id → YOLO class index
CAT_MAP = {1: 0, 2: 1}   # cleaning bee → 0,  normal bee → 1

# ── Load COCO annotations ──────────────────────────────────────────────────────
print("Loading COCO annotations…")
with open(COCO_JSON) as f:
    coco = json.load(f)

img_info  = {img["id"]: img for img in coco["images"]}
ann_by_img = {}
for ann in coco["annotations"]:
    if ann["category_id"] not in CAT_MAP:
        continue
    if "keypoints" not in ann:
        continue
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

# Keep only images that have at least one valid annotation
valid_ids = [img_id for img_id in img_info if img_id in ann_by_img]
print(f"  {len(valid_ids)} images with annotations  "
      f"({len(coco['images']) - len(valid_ids)} images without annotations skipped)")

# ── Split ─────────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(valid_ids)

n        = len(valid_ids)
n_train  = int(n * SPLIT_RATIOS["train"])
n_val    = int(n * SPLIT_RATIOS["val"])

splits = {
    "train": valid_ids[:n_train],
    "val":   valid_ids[n_train:n_train + n_val],
    "test":  valid_ids[n_train + n_val:],
}
for name, ids in splits.items():
    print(f"  {name:5s}: {len(ids)} images")

# ── Create output directories ──────────────────────────────────────────────────
if OUT_DIR.exists():
    print(f"\nRemoving existing {OUT_DIR} …")
    shutil.rmtree(OUT_DIR)

for split in splits:
    (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ── Conversion helper ─────────────────────────────────────────────────────────

def convert_image(img_id: int, split: str):
    info  = img_info[img_id]
    W, H  = info["width"], info["height"]
    fname = info["file_name"]

    src = COCO_DIR / fname
    dst_img = OUT_DIR / "images" / split / fname
    shutil.copy2(src, dst_img)

    lines = []
    for ann in ann_by_img[img_id]:
        cls = CAT_MAP[ann["category_id"]]

        # COCO bbox: [x_topleft, y_topleft, width, height] in pixels (may be str)
        bx, by, bw, bh = [float(v) for v in ann["bbox"]]
        cx = (bx + bw / 2) / W
        cy = (by + bh / 2) / H
        nw = bw / W
        nh = bh / H

        # Keypoints: [x1, y1, v1, x2, y2, v2, x3, y3, v3] in pixels
        kps = ann["keypoints"]
        kp_str = ""
        for i in range(0, len(kps), 3):
            kx = float(kps[i])     / W
            ky = float(kps[i + 1]) / H
            kv = int(kps[i + 2])
            kp_str += f" {kx:.6f} {ky:.6f} {kv}"

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}{kp_str}")

    dst_lbl = OUT_DIR / "labels" / split / (Path(fname).stem + ".txt")
    dst_lbl.write_text("\n".join(lines))


# ── Run conversion ─────────────────────────────────────────────────────────────
print("\nConverting…")
for split, ids in splits.items():
    for img_id in ids:
        convert_image(img_id, split)
    print(f"  {split}: done")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Dataset written to", OUT_DIR, "──")
for split in splits:
    n_img = len(list((OUT_DIR / "images" / split).iterdir()))
    n_lbl = len(list((OUT_DIR / "labels" / split).iterdir()))
    print(f"  {split:5s}: {n_img} images, {n_lbl} labels")

print("\nDone. Update bee_pose.yaml if needed (path should point to datasets/bee_yolo).")
