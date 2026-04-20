"""
yolo_convert.py
---------------
Convert the COCO-keypoint annotations to YOLO-pose format and write the
dataset directory structure that Ultralytics expects.

Output layout:
    datasets/bee_yolo/
        images/train/   *.png  (symlinks to originals)
        images/val/
        images/test/
        labels/train/   *.txt  (one per image)
        labels/val/
        labels/test/
    bee_pose.yaml

YOLO label line format (one instance per line):
    class_id  cx  cy  bw  bh  kp0x kp0y kp0v  kp1x kp1y kp1v  kp2x kp2y kp2v
    (all coordinates normalised to [0,1]; visibility 0/1/2)

Classes:
    0 = cleaning bee  (COCO category_id 1)
    1 = normal bee    (COCO category_id 2)
"""

import json
import os
import shutil
from pathlib import Path
import random

import config

SRC_JSON  = config.ANNOTATION_FILE        # Bees R-D.coco/train/_annotations.coco.json
SRC_IMGS  = config.DATA_ROOT              # Bees R-D.coco/train/
OUT_ROOT  = Path("datasets/bee_yolo")
YAML_PATH = Path("bee_pose.yaml")

CATEGORY_TO_CLASS = {1: 0, 2: 1}         # COCO id → YOLO class index
NUM_KP = config.MAX_KEYPOINTS             # 3
RANDOM_SEED = config.RANDOM_SEED


def convert(src_json=SRC_JSON, src_imgs=SRC_IMGS,
            out_root=OUT_ROOT, yaml_path=YAML_PATH,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO):

    with open(src_json) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    ann_by_img: dict = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Split image IDs (same seed as KRCNN training for consistency)
    ids = list(img_map.keys())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    splits = {
        "train": ids[:n_train],
        "val":   ids[n_train:n_train + n_val],
        "test":  ids[n_train + n_val:],
    }
    print(f"Split: {len(splits['train'])} train / "
          f"{len(splits['val'])} val / {len(splits['test'])} test")

    total_written = 0
    for split, split_ids in splits.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in split_ids:
            img_info = img_map[img_id]
            fname    = img_info["file_name"]
            W, H     = float(img_info["width"]), float(img_info["height"])
            anns     = [a for a in ann_by_img.get(img_id, [])
                        if a["category_id"] in CATEGORY_TO_CLASS]
            if not anns:
                continue

            # Symlink image
            src = (Path(src_imgs) / fname).resolve()
            dst = img_dir / fname
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except Exception:
                    shutil.copy2(str(src), str(dst))

            # Write label file
            lines = []
            for ann in anns:
                cls = CATEGORY_TO_CLASS[ann["category_id"]]
                x, y, bw, bh = [float(v) for v in ann["bbox"]]
                # YOLO: centre + size, normalised
                cx = (x + bw / 2) / W
                cy = (y + bh / 2) / H
                nw = bw / W
                nh = bh / H
                # Keypoints
                kp_raw = ann.get("keypoints", [])
                kps_out = []
                for k in range(NUM_KP):
                    if k * 3 + 2 < len(kp_raw):
                        kpx = float(kp_raw[k * 3])     / W
                        kpy = float(kp_raw[k * 3 + 1]) / H
                        kpv = int(kp_raw[k * 3 + 2])   # 0/1/2
                    else:
                        kpx, kpy, kpv = 0.0, 0.0, 0
                    kps_out.extend([kpx, kpy, kpv])
                kp_str = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                                  for v in kps_out)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {kp_str}")

            lbl_path = lbl_dir / (Path(fname).stem + ".txt")
            lbl_path.write_text("\n".join(lines))
            total_written += 1

    print(f"Written {total_written} label files to {out_root}")

    # Write YAML
    yaml_content = f"""# YOLOv8-pose dataset — bee keypoints
path: {out_root.resolve()}
train: images/train
val:   images/val
test:  images/test

kpt_shape: [{NUM_KP}, 3]   # [num_keypoints, (x y visibility)]
flip_idx:  [0, 1, 2]       # symmetric mapping for h-flip augmentation

nc: 2
names:
  0: cleaning bee
  1: normal bee
"""
    yaml_path.write_text(yaml_content)
    print(f"YAML → {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    convert()
