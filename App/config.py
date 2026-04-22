from pathlib import Path

OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CLASSES    = 2
MAX_KEYPOINTS  = 3
CLASS_NAMES    = ["cleaning bee", "normal bee"]
CATEGORY_MAP   = {1: 0, 2: 1}   # COCO category_id → class index
