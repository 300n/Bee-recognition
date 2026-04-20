from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("Bees R-D.coco/train")
ANNOTATION_FILE = DATA_ROOT / "_annotations.coco.json"
OUTPUT_DIR = Path("outputs")

# ── Dataset ────────────────────────────────────────────────────────────────
CROP_SIZE = 128          # resize all crops to this square size
MAX_KEYPOINTS = 3        # pad all instances to this many keypoints
NUM_CLASSES = 2          # cleaning bee (0), normal bee (1)
BBOX_PADDING = 0.25      # relative padding around the bounding box crop

# COCO category_id → class index
CATEGORY_MAP = {1: 0, 2: 1}   # 1=cleaning bee → 0, 2=normal bee → 1
CLASS_NAMES = ["cleaning bee", "normal bee"]

# ── Split ──────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── Augmentation ───────────────────────────────────────────────────────────
# Applied only to the training set
AUG_HFLIP_P        = 0.5
AUG_VFLIP_P        = 0.2
AUG_ROTATE_LIMIT   = 30
AUG_ROTATE_P       = 0.5
AUG_COLOR_JITTER_P = 0.7
AUG_BLUR_P         = 0.3
AUG_ERASING_P      = 0.3
AUG_AFFINE_P       = 0.3

# Normalisation (ImageNet statistics – used for all models including custom CNN)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 4          # multiprocess augmentation (spawn-safe on macOS Python 3.8+)
LR           = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 80
PATIENCE     = 15         # early stopping patience (epochs on val loss plateau)
LR_SCHEDULER = "cosine"   # "cosine" | "step"
WARMUP_EPOCHS = 3

# ── Loss weights ───────────────────────────────────────────────────────────
W_CLS = 1.0   # classification cross-entropy weight
W_KP  = 5.0   # visible keypoint MSE weight
W_VIS = 0.5   # keypoint visibility BCE weight

# ── Models ─────────────────────────────────────────────────────────────────
MODELS = {
    "custom_cnn":       {"pretrained": False, "lr_scale": 1.0},
    "resnet18":         {"pretrained": True,  "lr_scale": 0.1},
    "mobilenet_v3":     {"pretrained": True,  "lr_scale": 0.1},
    "efficientnet_b0":  {"pretrained": True,  "lr_scale": 0.1},
}

# ── Evaluation ─────────────────────────────────────────────────────────────
PCK_THRESHOLDS = [0.05, 0.10, 0.20]   # fraction of bbox diagonal
