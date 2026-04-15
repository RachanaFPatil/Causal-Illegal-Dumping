"""
Layer 1 — Configuration
Robust CCTV AI Engine: adds lighting analysis and dual-stream CLAHE settings.
All downstream layers (Layer 2+) are unaffected by new keys here.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "rtdetr-l.pt"
IMGSZ       = 640           # 640 is optimal for RT-DETR; higher hurts CPU perf
CONF_THRESH = 0.35
IOU_THRESH  = 0.35

DEVICE = "cpu"              # Change to "mps" (Mac) or "cuda" (GPU) as needed

KEEP_CLASSES = {
    "person",
    "bottle", "cup", "backpack", "handbag", "suitcase",
    "bag", "sports ball",
    "chair", "couch", "tv", "laptop",
    "box", "clock", "vase", "book",
    # Future: "trash can", "waste container" (not in COCO)
}

# ── Trash detection ────────────────────────────────────────────────────────────
TRASH_ENABLE          = True
TRASH_MIN_AREA        = 600
TRASH_HISTORY         = 200
TRASH_DIST2THRESHOLD  = 50.0
TRASH_LABEL           = "trash"

# ── Visualisation ─────────────────────────────────────────────────────────────
PERSON_COLOR  = (0, 200, 0)
OBJECT_COLOR  = (0, 100, 255)
TRASH_COLOR   = (0, 0, 255)
TEXT_COLOR    = (255, 255, 255)
BOX_THICKNESS = 2
FONT_SCALE    = 0.55

# ── Lighting Analysis (new) ───────────────────────────────────────────────────
# Brightness = mean pixel intensity of grayscale frame (0–255).
# Contrast   = standard deviation of grayscale pixel values (0–128 typical).
# CLAHE is applied when EITHER metric falls below its threshold.
LIGHTING_BRIGHTNESS_THRESH = 80    # below this → frame is considered "dark"
LIGHTING_CONTRAST_THRESH   = 40    # below this → frame is considered "flat/foggy"

# ── CLAHE Parameters (new) ────────────────────────────────────────────────────
# clipLimit: higher = more aggressive local contrast boost (typical: 2.0–4.0)
# tileGridSize: grid granularity for local histogram equalization
CLAHE_CLIP_LIMIT   = 2.5
CLAHE_TILE_SIZE    = (8, 8)

# ── Dual-Stream Fusion (new) ──────────────────────────────────────────────────
# IoU threshold used when merging detections from original + CLAHE streams.
# Pairs with IoU > this are treated as the same object → keep higher confidence.
FUSION_IOU_THRESH  = 0.45