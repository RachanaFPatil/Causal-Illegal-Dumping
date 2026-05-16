"""
Layer 1 — Configuration
Robust CCTV AI Engine: adds lighting analysis and dual-stream CLAHE settings.
All downstream layers (Layer 2+) are unaffected by new keys here.

WINDOWS/CPU FIX: CONF_THRESH lowered 0.35 → 0.20 and IOU_THRESH raised
  0.20 → 0.45 for CPU inference.

  WHY: On Apple MPS (Mac), RT-DETR returns handbag/bottle/object detections
  at conf ~0.35-0.50. On CPU (Windows), the same model returns these
  non-person objects at conf ~0.20-0.30 — persons still score ~0.85+.
  With CONF_THRESH=0.35, all non-person objects are filtered out before
  they reach the tracker. The diagnose output confirmed only 2 person
  detections on frame 1 with no objects — this threshold is why.

  IOU_THRESH raised 0.20 → 0.45: the NMS pass at 0.20 was too aggressive
  on CPU and was suppressing small object detections that slightly overlapped
  with person bboxes.

DEVICE is now computed at import time using auto-detection.
'mps' is only selected on Apple Silicon Macs. On Windows this
always resolves to 'cuda' (if available) or 'cpu'.
"""
import torch
import platform

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

DEVICE = _auto_device()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "rtdetr-l.pt"
IMGSZ       = 640           # 640 is optimal for RT-DETR; higher hurts CPU perf

# FIX: Lowered from 0.35 → 0.20 for CPU inference.
# On CPU, non-person objects (handbag, bottle, box) score 0.20–0.30.
# Persons still score 0.80+ so this does not increase person false positives.
CONF_THRESH = 0.20

# FIX: Raised from 0.20 → 0.45.
# 0.20 was suppressing small objects that slightly overlapped persons.
# 0.45 matches the FUSION_IOU_THRESH and is the standard NMS threshold.
IOU_THRESH  = 0.45

KEEP_CLASSES = {
    "person",
    "bottle", "cup", "backpack", "handbag",
    "bag", "sports ball",
    "chair", "couch", "tv", "laptop",
    "box", "clock", "vase", "book",
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

# ── Lighting Analysis ─────────────────────────────────────────────────────────
LIGHTING_BRIGHTNESS_THRESH = 80
LIGHTING_CONTRAST_THRESH   = 40

# ── CLAHE Parameters ──────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT   = 2.5
CLAHE_TILE_SIZE    = (8, 8)

# ── Dual-Stream Fusion ────────────────────────────────────────────────────────
FUSION_IOU_THRESH  = 0.45