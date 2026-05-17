"""
Layer 1 — Configuration
Robust CCTV AI Engine: adds lighting analysis and dual-stream CLAHE settings.

WINDOWS/CPU FIX: CONF_THRESH lowered 0.35 → 0.18 and IOU_THRESH raised
  to 0.45 for CPU inference.

  WHY CONF_THRESH=0.18 (was 0.20):
    Previous round found non-person objects (handbag) at 0.20–0.30 on CPU.
    Small thrown objects (tissue, small bag, cup) score even lower: 0.15–0.22.
    Lowering to 0.18 captures these without significantly increasing
    false positives (persons still score 0.75+).

  IOU_THRESH=0.45: standard NMS threshold, prevents over-suppression.

DEVICE auto-detection: 'mps' only on Apple Silicon Mac. Windows → cuda/cpu.
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
IMGSZ       = 640

# FIX: Lowered 0.35 → 0.18 for CPU inference.
# Handbag/bottle: 0.20–0.30 on CPU. Small thrown objects: 0.15–0.22.
CONF_THRESH = 0.18

# FIX: Raised 0.20 → 0.45 — standard NMS threshold.
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
TRASH_MIN_AREA        = 400    # FIX: lowered 600→400 to catch small thrown objects
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