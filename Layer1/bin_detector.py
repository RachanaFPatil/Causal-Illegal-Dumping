"""
Layer 1 — Trash Bin Detector
Wraps the fine-tuned RT-DETR model (nc=1, class: trash_bin).

Runs ONLY when bins are present in the frame — zero cost on non-bin videos.
Kept completely separate from RTDETRDetector so Layer 1's existing logic
is untouched.

FIX: Added geometry guards to prevent false positives on car bodies.
  The fine-tuned trash_bin model has low specificity on tall rectangular
  objects like car doors/windows (confirmed: screenshot shows entire car
  flagged as BIN #9000 conf=0.95).

  Guards added:
  1. ASPECT_RATIO: real trash bins are taller than wide (h/w > 0.8).
     A car door is very wide (h/w ~ 0.4-0.6). Reject if h/w < 0.5.
  2. MIN_AREA: tiny bins don't exist on street CCTV. Reject < 1500px².
  3. MAX_ASPECT: an extremely tall thin detection is also rejected (> 4.0)
     — it is a pole or a person, not a bin.
  4. MIN_DIMENSION: both w and h must be at least 40px to be a real bin.

  These are conservative guards that should not reject real bins
  (a typical street bin is ~80×150px on a CCTV frame = h/w ~1.9).
"""

from ultralytics import RTDETR
import numpy as np
from dataclasses import dataclass
from typing import List

# Path to fine-tuned weights (relative to project root)
BIN_MODEL_PATH  = "weights/trash_bin_detector.pt"
BIN_CLASS_NAME  = "trash_bin"
BIN_CONF_THRESH = 0.82
BIN_IOU_THRESH  = 0.35
BIN_IMGSZ       = 480

# ── Geometry guards (FIX) ─────────────────────────────────────────────────────
# Real street trash bins: tall, roughly 0.8–3.0 h/w aspect ratio
BIN_MIN_ASPECT  = 0.50   # h/w — reject wider-than-tall detections (cars, fences)
BIN_MAX_ASPECT  = 5.00   # h/w — reject extremely thin detections (poles, people)
BIN_MIN_W_PX    = 40     # minimum width in pixels
BIN_MIN_H_PX    = 40     # minimum height in pixels
BIN_MIN_AREA    = 1600   # minimum area in pixels² (40×40)


@dataclass
class BinDetection:
    """Single trash-bin detection from the fine-tuned model."""
    bbox:       np.ndarray   # [x1, y1, x2, y2]
    confidence: float
    class_name: str = BIN_CLASS_NAME


class BinDetector:
    """
    Wraps the fine-tuned trash_bin RT-DETR model.

    Usage:
        detector = BinDetector()
        bins = detector.detect(frame)   # List[BinDetection]

    Returns [] if no bins detected — no overhead on bin-free frames.
    """

    def __init__(self, device: str = "cpu"):
        print(f"[BinDetector] Loading fine-tuned model: {BIN_MODEL_PATH} on {device}")
        self.model  = RTDETR(BIN_MODEL_PATH)
        self.device = device
        print(f"[BinDetector] Ready — detects class: '{BIN_CLASS_NAME}'")

    def detect(self, frame: np.ndarray) -> List[BinDetection]:
        """
        Run fine-tuned model on full frame.
        Returns list of BinDetection (may be empty).
        """
        results = self.model.predict(
            source  = frame,
            imgsz   = BIN_IMGSZ,
            conf    = BIN_CONF_THRESH,
            iou     = BIN_IOU_THRESH,
            device  = self.device,
            verbose = False,
        )

        detections: List[BinDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])

                # ── Confidence guard ──────────────────────────────────────
                if conf < BIN_CONF_THRESH:
                    continue

                # ── Dimension guards (FIX) ────────────────────────────────
                if w < BIN_MIN_W_PX or h < BIN_MIN_H_PX:
                    continue

                # ── Area guard (FIX) ──────────────────────────────────────
                area = w * h
                if area < BIN_MIN_AREA:
                    continue

                # ── Aspect ratio guard (FIX) ──────────────────────────────
                aspect = h / max(w, 1)   # h/w; bins are taller than wide
                if aspect < BIN_MIN_ASPECT:
                    # Too wide — likely a car body, fence, or wall
                    print(f"[BinDetector] REJECT aspect={aspect:.2f} "
                          f"(h={h:.0f} w={w:.0f}) — too wide, likely not a bin")
                    continue
                if aspect > BIN_MAX_ASPECT:
                    # Too tall and thin — pole or person
                    print(f"[BinDetector] REJECT aspect={aspect:.2f} "
                          f"(h={h:.0f} w={w:.0f}) — too thin, likely not a bin")
                    continue

                detections.append(BinDetection(
                    bbox       = box.xyxy[0].cpu().numpy(),
                    confidence = conf,
                ))
        return detections