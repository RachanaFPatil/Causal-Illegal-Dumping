"""
Layer 1 — Trash Bin Detector
Wraps the fine-tuned RT-DETR model (nc=1, class: trash_bin).

Runs ONLY when bins are present in the frame — zero cost on non-bin videos.
Kept completely separate from RTDETRDetector so Layer 1's existing logic
is untouched.
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
BIN_IMGSZ = 480   # was 640 — bins are large, don't need full resolution


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
                # Reject tiny detections — real bins are not small
                if w < 30 or h < 30:
                    continue
                if float(box.conf[0]) < BIN_CONF_THRESH:
                    continue
                detections.append(BinDetection(
                    bbox       = box.xyxy[0].cpu().numpy(),
                    confidence = float(box.conf[0]),
                ))
        return detections