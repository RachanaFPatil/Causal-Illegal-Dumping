"""
Layer 1 — Visualizer
UNCHANGED from baseline EXCEPT: the per-frame stats block (Persons / Objects /
Total) has been REMOVED from draw_detections().

Reason: Layer 2's run.py renders a single unified stats panel with cumulative
counts.  Keeping the L1 stats block caused text to overlap / overwrite the L2
panel regardless of y-position tricks.  All bounding-box and label drawing is
100% unchanged.
"""

import cv2
import numpy as np
from typing import List

from .detector import Detection
from .trash_detector import TrashDetection
from .config import (
    PERSON_COLOR, OBJECT_COLOR, TRASH_COLOR,
    TEXT_COLOR, BOX_THICKNESS, FONT_SCALE,
)


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Draw bounding boxes and labels for all detections.
    Returns annotated copy.

    NOTE: Per-frame stats (Persons / Objects / Total) are intentionally NOT
    drawn here.  The unified stats panel is rendered by Layer 2's run.py so
    that cumulative counts are shown in one clean, non-overlapping block.
    """
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = PERSON_COLOR if det.class_name == "person" else OBJECT_COLOR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return out


def draw_trash(frame: np.ndarray, trash_detections: List[TrashDetection]) -> np.ndarray:
    """Draw trash event boxes and labels. Unchanged."""
    for det in trash_detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        lbl = f"TRASH({det.how}): {det.label}" if det.label else "TRASH"
        cv2.putText(
            frame, lbl,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame