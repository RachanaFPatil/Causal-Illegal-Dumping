"""
Layer 4 — Visualizer
Draws Layer 4 evaluation results onto the video frame.
"""

from typing import Dict, List
import cv2
import numpy as np

from .evaluator import EvaluationResult

# Colours
_COL_ILLEGAL = (0,   60, 255)   # red-orange
_COL_LEGAL   = (50, 220,  50)   # green
_COL_PENDING = (180, 180,  40)  # yellow
_COL_BG      = (20,  20,  20)


def draw_evaluations(
    frame:    np.ndarray,
    results:  List[EvaluationResult],
    show_reason: bool = False,
) -> np.ndarray:
    """
    Overlay Layer 4 results as text banners in the top-right corner.
    One row per active pair with a confirmed event.
    """
    visible = [r for r in results if r.event != "pending"]
    if not visible:
        return frame

    H, W = frame.shape[:2]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.55
    thick    = 1
    pad      = 6
    line_h   = 22
    x_margin = 8

    for i, res in enumerate(visible):
        colour = _COL_ILLEGAL if res.event == "illegal_dumping" else _COL_LEGAL
        label  = (
            f"{res.pair_id}  {res.event}  {res.confidence:.2f}"
        )
        if show_reason and res.reason:
            label += f"  [{res.reason}]"

        y = 32 + i * (line_h + 4)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        x = W - tw - x_margin - pad * 2

        # Background pill
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      _COL_BG, -1)
        cv2.putText(frame, label, (x, y), font, scale, colour, thick, cv2.LINE_AA)

    return frame
