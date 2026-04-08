# Layer4/visualizer.py
"""
Layer 4 — Visualizer

Draws model predictions on top of the Layer 3 visualisation.
Each active pair gets a probability bar + violation overlay.
"""

import cv2
import numpy as np
from typing import List, Tuple

from Layer3.pair_state import PairState
from .config import DUMP_THRESHOLD

SAFE_COLOR   = (0,  220,   0)
WARN_COLOR   = (0,  165, 255)
VIOL_COLOR   = (0,    0, 255)
BAR_BG_COLOR = (50,  50,  50)


def _prob_color(prob: float) -> Tuple[int, int, int]:
    if prob >= DUMP_THRESHOLD:
        return VIOL_COLOR
    if prob >= DUMP_THRESHOLD * 0.7:
        return WARN_COLOR
    return SAFE_COLOR


def draw_predictions(
    frame: np.ndarray,
    pairs: List[PairState],
    predictions: dict,
) -> np.ndarray:
    """
    Args:
        frame       : BGR frame (already has Layer 3 drawings)
        pairs       : active pairs from Layer 3
        predictions : {pair_key: float}  from DumpingInference
    """
    H, W = frame.shape[:2]
    keys = list(predictions.keys())

    for pair in pairs:
        key  = pair.pair_key
        prob = predictions.get(key, -1.0)
        if prob < 0:
            continue

        color   = _prob_color(prob)
        is_viol = prob >= DUMP_THRESHOLD
        label   = f"P{pair.person_id}→O{pair.object_id}  {prob:.0%}"

        # Probability bar
        bx, by    = 10, 60 + keys.index(key) * 28
        bw, bh    = 200, 18
        filled    = int(bw * prob)

        cv2.rectangle(frame, (bx, by),            (bx + bw, by + bh), BAR_BG_COLOR, -1)
        cv2.rectangle(frame, (bx, by),            (bx + filled, by + bh), color, -1)
        cv2.putText(frame, label, (bx + bw + 6, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Violation overlay
        if is_viol:
            cv2.rectangle(frame, (0, 0), (W, H), VIOL_COLOR, 4)
            cv2.putText(frame, f"DUMPING DETECTED — Person {pair.person_id}",
                        (10, H - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, VIOL_COLOR, 2, cv2.LINE_AA)

    return frame