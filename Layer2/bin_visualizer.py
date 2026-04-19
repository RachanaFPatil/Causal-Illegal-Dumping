"""
Layer 2 — Bin Visualiser
Draws tracked trash bins on the frame.

Kept separate from the existing draw_tracks() so Layer 2's visualiser
is untouched and this can be composed freely.
"""

import cv2
import numpy as np
from typing import List

from Layer2.bin_tracker import TrackedBin

# ── Colours ───────────────────────────────────────────────────────────────────
BIN_BOX_CONFIRMED  = (0, 200, 255)    # Orange — confirmed, not yet flagged
BIN_BOX_FLAGGED    = (0, 80,  255)    # Red-orange — flagged bin
BIN_TEXT_COLOR     = (255, 255, 255)
BIN_TRAIL_COLOR    = (0, 160, 200)
BIN_THICKNESS      = 2
BIN_FONT_SCALE     = 0.52


def draw_bins(
    frame:        np.ndarray,
    tracked_bins: List[TrackedBin],
    total_flagged: int = 0,
) -> np.ndarray:
    """
    Draw tracked bin bounding boxes, IDs, and a stats overlay.

    Args:
        frame:         BGR frame (drawn in-place and returned).
        tracked_bins:  Output of BinTracker.update().
        total_flagged: Cumulative bin-flag count from BinTracker.

    Returns:
        Annotated BGR frame.
    """
    for tb in tracked_bins:
        x1, y1, x2, y2 = tb.bbox.astype(int)
        color = BIN_BOX_FLAGGED if tb.flagged else BIN_BOX_CONFIRMED

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BIN_THICKNESS)

        # Label: "BIN #9000 ✓" or "BIN #9001 (new)"
        flag_marker = " [FLAGGED]" if tb.flagged else ""
        label = f"BIN #{tb.bin_id}{flag_marker}  {tb.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       BIN_FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, BIN_FONT_SCALE,
                    BIN_TEXT_COLOR, 1, cv2.LINE_AA)

        # Trail dots
        pts = list(tb.trail)
        for i in range(1, len(pts)):
            cv2.line(frame,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     BIN_TRAIL_COLOR, 1, cv2.LINE_AA)

    # ── Stats overlay (top-right area, below FPS counter) ────────────────────
    if tracked_bins or total_flagged > 0:
        H, W = frame.shape[:2]
        stats = [
            f"Bins visible : {len(tracked_bins)}",
            f"Bins flagged : {total_flagged}",
        ]
        for i, txt in enumerate(stats):
            cv2.putText(frame, txt,
                        (W - 190, 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 200, 255), 1, cv2.LINE_AA)

    return frame
