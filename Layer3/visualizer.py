"""
Layer 3 — Visualizer
====================
Draws Layer 3 debug overlays on frames:
  - bin_region boxes (green)
  - bin_zone boxes   (yellow)
  - trash trails
  - entry event scores
  - active pair IDs
"""

import cv2
import numpy as np
from typing import List, Dict

from Layer2.track_state import TrackedObject
from Layer2.bin_tracker import TrackedBin
from .feature_extractor import BinInteractionFeatureExtractor, _compute_bin_regions, _centroid
from .config import (
    DEBUG_BIN_REGION_COLOR,
    DEBUG_BIN_ZONE_COLOR,
    DEBUG_TRAIL_COLOR,
    DEBUG_TEXT_COLOR,
    DEBUG_THICKNESS,
)


def draw_layer3_debug(
    frame:           np.ndarray,
    tracked_bins:    List[TrackedBin],
    tracked_objects: List[TrackedObject],
    extractor:       BinInteractionFeatureExtractor,
    sequences:       List[Dict],
) -> np.ndarray:
    """
    Full Layer 3 debug overlay.

    Args:
        frame:           BGR frame to annotate (modified in-place)
        tracked_bins:    current frame bin list from BinTracker
        tracked_objects: current frame object list from ByteTrackWrapper
        extractor:       the BinInteractionFeatureExtractor instance
        sequences:       output of extractor.update() this frame

    Returns:
        Annotated frame.
    """
    # ── 1. Bin regions and zones ──────────────────────────────────────────────
    for tb in tracked_bins:
        region, zone = _compute_bin_regions(tb.bbox)
        center       = _centroid(tb.bbox)

        # Zone outline (cyan-yellow)
        cv2.rectangle(
            frame,
            (int(zone[0]),   int(zone[1])),
            (int(zone[2]),   int(zone[3])),
            DEBUG_BIN_ZONE_COLOR, DEBUG_THICKNESS,
        )
        cv2.putText(
            frame, f"ZONE B{tb.bin_id}",
            (int(zone[0]) + 2, int(zone[1]) + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, DEBUG_BIN_ZONE_COLOR, 1, cv2.LINE_AA,
        )

        # Region outline (green)
        cv2.rectangle(
            frame,
            (int(region[0]), int(region[1])),
            (int(region[2]), int(region[3])),
            DEBUG_BIN_REGION_COLOR, DEBUG_THICKNESS,
        )
        cv2.putText(
            frame, f"REGION B{tb.bin_id}",
            (int(region[0]) + 2, int(region[3]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, DEBUG_BIN_REGION_COLOR, 1, cv2.LINE_AA,
        )

        # Center cross
        cx, cy = int(center[0]), int(center[1])
        cv2.drawMarker(frame, (cx, cy), DEBUG_BIN_REGION_COLOR,
                       cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

    # ── 2. Trash trails ───────────────────────────────────────────────────────
    trash_objs = [o for o in tracked_objects if o.is_trash or o.class_name == "trash"]
    for obj in trash_objs:
        pts = list(obj.trail)
        for i in range(1, len(pts)):
            cv2.line(
                frame,
                (int(pts[i-1][0]), int(pts[i-1][1])),
                (int(pts[i][0]),   int(pts[i][1])),
                DEBUG_TRAIL_COLOR, 1, cv2.LINE_AA,
            )

    # ── 3. Active pair annotations ────────────────────────────────────────────
    active_pairs = {s["pair_id"]: s for s in sequences}
    for obj in trash_objs:
        # Find any sequence belonging to this trash object
        for pair_id, seq_dict in active_pairs.items():
            if f"trash_{obj.track_id}_" in pair_id and seq_dict["sequence"]:
                fv     = seq_dict["sequence"][-1]
                cx, cy = _centroid(obj.bbox)
                label  = (
                    f"T{obj.track_id} "
                    f"d:{fv[0]:.0f} "
                    f"zone:{int(fv[1])} "
                    f"reg:{int(fv[2])} "
                    f"entry:{fv[7]:.2f}"
                )
                # Background rectangle for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                cv2.rectangle(
                    frame,
                    (int(cx) - 2, int(cy) - th - 16),
                    (int(cx) + tw + 2, int(cy) - 14),
                    (30, 30, 30), -1,
                )
                cv2.putText(
                    frame, label,
                    (int(cx), int(cy) - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, DEBUG_TEXT_COLOR, 1, cv2.LINE_AA,
                )

    # ── 4. Sequence count overlay ─────────────────────────────────────────────
    cv2.putText(
        frame, f"L3 pairs: {len(sequences)}",
        (8, frame.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, DEBUG_TEXT_COLOR, 1, cv2.LINE_AA,
    )

    return frame