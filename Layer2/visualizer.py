"""
Layer 2 — Visualizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes from previous version:
  • draw_tracks() now accepts cumulative_persons and cumulative_objects as
    optional arguments so the stats panel shows running totals (never resets).
  • Stats panel moved to y=24 (top-left) since Layer 1 no longer draws there.
  • All bounding-box, trail, velocity, and label drawing is UNCHANGED.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import numpy as np
from typing import List

from .track_state import TrackedObject
from .config import SHOW_TRAILS, SHOW_VELOCITY, SHOW_TRACK_STATUS


def _id_color(tid: int) -> tuple:
    """Deterministic, visually distinct color per track ID."""
    np.random.seed(tid * 7 + 13)
    return tuple(int(x) for x in np.random.randint(80, 230, 3))


def draw_tracks(
    frame:                np.ndarray,
    tracks:               List[TrackedObject],
    total_trash_events:   int = 0,
    cumulative_persons:   int = 0,
    cumulative_objects:   int = 0,
) -> np.ndarray:
    """
    Draw all confirmed tracks onto the frame.

    Args:
        frame:               BGR numpy array — modified in place.
        tracks:              List[TrackedObject] from ByteTrackWrapper.update().
        total_trash_events:  cumulative trash event count from tracker.
        cumulative_persons:  total unique person track IDs seen so far.
        cumulative_objects:  total unique non-person track IDs seen so far.

    Returns:
        Annotated frame (same array, drawn in place).
    """
    H, W = frame.shape[:2]

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ── Colour logic ──────────────────────────────────────────────────────
        if t.is_trash:
            color = (0, 0, 220)            # red → confirmed trash
        else:
            color = _id_color(t.track_id)  # unique per ID

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Label ─────────────────────────────────────────────────────────────
        if t.is_trash:
            label = f"ID:{t.track_id} TRASH({t.trash_how}):{t.trash_label}"
        else:
            label = f"ID:{t.track_id} {t.class_name} {t.confidence:.2f}"

        if SHOW_TRACK_STATUS:
            status = "C" if t.confirmed else "T"
            label  = f"[{status}] {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1 = max(x1, 0)
        ly1 = max(y1 - th - 6, 0)
        lx2 = min(x1 + tw + 4, W - 1)
        ly2 = max(y1, 0)

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(
            frame, label,
            (lx1 + 2, ly2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # ── Trail (fading dots) ───────────────────────────────────────────────
        if SHOW_TRAILS and len(t.trail) > 1:
            pts = list(t.trail)
            n   = len(pts)
            for i in range(1, n):
                alpha = i / n
                c     = tuple(int(ch * alpha) for ch in color)
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 2, c, -1)

        # ── Velocity arrow (optional debug) ───────────────────────────────────
        if SHOW_VELOCITY and hasattr(t, "velocity"):
            vx, vy = float(t.velocity[0]), float(t.velocity[1])
            scale  = 5.0
            end_x  = int(np.clip(cx + vx * scale, 0, W - 1))
            end_y  = int(np.clip(cy + vy * scale, 0, H - 1))
            cv2.arrowedLine(
                frame, (cx, cy), (end_x, end_y),
                (0, 255, 255), 1, tipLength=0.35,
            )

    # ── Unified cumulative stats panel (top-left, y=24/46/68) ────────────────
    # Layer 1 no longer draws here — this is the single stats block.
    # Values passed in are running totals and never decrease.
    stats = [
        f"Tracked persons : {cumulative_persons}",
        f"Tracked objects : {cumulative_objects}",
        f"Trash events    : {total_trash_events}",
    ]
    for i, txt in enumerate(stats):
        cv2.putText(
            frame, txt,
            (10, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 200, 255), 1, cv2.LINE_AA,
        )

    return frame