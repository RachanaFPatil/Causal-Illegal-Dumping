"""
Layer 2 — ROI Recovery Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles localised re-detection and track recovery for high-uncertainty tracks.

Design principles:
  • Event-triggered ONLY — never called every frame
  • Does NOT modify Layer 1 detector (calls it as-is on ROI crops)
  • Does NOT classify dumping events (pure perception recovery)
  • Restores original track_id if a match is found
  • Treats Layer 1 detector as a black-box callable

Pipeline per uncertain track:
  Step 1 — Generate ROI     : expand last-known bbox + velocity direction
  Step 2 — Local re-detect  : call Layer 1 on cropped ROI only
  Step 3 — Re-associate     : motion proximity + IoU fallback
  Step 4 — Recover track    : restore ID, reset missed_frames → ACTIVE

Public API:
    recovery = ROIRecoveryModule(detector)
    recovered = recovery.recover(uncertain_tracks, frame)
    # recovered : List[RecoveryResult]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

from Layer1.detector import Detection
from .config import (
    ROI_EXPANSION_FACTOR,
    ROI_VELOCITY_SCALE,
    ROI_MIN_SIZE,
    ROI_MAX_SIZE,
    ROI_MATCH_IOU_THRESH,
    ROI_MATCH_DIST_THRESH,
    TRACK_HIGH_THRESH,
)

if TYPE_CHECKING:
    from .track_state import TrackedObject

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RecoveryResult:
    """
    Outcome of one ROI recovery attempt.

    track_id     : ID of the track that was being recovered
    recovered    : True if a matching detection was found in the ROI
    new_bbox     : recovered bbox in FULL-FRAME coordinates (or None)
    confidence   : detection confidence of the recovered bbox (or 0.0)
    """
    track_id:   int
    recovered:  bool
    new_bbox:   Optional[np.ndarray] = None
    confidence: float = 0.0


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _generate_roi(
    bbox:     np.ndarray,
    velocity: np.ndarray,
    frame_h:  int,
    frame_w:  int,
) -> Tuple[int, int, int, int]:
    """
    Generate an expanded ROI around the last known bbox.

    Expansion:
      - Uniform padding: (ROI_EXPANSION_FACTOR - 1) / 2 of bbox size each side
      - Directional shift: velocity * ROI_VELOCITY_SCALE in the direction of motion

    Returns:
        (x1, y1, x2, y2) clipped to frame boundaries.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Uniform expansion
    pad_x = max((w * (ROI_EXPANSION_FACTOR - 1)) / 2.0, 0)
    pad_y = max((h * (ROI_EXPANSION_FACTOR - 1)) / 2.0, 0)

    # Directional shift in velocity direction
    vx, vy = float(velocity[0]), float(velocity[1])
    shift_x = vx * ROI_VELOCITY_SCALE
    shift_y = vy * ROI_VELOCITY_SCALE

    rx1 = int(x1 - pad_x + shift_x)
    ry1 = int(y1 - pad_y + shift_y)
    rx2 = int(x2 + pad_x + shift_x)
    ry2 = int(y2 + pad_y + shift_y)

    # Clamp to frame
    rx1 = max(0, min(rx1, frame_w - ROI_MIN_SIZE))
    ry1 = max(0, min(ry1, frame_h - ROI_MIN_SIZE))
    rx2 = min(frame_w, max(rx2, rx1 + ROI_MIN_SIZE))
    ry2 = min(frame_h, max(ry2, ry1 + ROI_MIN_SIZE))

    # Cap at ROI_MAX_SIZE
    if rx2 - rx1 > ROI_MAX_SIZE:
        cx = (rx1 + rx2) // 2
        rx1 = cx - ROI_MAX_SIZE // 2
        rx2 = cx + ROI_MAX_SIZE // 2

    if ry2 - ry1 > ROI_MAX_SIZE:
        cy = (ry1 + ry2) // 2
        ry1 = cy - ROI_MAX_SIZE // 2
        ry2 = cy + ROI_MAX_SIZE // 2

    return rx1, ry1, rx2, ry2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] bboxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _centroid_dist(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return float(np.linalg.norm(ca - cb))


# ── Main Recovery Module ──────────────────────────────────────────────────────

class ROIRecoveryModule:
    """
    Localised re-detection and track recovery.

    Injected with the Layer 1 detector at construction.
    Called ONLY when the tracker signals high uncertainty.
    """

    def __init__(self, detector):
        """
        Args:
            detector: RTDETRDetector instance from Layer 1.
                      We call detector.detect(roi_crop) — no modification.
        """
        self._detector = detector
        logger.info("[ROIRecovery] Module initialised.")

    # ─────────────────────────────────────────────────────────────────────────

    def recover(
        self,
        uncertain_tracks: "List[TrackedObject]",
        frame:            np.ndarray,
    ) -> List[RecoveryResult]:
        """
        Attempt recovery for each high-uncertainty track.

        For each track:
          1. Generate ROI crop around last known position (+ velocity offset)
          2. Re-run Layer 1 detector on the crop only
          3. Re-associate best detection with the lost track
          4. Return recovery results (tracker applies the actual state changes)

        Args:
            uncertain_tracks: tracks whose uncertainty_score >= threshold
            frame:            current full-resolution BGR frame

        Returns:
            List[RecoveryResult] — one entry per uncertain track.
            Caller (tracker) uses results to restore track state.
        """
        if not uncertain_tracks:
            return []

        H, W = frame.shape[:2]
        results: List[RecoveryResult] = []

        for track in uncertain_tracks:
            result = self._recover_single(track, frame, H, W)
            results.append(result)
            if result.recovered:
                logger.debug(
                    "[ROIRecovery] track_id=%d RECOVERED  bbox=%s  conf=%.2f",
                    track.track_id, result.new_bbox, result.confidence,
                )
            else:
                logger.debug(
                    "[ROIRecovery] track_id=%d NOT recovered.", track.track_id
                )

        return results

    # ─────────────────────────────────────────────────────────────────────────

    def _recover_single(
        self,
        track:   "TrackedObject",
        frame:   np.ndarray,
        frame_h: int,
        frame_w: int,
    ) -> RecoveryResult:
        """
        Single-track recovery pipeline.

        Step 1: Generate ROI
        Step 2: Crop frame → run Layer 1 detector
        Step 3: Translate detections back to full-frame coords
        Step 4: Re-associate best match with the lost track
        """

        # ── Step 1: ROI generation ────────────────────────────────────────────
        rx1, ry1, rx2, ry2 = _generate_roi(
            track.bbox, track.velocity, frame_h, frame_w
        )
        roi_crop = frame[ry1:ry2, rx1:rx2]

        if roi_crop.size == 0 or roi_crop.shape[0] < 8 or roi_crop.shape[1] < 8:
            return RecoveryResult(track_id=track.track_id, recovered=False)

        # ── Step 2: Re-run Layer 1 on the ROI crop ────────────────────────────
        try:
            roi_detections: List[Detection] = self._detector.detect(roi_crop)
        except Exception as exc:
            logger.warning("[ROIRecovery] Detector failed on ROI: %s", exc)
            return RecoveryResult(track_id=track.track_id, recovered=False)

        if not roi_detections:
            return RecoveryResult(track_id=track.track_id, recovered=False)

        # ── Step 3: Translate ROI-local coords → full-frame coords ────────────
        full_frame_dets: List[Detection] = []
        for d in roi_detections:
            # Filter by class — don't recover a person as a bottle etc.
            if track.locked_class and d.class_name != track.locked_class:
                continue
            # Translate bbox from ROI space → full frame space
            fx1 = d.bbox[0] + rx1
            fy1 = d.bbox[1] + ry1
            fx2 = d.bbox[2] + rx1
            fy2 = d.bbox[3] + ry1
            translated = Detection(
                bbox       = np.array([fx1, fy1, fx2, fy2], dtype=np.float32),
                class_name = d.class_name,
                confidence = d.confidence,
                class_id   = d.class_id,
            )
            full_frame_dets.append(translated)

        if not full_frame_dets:
            return RecoveryResult(track_id=track.track_id, recovered=False)

        # ── Step 4: Re-associate best detection ───────────────────────────────
        best_det:  Optional[Detection] = None
        best_score: float              = -1.0

        for det in full_frame_dets:
            iou  = _iou(track.bbox, det.bbox)
            dist = _centroid_dist(track.bbox, det.bbox)

            # Primary: IoU match
            if iou >= ROI_MATCH_IOU_THRESH and iou > best_score:
                best_score = iou
                best_det   = det
            # Fallback: proximity match (for partial overlaps)
            elif best_det is None and dist <= ROI_MATCH_DIST_THRESH:
                best_score = 1.0 / (1.0 + dist)  # higher score = closer
                best_det   = det

        if best_det is None:
            return RecoveryResult(track_id=track.track_id, recovered=False)

        return RecoveryResult(
            track_id   = track.track_id,
            recovered  = True,
            new_bbox   = best_det.bbox.copy(),
            confidence = best_det.confidence,
        )