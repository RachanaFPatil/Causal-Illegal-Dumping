"""
Layer 2 — Bin Tracker
=====================
Fix: Use bottom-center of bbox as matching anchor instead of full centroid.
     When a bin lid opens, the bbox top shifts upward but the BOTTOM EDGE
     stays fixed. This keeps the track stable through lid-open events so
     the bin keeps its original ID (e.g. #9001) throughout.
     BIN_MIN_FRAMES kept at 7 (user-tuned to avoid ghost detections).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque

from Layer1.bin_detector import BinDetection

# ── Config ────────────────────────────────────────────────────────────────────

BIN_MATCH_DIST_PX    = 60      # centroid distance to match — now uses BOTTOM-center
BIN_REGISTRY_DIST_PX = 80      # min distance between two confirmed bin BOTTOM-centers
BIN_MIN_FRAMES         = 7     # user-tuned: avoids ghost detections
BIN_MAX_LOST_FRAMES    = 240   # ~8s @30fps
BIN_ID_START           = 9000


# ── Anchor helper — BOTTOM-CENTER of bbox ─────────────────────────────────────

def _bottom_center(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Return the bottom-center point of a bbox [x1,y1,x2,y2].
    This point is STABLE even when a bin lid opens (lid extends top upward,
    bottom edge never moves).
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = float(bbox[3])          # bottom edge
    return (cx, cy)


def _pt_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)


def _match_score(track_bbox: np.ndarray, det_bbox: np.ndarray) -> float:
    """
    Match using BOTTOM-CENTER distance.
    Lid-open shifts top of bbox upward but bottom stays put →
    bottom-center distance stays small → same track ID preserved.
    Two separate physical bins have different bottom-centers → never merge.
    """
    dist = _pt_dist(_bottom_center(track_bbox), _bottom_center(det_bbox))
    if dist <= BIN_MATCH_DIST_PX:
        return 1.0 - (dist / BIN_MATCH_DIST_PX)
    return 0.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TrackedBin:
    bin_id:      int
    bbox:        np.ndarray
    confidence:  float
    flagged:     bool
    frames_seen: int
    trail: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)

    def centroid(self):
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)


class _BinTrack:
    _next_id: int = BIN_ID_START

    def __init__(self, det: BinDetection):
        self.bin_id: Optional[int] = None
        self.bbox        = det.bbox.copy()
        self.confidence  = det.confidence
        self.frames_seen = 1
        self.frames_lost = 0
        self.flagged     = False
        self.trail: deque = deque(maxlen=30)
        self._update_trail()

    def assign_id(self) -> int:
        if self.bin_id is None:
            self.bin_id        = _BinTrack._next_id
            _BinTrack._next_id += 1
        return self.bin_id

    def update(self, det: BinDetection):
        self.bbox        = det.bbox.copy()
        self.confidence  = det.confidence
        self.frames_seen += 1
        self.frames_lost = 0
        self._update_trail()

    def mark_lost(self):
        self.frames_lost += 1

    def _update_trail(self):
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        self.trail.append((cx, cy))

    @property
    def confirmed(self) -> bool:
        return self.frames_seen >= BIN_MIN_FRAMES

    def bottom_center(self) -> Tuple[float, float]:
        return _bottom_center(self.bbox)

    def centroid(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)


# ── Main BinTracker ───────────────────────────────────────────────────────────

class BinTracker:
    """
    Registry uses BOTTOM-CENTER as the spatial anchor.
    Two confirmed bins with different bottom-centers never merge.
    Same bin with lid open keeps same bottom-center → same ID.
    """

    def __init__(self):
        self._tracks: List[_BinTrack]                         = []
        self._confirmed_registry: Dict[int, Tuple[float, float]] = {}  # bin_id → bottom_center
        self.total_bins_flagged: int                          = 0
        self._flagged_ids: set                                = set()

    def _find_registry_match(self, det_bbox: np.ndarray) -> Optional[int]:
        """
        Check if this detection's bottom-center is near a confirmed bin's
        registered bottom-center. Uses BIN_REGISTRY_DIST_PX.
        """
        bc = _bottom_center(det_bbox)
        for bid, reg_bc in self._confirmed_registry.items():
            if _pt_dist(bc, reg_bc) <= BIN_REGISTRY_DIST_PX:
                return bid
        return None

    def update(self, detections: List[BinDetection]) -> List[TrackedBin]:
        used_det_indices = set()

        # ── Step 1: Match detections to existing tracks (bottom-center) ───────
        for bin_track in self._tracks:
            best_score = 0.0
            best_idx   = -1
            for i, det in enumerate(detections):
                if i in used_det_indices:
                    continue
                score = _match_score(bin_track.bbox, det.bbox)
                if score > best_score:
                    best_score = score
                    best_idx   = i
            if best_idx >= 0 and best_score > 0.0:
                bin_track.update(detections[best_idx])
                used_det_indices.add(best_idx)
            else:
                bin_track.mark_lost()

        # ── Step 2: New tracks for unmatched detections ───────────────────────
        for i, det in enumerate(detections):
            if i in used_det_indices:
                continue
            existing_id = self._find_registry_match(det.bbox)
            if existing_id is not None:
                for bin_track in self._tracks:
                    if bin_track.bin_id == existing_id:
                        bin_track.update(det)
                        used_det_indices.add(i)
                        break
                continue
            self._tracks.append(_BinTrack(det))

        # ── Step 3: Remove tracks lost too long ───────────────────────────────
        self._tracks = [t for t in self._tracks
                        if t.frames_lost <= BIN_MAX_LOST_FRAMES]

        # ── Step 4: Confirm + register + flag ────────────────────────────────
        for bin_track in self._tracks:
            if not bin_track.confirmed:
                continue
            if bin_track.bin_id is None:
                existing_id = self._find_registry_match(bin_track.bbox)
                if existing_id is not None:
                    bin_track.bin_id = existing_id
                else:
                    bin_track.assign_id()
                    # Register using BOTTOM-CENTER as the stable spatial anchor
                    self._confirmed_registry[bin_track.bin_id] = bin_track.bottom_center()
                    print(f"[BinTracker] 📍 Bin registered — "
                          f"ID #{bin_track.bin_id} "
                          f"bottom-center={bin_track.bottom_center()}")
            if not bin_track.flagged:
                bin_track.flagged = True
                if bin_track.bin_id not in self._flagged_ids:
                    self._flagged_ids.add(bin_track.bin_id)
                    self.total_bins_flagged += 1
                    print(f"[BinTracker] 🗑️  Bin flagged — "
                          f"ID #{bin_track.bin_id} "
                          f"(total flagged: {self.total_bins_flagged})")

        # ── Step 5: Output confirmed tracks ──────────────────────────────────
        # Rule: if two confirmed bins are within BIN_REGISTRY_DIST_PX of each
        # other, suppress the one that is currently LOST (frames_lost > 0).
        # If both are active, suppress the one with fewer frames_seen.
        # This keeps the real physical bin visible and hides the phantom.

        confirmed = [t for t in self._tracks if t.confirmed]
        suppressed_ids: set = set()

        for i, a in enumerate(confirmed):
            for j, b in enumerate(confirmed):
                if i >= j:
                    continue
                if _pt_dist(a.bottom_center(), b.bottom_center()) > BIN_REGISTRY_DIST_PX:
                    continue
                # a and b are too close — suppress the weaker one
                # Prefer active (frames_lost==0) over lost
                # Among equal, prefer more frames_seen
                a_score = (0 if a.frames_lost == 0 else -1) * 10000 + a.frames_seen
                b_score = (0 if b.frames_lost == 0 else -1) * 10000 + b.frames_seen
                if a_score >= b_score:
                    suppressed_ids.add(b.bin_id)
                else:
                    suppressed_ids.add(a.bin_id)

        output: List[TrackedBin] = []
        for bin_track in confirmed:
            if bin_track.bin_id in suppressed_ids:
                continue
            output.append(TrackedBin(
                bin_id      = bin_track.bin_id,
                bbox        = bin_track.bbox.copy(),
                confidence  = bin_track.confidence if bin_track.frames_lost == 0
                              else max(0.3, bin_track.confidence - 0.05 * bin_track.frames_lost),
                flagged     = bin_track.flagged,
                frames_seen = bin_track.frames_seen,
                trail       = bin_track.trail,
            ))
        return output