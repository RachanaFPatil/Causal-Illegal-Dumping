"""
Layer 2 — Bin Tracker
=====================
Tracks trash_bin detections across frames using centroid distance matching.
Enforces two guarantees:
  1. A permanent ID is only assigned at confirmation (BIN_MIN_FRAMES).
  2. Once an ID is assigned to a spatial position, NO other bin at a
     different position can ever receive that same ID — enforced via a
     confirmed-centroid registry.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque

from Layer1.bin_detector import BinDetection

# ── Config ────────────────────────────────────────────────────────────────────

BIN_MATCH_DIST_PX    = 60     # centroid distance to match a detection to a track
BIN_REGISTRY_DIST_PX = 80      # min distance between two confirmed bin positions
                                 # — new track closer than this to a confirmed bin
                                 # is treated as the SAME bin, not a new one
BIN_MIN_FRAMES         = 12     # frames before a bin gets a permanent ID
BIN_MAX_LOST_FRAMES    = 240    # ~8s @30fps — survive long occlusions
BIN_ID_START           = 9000   # no collision with ByteTrack IDs


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


# ── Internal track ────────────────────────────────────────────────────────────

class _BinTrack:
    """
    bin_id is None until the track is confirmed.
    Once confirmed, the ID is permanent and tied to a spatial centroid
    stored in BinTracker._confirmed_registry.
    """
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
        """Assign next available permanent ID. Call only once."""
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

    def centroid(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centroid_dist(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    cx_a = (bbox_a[0] + bbox_a[2]) / 2;  cy_a = (bbox_a[1] + bbox_a[3]) / 2
    cx_b = (bbox_b[0] + bbox_b[2]) / 2;  cy_b = (bbox_b[1] + bbox_b[3]) / 2
    return float(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5)


def _pt_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _match_score(track_bbox: np.ndarray, det_bbox: np.ndarray) -> float:
    """
    Centroid-primary matching — bins don't move so centroid is the
    most reliable cue even when IoU collapses under occlusion.
    """
    dist = _centroid_dist(track_bbox, det_bbox)
    if dist <= BIN_MATCH_DIST_PX:
        # Score is inverse-distance: closer = higher score
        return 1.0 - (dist / BIN_MATCH_DIST_PX)
    return 0.0


# ── Main BinTracker ───────────────────────────────────────────────────────────

class BinTracker:
    """
    _confirmed_registry: Dict[bin_id -> centroid (cx, cy)]
    Once an ID is registered here, NO new track within BIN_REGISTRY_DIST_PX
    of that centroid will ever receive a different ID — it will be suppressed
    or re-associated to the existing confirmed track.
    """

    def __init__(self):
        self._tracks: List[_BinTrack]                  = []
        self._confirmed_registry: Dict[int, Tuple[float, float]] = {}
        self.total_bins_flagged: int                   = 0
        self._flagged_ids: set                         = set()

    # ── Internal: check if a detection falls inside an already-confirmed zone ─

    def _find_registry_match(self, det_bbox: np.ndarray) -> Optional[int]:
        """
        Returns the bin_id of a confirmed bin whose centroid is within
        BIN_REGISTRY_DIST_PX of this detection, or None if no match.
        This prevents a new unconfirmed track from spawning near an existing
        confirmed bin and eventually stealing or duplicating its ID.
        """
        cx = (det_bbox[0] + det_bbox[2]) / 2
        cy = (det_bbox[1] + det_bbox[3]) / 2
        for bid, reg_centroid in self._confirmed_registry.items():
            if _pt_dist((cx, cy), reg_centroid) <= BIN_REGISTRY_DIST_PX:
                return bid
        return None

    def update(self, detections: List[BinDetection]) -> List[TrackedBin]:
        used_det_indices = set()

        # ── Step 1: Match detections to existing tracks (centroid-primary) ────
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

            # ⚠️ Spatial registry guard — if this detection is near a
            # confirmed bin's known position, do NOT create a new track.
            # It means the confirmed track is temporarily lost (occlusion)
            # and this detection will re-associate once the track recovers.
            existing_id = self._find_registry_match(det.bbox)
            if existing_id is not None:
                # Find the lost confirmed track and revive it directly
                for bin_track in self._tracks:
                    if bin_track.bin_id == existing_id:
                        bin_track.update(det)
                        used_det_indices.add(i)
                        break
                # If confirmed track was already pruned, skip — avoids ghost
                continue

            self._tracks.append(_BinTrack(det))

        # ── Step 3: Remove tracks lost too long ───────────────────────────────
        self._tracks = [t for t in self._tracks
                        if t.frames_lost <= BIN_MAX_LOST_FRAMES]

        # ── Step 4: Confirm + register + flag ────────────────────────────────
        for bin_track in self._tracks:
            if not bin_track.confirmed:
                continue

            # Assign permanent ID exactly once
            if bin_track.bin_id is None:
                # Final spatial check — don't assign a new ID if another
                # confirmed bin already owns this position
                existing_id = self._find_registry_match(bin_track.bbox)
                if existing_id is not None:
                    # Merge into existing confirmed bin — suppress this track
                    bin_track.bin_id = existing_id
                else:
                    bin_track.assign_id()
                    # Register this bin's centroid permanently
                    self._confirmed_registry[bin_track.bin_id] = bin_track.centroid()
                    print(f"[BinTracker] 📍 Bin registered — "
                          f"ID #{bin_track.bin_id} at {bin_track.centroid()}")

            # Flag once per ID
            if not bin_track.flagged:
                bin_track.flagged = True
                if bin_track.bin_id not in self._flagged_ids:
                    self._flagged_ids.add(bin_track.bin_id)
                    self.total_bins_flagged += 1
                    print(f"[BinTracker] 🗑️  Bin flagged — "
                          f"ID #{bin_track.bin_id} "
                          f"(total flagged: {self.total_bins_flagged})")

        # ── Step 5: Output — confirmed, active tracks only ────────────────────
        output: List[TrackedBin] = []
        for bin_track in self._tracks:
            if not bin_track.confirmed or bin_track.frames_lost > 0:
                continue
            output.append(TrackedBin(
                bin_id      = bin_track.bin_id,
                bbox        = bin_track.bbox.copy(),
                confidence  = bin_track.confidence,
                flagged     = bin_track.flagged,
                frames_seen = bin_track.frames_seen,
                trail       = bin_track.trail,
            ))

        return output