"""
Layer 2 — Track State

Extended version that includes KalmanFilter2D required by the production
ByteTrackWrapper in tracker.py.

FIX (CRITICAL): Added last_innovation property to KalmanFilter2D.
  tracker.py line 184 calls self._kalman.last_innovation inside
  compute_uncertainty(). Without this attribute, every call to
  score_tracks() raises AttributeError, crashing ByteTrackWrapper.update()
  at Stage 6 — BEFORE Stage 9 builds the output list — so the tracker
  returns an empty list every single frame once any confirmed track exists.
  This is why handbag/bottle/object tracks never reach Layer 4 or 5.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


# ══════════════════════════════════════════════════════════════════════════════
#  Kalman Filter for 2D centroid tracking
#  State vector: [cx, cy, vx, vy]
#  Measurement:  [cx, cy]
# ══════════════════════════════════════════════════════════════════════════════

class KalmanFilter2D:
    """
    Lightweight constant-velocity Kalman filter for 2D centroid tracking.

    State  : [cx, cy, vx, vy]  (position + velocity)
    Measure: [cx, cy]

    Used by _InternalTrack in tracker.py to predict position during lost frames
    and to smooth bbox updates.

    FIXED: Added self.last_innovation tracking in update().
           tracker.py compute_uncertainty() calls self._kalman.last_innovation
           on every confirmed track. Without it, AttributeError crashes the
           entire ByteTrackWrapper.update() call at Stage 6, causing the
           tracker to return [] for every frame once tracks are confirmed.
    """

    def __init__(self, cx: float, cy: float):
        # State vector: [cx, cy, vx, vy]
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (we observe cx, cy only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        q = 1e-2
        self.Q = np.eye(4, dtype=np.float64) * q

        # Measurement noise covariance
        r = 1e-1
        self.R = np.eye(2, dtype=np.float64) * r

        # Error covariance matrix
        self.P = np.eye(4, dtype=np.float64) * 10.0

        # FIX: Store innovation magnitude so tracker.py compute_uncertainty()
        # can access self._kalman.last_innovation without AttributeError.
        # Innovation = distance between predicted and measured position (px).
        self.last_innovation: float = 0.0

    def predict(self) -> np.ndarray:
        """Predict next state. Returns predicted [cx, cy]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, cx: float, cy: float) -> np.ndarray:
        """Update with a new measurement. Returns corrected [cx, cy]."""
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.x                         # innovation vector
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # FIX: record scalar innovation magnitude (pixels) for uncertainty scoring
        self.last_innovation = float(np.linalg.norm(y))

        return self.x[:2].copy()

    @property
    def position(self) -> np.ndarray:
        """Current estimated [cx, cy]."""
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated [vx, vy]."""
        return self.x[2:4].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedObject — public output of Layer 2, consumed by Layers 3–5
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObject:
    """
    A single object being tracked across frames.

    Produced by Layer 2, consumed by Layers 3, 4, 5.

    Fields marked 'production-only' are populated by the production
    ByteTrackWrapper but not by the simple tracker — downstream code
    must handle defaults gracefully.
    """
    track_id:    int              # stable ID across frames
    bbox:        np.ndarray       # [x1, y1, x2, y2] current position
    class_name:  str              # "person", "bottle", "trash", etc.
    confidence:  float            # detection confidence this frame

    # Trash tagging (set by _tag_trash in tracker)
    is_trash:    bool = False     # True if Layer 1/2 flagged this as dropped trash
    trash_label: str  = ""        # original class name e.g. "bottle"
    trash_how:   str  = ""        # "dropped" or "thrown"

    # Trajectory history — Layer 3 uses this
    trail: deque = field(
        default_factory=lambda: deque(maxlen=30),
        repr=False,
    )

    # Production-only fields (populated by ByteTrackWrapper, default-safe)
    age:               int   = 0
    missed_frames:     int   = 0
    confirmed:         bool  = True
    locked_class:      Optional[str]   = None
    velocity:          np.ndarray      = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    status:            str   = "ACTIVE"
    uncertainty_score: float = 0.0

    def centroid(self):
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2),
        )

    def update_trail(self):
        """Call once per frame after bbox is updated."""
        self.trail.append(self.centroid())