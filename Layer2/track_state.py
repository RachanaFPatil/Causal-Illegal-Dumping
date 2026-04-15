"""
Layer 2 — Track State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Represents a single object being tracked across frames.

Upgrades over baseline:
  • age            — total frames the track has existed
  • missed_frames  — consecutive frames without a matching detection
  • confirmed      — True once the track survives MIN_CONFIRM_FRAMES
  • velocity       — EMA-smoothed (vx, vy) in pixels/frame
  • locked_class   — set on confirmation; prevents class-name flip
  • predicted_bbox — linear extrapolation for occlusion bridging
  • trail maxlen   — increased to TRAIL_MAXLEN (60) for Layer 3

NEW — Failure Detection additions:
  • kalman_state   — 4-state Kalman filter [cx, cy, vx, vy]
  • uncertainty_score — internal [0,1] confidence score per track
  • status         — ACTIVE / LOST / RECOVERABLE (Layer 3 compatible)

Public API unchanged:
  • TrackedObject.centroid()
  • TrackedObject.update_trail()
  • All original fields: track_id, bbox, class_name, confidence,
    is_trash, trash_label, trash_how, trail
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque

from .config import (
    TRAIL_MAXLEN,
    VELOCITY_ALPHA,
    LOCK_CLASS_ON_CONFIRM,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    MOTION_ERROR_WEIGHT,
    MISSED_FRAMES_WEIGHT,
    SUDDEN_LOSS_WEIGHT,
    MISSED_FRAMES_SPIKE,
    MOTION_ERROR_THRESH_PX,
    UNCERTAINTY_THRESHOLD,
)


# ── Kalman Filter (lightweight 4-state constant-velocity model) ───────────────

class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for 2D centroid tracking.

    State vector: [cx, cy, vx, vy]
    Measurement:  [cx, cy]

    CPU-friendly: pure numpy, no external dependencies.
    """

    def __init__(self, cx: float, cy: float):
        dt = 1.0  # one frame timestep

        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # Measurement matrix (observe only position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        q = KALMAN_PROCESS_NOISE
        r = KALMAN_MEASUREMENT_NOISE

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * q
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * r

        # Initial state and covariance
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0

        # Last prediction error (pixels) — used for uncertainty scoring
        self.last_innovation: float = 0.0

    def predict(self) -> np.ndarray:
        """Predict next state. Returns predicted [cx, cy]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, cx: float, cy: float) -> None:
        """Correct state with measurement [cx, cy]."""
        z = np.array([cx, cy], dtype=np.float32)
        y = z - self.H @ self.x                        # innovation
        self.last_innovation = float(np.linalg.norm(y))

        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def kalman_velocity(self) -> np.ndarray:
        return self.x[2:4].copy()


# ── Public Track Object (consumed by Layer 3) ─────────────────────────────────

@dataclass
class TrackedObject:
    """
    A single object being tracked across frames.

    Produced by Layer 2 → consumed by Layer 3 (MemoryEngine).
    All fields that existed in the baseline are preserved exactly.
    New fields are additive and have safe defaults so Layer 3 code
    that does not reference them continues to work without change.
    """

    # ── Core identity (unchanged) ─────────────────────────────────────────────
    track_id:    int
    bbox:        np.ndarray       # [x1, y1, x2, y2] current position
    class_name:  str              # "person", "bottle", "trash", …
    confidence:  float

    # ── Trash tagging (unchanged) ─────────────────────────────────────────────
    is_trash:    bool = False
    trash_label: str  = ""
    trash_how:   str  = ""

    # ── Trajectory (maxlen increased; Layer 3 uses this) ─────────────────────
    trail: deque = field(
        default_factory=lambda: deque(maxlen=TRAIL_MAXLEN),
        repr=False,
    )

    # ── Lifecycle (new) ───────────────────────────────────────────────────────
    age:           int  = 0
    missed_frames: int  = 0
    confirmed:     bool = False

    # ── Status field (NEW — Layer 3 compatible, safe default) ─────────────────
    # Values: "ACTIVE" | "LOST" | "RECOVERABLE"
    status: str = "ACTIVE"

    # ── Class lock (new) ──────────────────────────────────────────────────────
    locked_class: Optional[str] = field(default=None, repr=False)

    # ── Velocity (new) ────────────────────────────────────────────────────────
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32),
        repr=False,
    )

    # ── Predicted position (internal) ─────────────────────────────────────────
    predicted_bbox: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Uncertainty score (NEW — internal, not consumed by Layer 3) ───────────
    # Float in [0, 1]. Above UNCERTAINTY_THRESHOLD → triggers ROI re-scan.
    uncertainty_score: float = 0.0

    # ── Kalman filter (NEW — internal) ────────────────────────────────────────
    _kalman: Optional[KalmanFilter2D] = field(default=None, repr=False)

    # ── Internal bookkeeping ──────────────────────────────────────────────────
    _prev_centroid:       Optional[np.ndarray] = field(default=None, repr=False)
    _sudden_loss_flag:    bool                 = field(default=False, repr=False)
    _consecutive_hits:    int                  = field(default=0,     repr=False)

    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self):
        """Initialise Kalman filter on the current bbox centroid."""
        if self._kalman is None:
            cx, cy = self.centroid()
            self._kalman = KalmanFilter2D(cx, cy)

    def centroid(self) -> Tuple[float, float]:
        """Returns (cx, cy) of the current bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def centroid_array(self) -> np.ndarray:
        """Returns centroid as float32 numpy array [cx, cy]."""
        return np.array(self.centroid(), dtype=np.float32)

    def update_trail(self) -> None:
        """
        Call once per frame after bbox is updated.
        Appends current centroid to trail AND updates velocity EMA.
        Also runs Kalman predict→update cycle.
        """
        cx, cy = self.centroid()
        curr = np.array([cx, cy], dtype=np.float32)

        # ── Kalman filter update ───────────────────────────────────────
        if self._kalman is None:
            self._kalman = KalmanFilter2D(cx, cy)
        self._kalman.predict()
        self._kalman.update(cx, cy)

        # ── Velocity EMA ──────────────────────────────────────────────
        if self._prev_centroid is not None:
            raw_vel       = curr - self._prev_centroid
            self.velocity = (
                VELOCITY_ALPHA * raw_vel
                + (1.0 - VELOCITY_ALPHA) * self.velocity
            )
        self._prev_centroid = curr

        # ── Trail append ──────────────────────────────────────────────
        self.trail.append((cx, cy))

    def kalman_predict(self) -> np.ndarray:
        """
        Run Kalman prediction step (used during LOST frames).
        Returns predicted [cx, cy].
        """
        if self._kalman is None:
            cx, cy = self.centroid()
            self._kalman = KalmanFilter2D(cx, cy)
        return self._kalman.predict()

    def predict_next_bbox(self) -> np.ndarray:
        """
        Linear extrapolation using EMA velocity: shift bbox by (vx, vy).
        Used by the tracker for motion-compensated matching.
        """
        vx, vy = float(self.velocity[0]), float(self.velocity[1])
        return self.bbox + np.array([vx, vy, vx, vy], dtype=np.float32)

    def confirm(self) -> None:
        """Mark track as confirmed and lock its class label."""
        if not self.confirmed:
            self.confirmed    = True
            self.locked_class = self.class_name

    def apply_class_lock(self, candidate_class: str) -> str:
        """
        If class-locking is enabled and the track is confirmed,
        return the locked class instead of the candidate.
        """
        if LOCK_CLASS_ON_CONFIRM and self.confirmed and self.locked_class:
            return self.locked_class
        return candidate_class

    def speed(self) -> float:
        """Scalar speed in pixels/frame (magnitude of velocity vector)."""
        return float(np.linalg.norm(self.velocity))

    # ── Uncertainty scoring ───────────────────────────────────────────────────

    def compute_uncertainty(self) -> float:
        """
        Compute internal uncertainty score in [0, 1].

        Three components:
          1. Motion prediction error from Kalman (vs last measurement)
          2. Consecutive missed frames
          3. Sudden loss flag (disappeared without crossing frame edge)

        Score above UNCERTAINTY_THRESHOLD → ROI re-scan should be triggered.
        This is an INTERNAL signal — not exported to Layer 3.
        """
        # Component 1: Kalman innovation (prediction error in pixels)
        if self._kalman is not None:
            raw_error = self._kalman.last_innovation
        else:
            raw_error = 0.0
        motion_component = min(raw_error / (MOTION_ERROR_THRESH_PX + 1e-6), 1.0)

        # Component 2: missed frames (normalised against spike threshold)
        missed_component = min(
            self.missed_frames / max(MISSED_FRAMES_SPIKE, 1), 1.0
        )

        # Component 3: sudden disappearance signal
        sudden_component = 1.0 if self._sudden_loss_flag else 0.0

        score = (
            MOTION_ERROR_WEIGHT  * motion_component
            + MISSED_FRAMES_WEIGHT * missed_component
            + SUDDEN_LOSS_WEIGHT   * sudden_component
        )
        self.uncertainty_score = float(np.clip(score, 0.0, 1.0))
        return self.uncertainty_score

    @property
    def is_uncertain(self) -> bool:
        """True when uncertainty score exceeds the configured threshold."""
        return self.uncertainty_score >= UNCERTAINTY_THRESHOLD

    def mark_sudden_loss(self) -> None:
        """
        Set the sudden-loss flag.  Called by the tracker when a confirmed
        track disappears without having a velocity that would carry it out
        of frame — indicates occlusion or fast motion, not a natural exit.
        """
        self._sudden_loss_flag = True

    def clear_sudden_loss(self) -> None:
        self._sudden_loss_flag = False