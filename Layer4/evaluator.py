"""
Layer 4 — Context-Aware Temporal Intent Evaluator
==================================================
Converts Layer 3 temporal sequences into final behavioral decisions:
  • "legal_disposal"  → object disposed near a bin
  • "illegal_dumping" → object thrown without proper disposal

Design constraints (STRICT):
  • NO deep learning (no LSTM / GRU / Transformer)
  • CPU-only, real-time
  • Does NOT modify Layer 1 / 2 / 3
  • Tolerates noisy tracking and missing frames
  • Works independently per pair_id
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from Layer2.bin_tracker import TrackedBin
from Layer2.track_state import TrackedObject

# ──────────────────────────────────────────────────────────────────────────────
#  Config  (tune here, touch nothing else)
# ──────────────────────────────────────────────────────────────────────────────

# --- Throw / release thresholds ---
MIN_HELD_FRAMES       = 3    # how many frames object must have been held
MIN_POST_RELEASE      = 4    # frames of independent motion after release
RELEASE_DIST_DELTA    = 20   # min pixel jump per frame to count as release motion
MIN_SEQ_LEN           = 6    # skip evaluation if sequence is too short

# --- Velocity / motion ---
MOTION_VEL_THRESHOLD  = 2.5  # px/frame — minimum object speed after release
THROW_VEL_THRESHOLD   = 5.0  # px/frame — strong throw signal

# --- Bin proximity ---
BIN_NEAR_PX           = 120  # object final position ≤ this from bin centroid → legal
BIN_PRESENT_FRAMES    = 3    # bin must appear in at least this many recent frames

# --- Confidence weights ---
W_HELD        = 0.20
W_RELEASE     = 0.35
W_MOTION      = 0.25
W_BIN_SPATIAL = 0.20

# --- Stale result eviction ---
MAX_RESULT_AGE = 60  # frames before a cached result is cleared

# ──────────────────────────────────────────────────────────────────────────────
#  Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    event:      str    # "legal_disposal" | "illegal_dumping" | "pending"
    confidence: float  # [0.0, 1.0]
    pair_id:    str
    reason:     str    = ""   # human-readable debug string

    def to_dict(self) -> Dict:
        return {"event": self.event, "confidence": round(self.confidence, 3)}


# ──────────────────────────────────────────────────────────────────────────────
#  Per-pair internal state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _PairEvalState:
    pair_id:            str
    held_frames:        int   = 0
    release_confirmed:  bool  = False
    post_release_count: int   = 0
    frames_since_update: int  = 0
    last_result: Optional[EvaluationResult] = None

    # Recent object centroid trail (after release, used for post-release motion)
    post_release_trail: deque = field(default_factory=lambda: deque(maxlen=30))


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean_velocity(trail: deque) -> float:
    """Average frame-to-frame speed over the trail."""
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    speeds = [
        math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        for i in range(1, len(pts))
    ]
    return sum(speeds) / len(speeds)


def _nearest_bin_dist(
    obj_centroid: Tuple[float, float],
    tracked_bins: List[TrackedBin],
) -> float:
    """Return distance to nearest confirmed bin, or inf if none."""
    if not tracked_bins:
        return float("inf")
    return min(_dist(obj_centroid, _centroid(tb.bbox)) for tb in tracked_bins)


# ──────────────────────────────────────────────────────────────────────────────
#  Feature vector accessors (Layer 3 sequence format)
#
#  Layer 3 (BinInteractionFeatureExtractor) feature indices:
#    0  distance_to_bin_center
#    1  is_in_zone
#    2  is_in_region
#    3  velocity_y
#    4  trajectory_slope
#    5  time_in_region
#    6  min_distance_to_bin
#    7  entry_event_score
# ──────────────────────────────────────────────────────────────────────────────

_F_DIST      = 0
_F_IN_ZONE   = 1
_F_IN_REGION = 2
_F_VEL_Y     = 3
_F_SLOPE     = 4
_F_T_REGION  = 5
_F_MIN_DIST  = 6
_F_ENTRY     = 7


def _seq_array(sequence: List[List[float]]) -> np.ndarray:
    """Convert list-of-lists to (T, F) numpy array."""
    return np.array(sequence, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 4 Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class TemporalIntentEvaluator:
    """
    Layer 4 — Context-Aware Temporal Intent Evaluator.

    Two integration modes (auto-detected from input):

    MODE A — Person↔Trash PairState (preferred, from your full pipeline):
        result = evaluator.evaluate_pair(
            pair_id          = "person_3_trash_7",
            held_frames      = pair_state.held_frames,
            release_confirmed= pair_state.release_confirmed,
            post_release_count=pair_state.post_release_count,
            sequence         = pair_state.sequence,          # (T, F) or List[List]
            obj_bbox         = trash_track.bbox,             # np.ndarray [x1,y1,x2,y2]
            tracked_bins     = tracked_bins,                 # List[TrackedBin]
        )

    MODE B — Layer 3 BinInteraction sequence dict (plug-and-play):
        result = evaluator.evaluate_sequence(
            seq_dict     = layer3_output_dict,   # {"pair_id", "sequence", "timestamps"}
            tracked_bins = tracked_bins,
        )

    Both return EvaluationResult with .to_dict() → {"event": ..., "confidence": ...}
    """

    def __init__(self):
        self._states: Dict[str, _PairEvalState] = {}

    # ── Public — MODE A ───────────────────────────────────────────────────────

    def evaluate_pair(
        self,
        pair_id:           str,
        held_frames:       int,
        release_confirmed: bool,
        post_release_count: int,
        sequence:          List[List[float]],
        obj_bbox:          np.ndarray,
        tracked_bins:      List[TrackedBin],
        obj_trail:         Optional[deque] = None,
    ) -> EvaluationResult:
        """
        Evaluate a person↔trash pair for illegal dumping.
        Call once per frame per active pair.
        """
        state = self._get_or_create(pair_id)
        state.held_frames         = held_frames
        state.release_confirmed   = release_confirmed
        state.post_release_count  = post_release_count
        state.frames_since_update = 0

        # --- Gate 1: release must be confirmed ---
        if not release_confirmed:
            return EvaluationResult("pending", 0.0, pair_id, "awaiting_release")

        # --- Gate 2: must have been held ---
        if held_frames < MIN_HELD_FRAMES:
            return EvaluationResult(
                "pending", 0.1, pair_id,
                f"held_frames={held_frames} < {MIN_HELD_FRAMES}"
            )

        # --- Gate 3: require post-release motion ---
        if post_release_count < MIN_POST_RELEASE:
            return EvaluationResult(
                "pending", 0.15, pair_id,
                f"post_release={post_release_count} < {MIN_POST_RELEASE}"
            )

        # --- Update post-release trail ---
        if obj_trail is not None:
            pts = list(obj_trail)
            for pt in pts[-post_release_count:]:
                state.post_release_trail.append(pt)
        else:
            obj_cx, obj_cy = _centroid(obj_bbox)
            state.post_release_trail.append((obj_cx, obj_cy))

        result = self._decide(
            state        = state,
            sequence     = sequence,
            obj_bbox     = obj_bbox,
            tracked_bins = tracked_bins,
        )
        state.last_result = result
        return result

    # ── Public — MODE B ───────────────────────────────────────────────────────

    def evaluate_sequence(
        self,
        seq_dict:     Dict,
        tracked_bins: List[TrackedBin],
    ) -> EvaluationResult:
        """
        Evaluate directly from a Layer 3 BinInteractionFeatureExtractor output dict.
        Infers hold/release signals from the feature sequence itself.
        """
        pair_id  = seq_dict.get("pair_id", "unknown")
        sequence = seq_dict.get("sequence", [])

        if len(sequence) < MIN_SEQ_LEN:
            return EvaluationResult("pending", 0.0, pair_id, "sequence_too_short")

        arr = _seq_array(sequence)

        # Infer held_frames from proximity/in-zone signal
        in_zone = arr[:, _F_IN_ZONE]
        held_frames = int(np.sum(in_zone > 0.5))

        # Infer release: was in zone, now out
        was_in   = np.any(in_zone[:len(in_zone)//2] > 0.5)
        now_out  = np.all(in_zone[-3:] < 0.5) if len(in_zone) >= 3 else False
        release_confirmed = bool(was_in and now_out)

        # Post-release count: frames after last in-zone
        last_in_idx = len(in_zone) - 1
        while last_in_idx >= 0 and in_zone[last_in_idx] < 0.5:
            last_in_idx -= 1
        post_release_count = len(in_zone) - 1 - last_in_idx

        # Reconstruct a dummy bbox from last known distance
        # (used only for bin proximity check — use Layer 3 min_dist feature)
        min_dist_feat = float(arr[-1, _F_MIN_DIST])
        # We don't have absolute position in MODE B — use feature-based evaluation
        return self._decide_from_features(
            pair_id           = pair_id,
            arr               = arr,
            held_frames       = held_frames,
            release_confirmed = release_confirmed,
            post_release_count= post_release_count,
            tracked_bins      = tracked_bins,
            min_dist_feat     = min_dist_feat,
        )

    # ── Internal — Decision Logic ─────────────────────────────────────────────

    def _decide(
        self,
        state:        _PairEvalState,
        sequence:     List[List[float]],
        obj_bbox:     np.ndarray,
        tracked_bins: List[TrackedBin],
    ) -> EvaluationResult:
        """
        Full decision path using bbox + tracked_bins (MODE A).
        """
        bin_present = len(tracked_bins) >= 1

        # --- Compute confidence components ---
        # 1. Hold signal (normalized, capped at 1.0)
        hold_score = min(state.held_frames / 15.0, 1.0)

        # 2. Release signal from post-release trail velocity
        avg_vel = _mean_velocity(state.post_release_trail)
        release_score = min(avg_vel / THROW_VEL_THRESHOLD, 1.0)

        # 3. Post-release motion duration
        motion_score = min(state.post_release_count / 20.0, 1.0)

        # 4. Sequence features (if available)
        seq_score = self._score_sequence(sequence) if len(sequence) >= MIN_SEQ_LEN else 0.5

        obj_cx, obj_cy = _centroid(obj_bbox)

        # --- CASE A: Bin present ---
        if bin_present:
            nearest_bin_d = _nearest_bin_dist((obj_cx, obj_cy), tracked_bins)
            bin_near = nearest_bin_d <= BIN_NEAR_PX

            # Bin-spatial score: how close did the object end up to the bin?
            bin_spatial_score = max(0.0, 1.0 - (nearest_bin_d / (BIN_NEAR_PX * 2)))

            base_conf = (
                W_HELD        * hold_score    +
                W_RELEASE     * release_score +
                W_MOTION      * motion_score  +
                W_BIN_SPATIAL * bin_spatial_score
            ) * seq_score

            if bin_near:
                # Object ended near bin → legal disposal
                confidence = min(base_conf * 1.2, 1.0)
                return EvaluationResult(
                    "legal_disposal", confidence, state.pair_id,
                    f"bin_dist={nearest_bin_d:.0f}px vel={avg_vel:.1f}px/f"
                )
            else:
                # Bin exists but object thrown away from it → illegal
                confidence = min(base_conf, 1.0)
                return EvaluationResult(
                    "illegal_dumping", confidence, state.pair_id,
                    f"bin_present but dist={nearest_bin_d:.0f}px > {BIN_NEAR_PX}px"
                )

        # --- CASE B: No bin in scene ---
        else:
            # Check if it's a valid throw (motion after release)
            is_valid_throw = (
                avg_vel >= MOTION_VEL_THRESHOLD and
                state.post_release_count >= MIN_POST_RELEASE
            )

            base_conf = (
                W_HELD    * hold_score    +
                W_RELEASE * release_score +
                W_MOTION  * motion_score
            ) * seq_score

            if is_valid_throw:
                confidence = min(base_conf / (W_HELD + W_RELEASE + W_MOTION), 1.0)
                return EvaluationResult(
                    "illegal_dumping", confidence, state.pair_id,
                    f"no_bin vel={avg_vel:.1f}px/f held={state.held_frames}f"
                )
            else:
                # Weak signal — low confidence illegal dumping
                confidence = min(base_conf * 0.6, 0.45)
                return EvaluationResult(
                    "illegal_dumping", confidence, state.pair_id,
                    f"no_bin weak_throw vel={avg_vel:.1f}px/f"
                )

    def _decide_from_features(
        self,
        pair_id:           str,
        arr:               np.ndarray,
        held_frames:       int,
        release_confirmed: bool,
        post_release_count: int,
        tracked_bins:      List[TrackedBin],
        min_dist_feat:     float,
    ) -> EvaluationResult:
        """
        Decision path using only feature vectors (MODE B — no bbox).
        Uses entry_event_score and velocity_y from the sequence.
        """
        if not release_confirmed:
            return EvaluationResult("pending", 0.0, pair_id, "awaiting_release")
        if held_frames < MIN_HELD_FRAMES:
            return EvaluationResult("pending", 0.1, pair_id, f"held={held_frames}")
        if post_release_count < MIN_POST_RELEASE:
            return EvaluationResult("pending", 0.15, pair_id, f"post_rel={post_release_count}")

        bin_present = len(tracked_bins) >= 1

        # Score from features
        entry_score  = float(np.max(arr[:, _F_ENTRY]))      # peak entry event score
        avg_vel_y    = float(np.mean(np.abs(arr[-5:, _F_VEL_Y])))  # recent vertical speed
        in_region    = float(np.any(arr[:, _F_IN_REGION] > 0.5))   # ever entered bin region
        hold_score   = min(held_frames / 15.0, 1.0)
        motion_score = min(post_release_count / 20.0, 1.0)

        seq_score = self._score_sequence(arr.tolist())

        if bin_present and in_region and entry_score >= 0.4:
            # Strong legal signal: was in zone + high entry score
            confidence = min((hold_score * 0.3 + entry_score * 0.5 + motion_score * 0.2) * seq_score, 1.0)
            return EvaluationResult("legal_disposal", confidence, pair_id,
                                    f"entry_score={entry_score:.2f} in_region=True")

        else:
            # Illegal: either no bin, or object never reached region
            vel_score  = min(avg_vel_y / THROW_VEL_THRESHOLD, 1.0)
            confidence = min(
                (hold_score * 0.3 + vel_score * 0.4 + motion_score * 0.3) * seq_score,
                1.0
            )
            reason = "no_bin" if not bin_present else f"entry_score={entry_score:.2f}_low"
            return EvaluationResult("illegal_dumping", confidence, pair_id, reason)

    def _score_sequence(self, sequence: List[List[float]]) -> float:
        """
        Quality multiplier [0.5, 1.0] based on sequence length and stability.
        Short or noisy sequences get penalized.
        """
        T = len(sequence)
        if T == 0:
            return 0.5
        # Length bonus: full credit at SEQUENCE_WINDOW frames (from L3 config = 24)
        length_score = min(T / 24.0, 1.0)
        # Temporal stability: low variance in distance col → stable tracking
        if T >= 4:
            dists = [v[_F_DIST] for v in sequence if len(v) > _F_DIST]
            if dists:
                std = float(np.std(dists))
                stability = max(0.0, 1.0 - (std / 300.0))  # 300px std → 0 score
            else:
                stability = 0.7
        else:
            stability = 0.7
        return 0.5 + 0.5 * (0.6 * length_score + 0.4 * stability)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _get_or_create(self, pair_id: str) -> _PairEvalState:
        if pair_id not in self._states:
            self._states[pair_id] = _PairEvalState(pair_id=pair_id)
        return self._states[pair_id]

    def tick(self) -> None:
        """
        Call once per frame to age stale states.
        States not updated for MAX_RESULT_AGE frames are purged.
        """
        expired = []
        for pid, state in self._states.items():
            state.frames_since_update += 1
            if state.frames_since_update > MAX_RESULT_AGE:
                expired.append(pid)
        for pid in expired:
            del self._states[pid]

    def get_last_result(self, pair_id: str) -> Optional[EvaluationResult]:
        state = self._states.get(pair_id)
        return state.last_result if state else None
