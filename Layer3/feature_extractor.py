"""
Layer 3 — Bin Interaction Feature Extractor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
  Convert raw per-frame tracking data (Layer 2 output) into fixed-length
  feature sequences that encode how a trash object interacts with a bin
  over time.  These sequences feed Layer 4 (LSTM/GRU) for learned
  classification — this layer does NOT decide legal vs illegal.

WHAT IT DOES PER FRAME
  For every (trash_track, bin_track) pair within association distance:
    1. Compute per-frame spatial + kinematic features
    2. Accumulate temporal state (time-in-region, min distance …)
    3. Compute a soft entry-event score in [0, 1]
    4. Append the 8D feature vector to a sliding window sequence

OUTPUT (per pair, per frame update)
  {
    "pair_id"    : "trash_<tid>_bin_<bid>",
    "sequence"   : [ [f0…f7], [f0…f7], … ],   # up to SEQUENCE_WINDOW vectors
    "timestamps" : [ t0, t1, … ],
  }

FEATURE VECTOR  (fixed length = 8)
  idx  name                  description
  ---  ────────────────────  ─────────────────────────────────────────────────
   0   distance_to_bin_center  euclidean pixels from trash centroid → bin center
   1   is_in_zone            1.0 if inside expanded bin zone, else 0.0
   2   is_in_region          1.0 if inside shrunk bin region, else 0.0
   3   velocity_y            vertical pixel/frame speed (+ve = downward)
   4   trajectory_slope      slope of the last N-point trail (rise/run)
   5   time_in_region        cumulative frames spent inside bin_region
   6   min_distance_to_bin   minimum distance seen so far this sequence
   7   entry_event_score     soft [0,1] signal of a bin-entry event

DESIGN CONSTRAINTS (from prompt)
  • Does NOT classify dumping
  • CPU-only, O(N²) per frame max
  • Modular, production-ready
  • Debug visualisation is optional (pass debug=True)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from Layer2.track_state import TrackedObject
from Layer2.bin_tracker import TrackedBin

from .config import (
    BIN_REGION_SHRINK,
    BIN_ZONE_EXPAND,
    BIN_ASSOCIATION_MAX_DIST,
    SEQUENCE_WINDOW,
    ENTRY_VY_THRESHOLD,
    ENTRY_MIN_FRAMES,
    DBG_REGION_COLOR,
    DBG_ZONE_COLOR,
    DBG_TRAIL_COLOR,
    DBG_LINE_COLOR,
    DBG_SCORE_COLOR,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Feature vector index constants — makes code self-documenting
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DIST_TO_CENTER   = 0
FEAT_IS_IN_ZONE       = 1
FEAT_IS_IN_REGION     = 2
FEAT_VELOCITY_Y       = 3
FEAT_TRAJ_SLOPE       = 4
FEAT_TIME_IN_REGION   = 5
FEAT_MIN_DIST         = 6
FEAT_ENTRY_SCORE      = 7
FEATURE_DIM           = 8   # total fixed length — must never change


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers (pure numpy, no OpenCV needed)
# ─────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) from [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _scale_bbox(
    bbox: np.ndarray, factor: float
) -> Tuple[float, float, float, float]:
    """
    Scale a bbox [x1,y1,x2,y2] about its center by `factor`.
    factor < 1 → shrink (bin_region)
    factor > 1 → expand (bin_zone)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) / 2.0 * factor
    hh = (y2 - y1) / 2.0 * factor
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _point_in_box(
    pt: Tuple[float, float],
    box: Tuple[float, float, float, float],
) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _trail_slope(trail: deque, n: int = 8) -> float:
    """
    Estimate the slope (dy/dx) of the most recent `n` trail points.
    Returns 0.0 if the trail is too short or the x-range is near zero.
    This captures whether the object is moving diagonally toward the bin.
    """
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    recent = pts[-n:] if len(pts) >= n else pts
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    if abs(dx) < 1e-3:
        return 0.0
    return dy / dx


# ─────────────────────────────────────────────────────────────────────────────
#  Per-(trash, bin) temporal state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _PairState:
    """
    Accumulates temporal state for one (trash_id, bin_id) pair.
    Reset if either track disappears and the sequence expires.
    """
    trash_id:             int
    bin_id:               int

    # Sliding window of feature vectors and their timestamps
    sequence:  List[List[float]] = field(default_factory=list)
    timestamps: List[float]      = field(default_factory=list)

    # Temporal accumulators
    time_in_region:       int   = 0      # frames spent inside bin_region
    entered_region_flag:  bool  = False  # True once trash first enters region
    min_distance_to_bin:  float = float("inf")
    final_distance:       float = float("inf")

    # Entry event bookkeeping
    consecutive_in_region: int  = 0     # consecutive frames inside region
    entry_event_score:     float = 0.0  # last computed score

    # Staleness — frames since this pair was last updated
    frames_since_update:  int   = 0

    def pair_id(self) -> str:
        return f"trash_{self.trash_id}_bin_{self.bin_id}"

    def append(self, vec: List[float], ts: float) -> None:
        """Append one feature vector and timestamp; maintain sliding window."""
        self.sequence.append(vec)
        self.timestamps.append(ts)
        # Keep only the most recent SEQUENCE_WINDOW entries
        if len(self.sequence) > SEQUENCE_WINDOW:
            self.sequence  = self.sequence[-SEQUENCE_WINDOW:]
            self.timestamps = self.timestamps[-SEQUENCE_WINDOW:]

    def as_output(self) -> Dict:
        """Return the standard Layer 3 → Layer 4 output dict."""
        return {
            "pair_id":    self.pair_id(),
            "sequence":   [v[:] for v in self.sequence],   # deep copy
            "timestamps": self.timestamps[:],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class BinInteractionFeatureExtractor:
    """
    Layer 3 core class.

    Usage (in run_pipeline.py):
        extractor = BinInteractionFeatureExtractor()

        # inside the frame loop:
        sequences = extractor.update(tracked_objects, tracked_bins, timestamp)
        # sequences → List[Dict] with "pair_id", "sequence", "timestamps"

    Optional debug overlay:
        frame = extractor.draw_debug(frame, tracked_objects, tracked_bins)
    """

    def __init__(self, debug: bool = False):
        self._debug = debug

        # Active pair states: (trash_id, bin_id) → _PairState
        self._states: Dict[Tuple[int, int], _PairState] = {}

        # How many frames a pair state survives without being refreshed.
        # If a trash track disappears for longer than this it is cleared.
        self._STALE_FRAMES = SEQUENCE_WINDOW * 2

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        tracked_objects: List[TrackedObject],
        tracked_bins:    List[TrackedBin],
        timestamp:       Optional[float] = None,
    ) -> List[Dict]:
        """
        Process one frame of Layer 2 output.

        Args:
            tracked_objects: all active tracks from ByteTrackWrapper.update()
            tracked_bins:    all confirmed bins from BinTracker.update()
            timestamp:       wall-clock time (seconds); defaults to time.time()

        Returns:
            List of output dicts (one per active trash↔bin pair this frame).
            Each dict: {"pair_id": str, "sequence": List[List[float]], "timestamps": List[float]}
        """
        ts = timestamp if timestamp is not None else time.time()

        # Separate trash tracks from other objects
        trash_tracks = [
            t for t in tracked_objects
            if t.is_trash or t.class_name == "trash"
        ]

        # Tick staleness for all existing states
        for state in self._states.values():
            state.frames_since_update += 1

        active_keys: set = set()

        # ── For each trash↔bin pair within association distance ───────────────
        for trash in trash_tracks:
            trash_cx, trash_cy = _centroid(trash.bbox)

            # Find the best bin — nearest AND trajectory-converging
            best_bin, best_dist = self._best_bin(
                (trash_cx, trash_cy), trash.trail, tracked_bins
            )
            if best_bin is None:
                continue   # no bin visible — skip this trash track

            key = (trash.track_id, best_bin.bin_id)

            # ── Pair locking: once a trash↔bin pair is established, NEVER switch ──
            # Check if this trash already has an established pair with a different bin.
            # If yes, KEEP the original pair — do not re-associate.
            # This prevents phantom bins or temporarily closer bins from stealing
            # a correct established relationship.
            existing_pair_key = None
            for existing_key in list(self._states.keys()):
                ex_trash_id, ex_bin_id = existing_key
                if ex_trash_id == trash.track_id:
                    existing_pair_key = existing_key
                    break

            if existing_pair_key is not None:
                # Pair already exists — always use the original bin, find it in tracked_bins
                _, locked_bin_id = existing_pair_key
                locked_bin = next(
                    (tb for tb in tracked_bins if tb.bin_id == locked_bin_id),
                    None
                )
                if locked_bin is not None:
                    # Use the locked bin instead of best_bin
                    best_bin = locked_bin
                    key = existing_pair_key
                # If locked bin not in tracked_bins at all (fully gone), fall through to best_bin

            active_keys.add(key)

            # Get or create pair state
            if key not in self._states:
                self._states[key] = _PairState(
                    trash_id = trash.track_id,
                    bin_id   = best_bin.bin_id,
                )
            state = self._states[key]
            state.frames_since_update = 0   # reset staleness

            # Update nearest_bin reference for geometry computation below
            nearest_bin = best_bin

            # ── Compute bin geometry ──────────────────────────────────────────
            bin_region = _scale_bbox(nearest_bin.bbox, BIN_REGION_SHRINK)
            bin_zone   = _scale_bbox(nearest_bin.bbox, BIN_ZONE_EXPAND)
            bin_center = _centroid(nearest_bin.bbox)

            # ── Per-frame spatial features ────────────────────────────────────
            dist_to_center = _euclidean((trash_cx, trash_cy), bin_center)
            is_in_zone     = 1.0 if _point_in_box((trash_cx, trash_cy), bin_zone)   else 0.0
            is_in_region   = 1.0 if _point_in_box((trash_cx, trash_cy), bin_region) else 0.0

            # ── Kinematic features ────────────────────────────────────────────
            velocity_y = float(trash.trail[-1][1] - trash.trail[-2][1]) \
                         if len(trash.trail) >= 2 else 0.0
            traj_slope = _trail_slope(trash.trail, n=8)

            # ── Temporal accumulation ─────────────────────────────────────────
            if is_in_region:
                state.time_in_region        += 1
                state.consecutive_in_region += 1
                if not state.entered_region_flag:
                    state.entered_region_flag = True
            else:
                state.consecutive_in_region  = 0

            state.min_distance_to_bin  = min(state.min_distance_to_bin, dist_to_center)
            state.final_distance       = dist_to_center

            # ── Entry event score (soft signal in [0, 1]) ─────────────────────
            entry_score = self._compute_entry_score(state, velocity_y, is_in_region)
            state.entry_event_score = entry_score

            # ── Build feature vector (MUST keep FEATURE_DIM = 8) ─────────────
            vec = [0.0] * FEATURE_DIM
            vec[FEAT_DIST_TO_CENTER] = float(dist_to_center)
            vec[FEAT_IS_IN_ZONE]     = float(is_in_zone)
            vec[FEAT_IS_IN_REGION]   = float(is_in_region)
            vec[FEAT_VELOCITY_Y]     = float(velocity_y)
            vec[FEAT_TRAJ_SLOPE]     = float(traj_slope)
            vec[FEAT_TIME_IN_REGION] = float(state.time_in_region)
            vec[FEAT_MIN_DIST]       = float(state.min_distance_to_bin)
            vec[FEAT_ENTRY_SCORE]    = float(entry_score)

            state.append(vec, ts)

        # ── Purge stale pair states (track disappeared) ───────────────────────
        stale_keys = [
            k for k, s in self._states.items()
            if s.frames_since_update > self._STALE_FRAMES
        ]
        for k in stale_keys:
            del self._states[k]

        # ── Return output for all active pairs this frame ─────────────────────
        outputs = [
            self._states[k].as_output()
            for k in active_keys
            if k in self._states and len(self._states[k].sequence) > 0
        ]
        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    #  Helper — find nearest bin
    # ─────────────────────────────────────────────────────────────────────────

    def _best_bin(
        self,
        trash_centroid: Tuple[float, float],
        trail:          deque,
        bins:           List[TrackedBin],
    ) -> Tuple[Optional[TrackedBin], float]:
        """
        Pick the best bin for this trash track using a composite score:

          score = 0.6 * (1 - norm_dist) + 0.4 * convergence

        where:
          norm_dist   = distance / BIN_ASSOCIATION_MAX_DIST  (0=close, 1=far)
          convergence = dot product of trash velocity vector and the unit
                        vector pointing FROM trash TOWARD the bin.
                        Range [-1, 1]: +1 = moving straight at the bin,
                        -1 = moving directly away.

        This means a bin the trash is moving TOWARD scores higher than a
        bin that is merely closer but in the wrong direction — fixing the
        multi-bin wrong-association bug.

        Returns (None, inf) if no bin is within BIN_ASSOCIATION_MAX_DIST.
        """
        # Estimate trash velocity vector from the last few trail points
        pts = list(trail)
        if len(pts) >= 3:
            # Use last 3 points for a stable velocity estimate
            vx = pts[-1][0] - pts[-3][0]
            vy = pts[-1][1] - pts[-3][1]
        elif len(pts) == 2:
            vx = pts[-1][0] - pts[-2][0]
            vy = pts[-1][1] - pts[-2][1]
        else:
            vx, vy = 0.0, 0.0

        speed = math.hypot(vx, vy)
        has_velocity = speed > 1e-3   # only use direction if moving

        best_bin   = None
        best_score = -1.0
        best_dist  = float("inf")

        tx, ty = trash_centroid

        for tb in bins:
            bin_cx, bin_cy = _centroid(tb.bbox)
            d = _euclidean((tx, ty), (bin_cx, bin_cy))

            if d > BIN_ASSOCIATION_MAX_DIST:
                continue   # too far — ignore

            # Normalised distance score (1 = touching, 0 = at max distance)
            norm_dist = d / BIN_ASSOCIATION_MAX_DIST
            dist_score = 1.0 - norm_dist   # higher = closer

            # Convergence score — is trash moving TOWARD this bin?
            if has_velocity:
                # Unit vector from trash toward this bin
                dx = bin_cx - tx
                dy = bin_cy - ty
                bin_dist = math.hypot(dx, dy) + 1e-6
                ux = dx / bin_dist
                uy = dy / bin_dist
                # Dot product with normalised velocity
                dot = (vx * ux + vy * uy) / (speed + 1e-6)
                # dot in [-1, 1]; map to [0, 1]
                convergence = (dot + 1.0) / 2.0
            else:
                convergence = 0.5   # neutral when stationary

            # Composite score — trajectory convergence weighted heavily
            score = 0.6 * dist_score + 0.4 * convergence

            if score > best_score:
                best_score = score
                best_bin   = tb
                best_dist  = d

        return best_bin, best_dist

    # keep old name as alias for debug draw which also calls it
    def _nearest_bin(
        self,
        trash_centroid: Tuple[float, float],
        bins:           List[TrackedBin],
    ) -> Tuple[Optional[TrackedBin], float]:
        """Alias → delegates to _best_bin with empty trail (distance-only)."""
        return self._best_bin(trash_centroid, deque(), bins)

    # ─────────────────────────────────────────────────────────────────────────
    #  Entry event score computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_entry_score(
        self,
        state:      _PairState,
        velocity_y: float,
        is_in_region: float,
    ) -> float:
        """
        Soft signal in [0, 1] indicating a bin-entry event.

        Factors:
          A. trash is currently inside bin_region                  → +0.4
          B. downward velocity > ENTRY_VY_THRESHOLD                → +0.3
          C. persistence: consecutive_in_region >= ENTRY_MIN_FRAMES → +0.3

        No single factor alone fires the event — they compound.
        This is a *signal*, not a classification.
        """
        score = 0.0

        if is_in_region:
            score += 0.4

        if velocity_y >= ENTRY_VY_THRESHOLD:
            score += 0.3

        if state.consecutive_in_region >= ENTRY_MIN_FRAMES:
            score += 0.3

        return min(score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    #  Debug visualisation (optional — call only if debug=True)
    # ─────────────────────────────────────────────────────────────────────────

    def draw_debug(
        self,
        frame:          np.ndarray,
        tracked_objects: List[TrackedObject],
        tracked_bins:   List[TrackedBin],
    ) -> np.ndarray:
        """
        Draw Layer 3 debug overlays onto the frame.

        • Green rectangle  = bin_region  (inner shrunk zone)
        • Yellow rectangle = bin_zone    (outer expanded zone)
        • Purple dots      = trash trail
        • Orange line      = trash → nearest bin connector
        • White text       = entry_event_score for the pair

        Safe to call even if debug=False — will just return frame unchanged.
        """
        if not self._debug:
            return frame

        # Draw bin regions and zones
        for tb in tracked_bins:
            bin_region = _scale_bbox(tb.bbox, BIN_REGION_SHRINK)
            bin_zone   = _scale_bbox(tb.bbox, BIN_ZONE_EXPAND)

            # Zone (outer, yellow)
            cv2.rectangle(
                frame,
                (int(bin_zone[0]), int(bin_zone[1])),
                (int(bin_zone[2]), int(bin_zone[3])),
                DBG_ZONE_COLOR, 1,
            )
            cv2.putText(
                frame, "zone",
                (int(bin_zone[0]) + 2, int(bin_zone[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, DBG_ZONE_COLOR, 1, cv2.LINE_AA,
            )

            # Region (inner, green)
            cv2.rectangle(
                frame,
                (int(bin_region[0]), int(bin_region[1])),
                (int(bin_region[2]), int(bin_region[3])),
                DBG_REGION_COLOR, 1,
            )
            cv2.putText(
                frame, "region",
                (int(bin_region[0]) + 2, int(bin_region[3]) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, DBG_REGION_COLOR, 1, cv2.LINE_AA,
            )

        # Draw per-trash debug info
        trash_tracks = [
            t for t in tracked_objects
            if t.is_trash or t.class_name == "trash"
        ]
        for trash in trash_tracks:
            trash_cx, trash_cy = _centroid(trash.bbox)

            # Trail (purple dots)
            pts = list(trash.trail)
            for i in range(1, len(pts)):
                cv2.circle(
                    frame,
                    (int(pts[i][0]), int(pts[i][1])),
                    2, DBG_TRAIL_COLOR, -1,
                )

            # Best bin connector (orange line) + entry score
            nearest_bin, _ = self._best_bin(
                (trash_cx, trash_cy), trash.trail, tracked_bins
            )
            if nearest_bin is not None:
                bin_cx, bin_cy = _centroid(nearest_bin.bbox)
                cv2.line(
                    frame,
                    (int(trash_cx), int(trash_cy)),
                    (int(bin_cx), int(bin_cy)),
                    DBG_LINE_COLOR, 1, cv2.LINE_AA,
                )

                key = (trash.track_id, nearest_bin.bin_id)
                state = self._states.get(key)
                if state is not None:
                    label = (
                        f"entry:{state.entry_event_score:.2f} "
                        f"t_in:{state.time_in_region}"
                    )
                    cv2.putText(
                        frame, label,
                        (int(trash_cx) + 4, int(trash_cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        DBG_SCORE_COLOR, 1, cv2.LINE_AA,
                    )

        return frame

    # ─────────────────────────────────────────────────────────────────────────
    #  Utility — access current sequences (for Layer 4 polling)
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_sequences(self) -> List[Dict]:
        """
        Return all current pair sequences (not just those active this frame).
        Useful for Layer 4 to poll all in-progress interactions at any time.
        """
        return [
            s.as_output()
            for s in self._states.values()
            if len(s.sequence) > 0
        ]

    def get_sequence(self, trash_id: int, bin_id: int) -> Optional[Dict]:
        """Retrieve the sequence for a specific (trash_id, bin_id) pair."""
        state = self._states.get((trash_id, bin_id))
        if state is None or len(state.sequence) == 0:
            return None
        return state.as_output()