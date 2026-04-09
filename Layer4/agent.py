# Layer4/agent.py
"""
Hybrid Decision Agent — Layer 4

Combines:
    1. Rule-based score  (deterministic, human-readable reasons)
    2. ML model score    (learned from log data via MLScoringModel)
    3. Transformer score (existing DumpingClassifier via DumpingInference)

Final decision formula:
    final_score = α * transformer_score + β * ml_score + γ * rule_score

Default weights (tunable in config or at runtime):
    α = 0.50   (transformer — primary signal)
    β = 0.30   (ML model   — secondary learned signal)
    γ = 0.20   (rules      — human-readable safety net)

Output dict:
    {
        "alert":            bool,
        "confidence":       float [0, 1],
        "type":             "vehicle_dumping" | "pedestrian_dumping" | "anomaly",
        "transformer_score": float,
        "ml_score":         float,
        "rule_score":       float,
        "final_score":      float,
        "reasons":          List[str],
    }

Usage:
    agent = HybridAgent()
    decision = agent.decide(pair, tracks, transformer_prob, ml_output)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from Layer3.pair_state  import PairState
from Layer2.track_state import TrackedObject

# ── Tunable blend weights ──────────────────────────────────
ALPHA = 0.50   # transformer
BETA  = 0.30   # ml model
GAMMA = 0.20   # rule-based

# ── Alert threshold on final blended score ─────────────────
ALERT_THRESHOLD = 0.45

# ── Rule thresholds ────────────────────────────────────────
RULE_MIN_HELD_FRAMES    = 5     # must have been held for this many frames
RULE_MIN_RELEASED       = 6     # must be released for this many frames
RULE_MAX_VELOCITY       = 80.0  # pixels/frame — fast moving object = not dumped yet
RULE_MIN_INTERACTION    = 8     # minimum frames of interaction before scoring
RULE_STATIONARY_THRESH  = 0.3   # normalised object velocity below this = stationary
RULE_AREA_MIN           = 500   # ignore tiny objects (noise)

# Vehicle class names (from COCO — RT-DETR uses these)
VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle", "bicycle"}


# ══════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════

def _centroid(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], dtype=np.float32)


def _area(bbox: np.ndarray) -> float:
    return float((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))


def _vehicle_present(tracks: List[TrackedObject]) -> bool:
    return any(t.class_name in VEHICLE_CLASSES for t in tracks)


def _classify_type(pair: PairState,
                   tracks: List[TrackedObject]) -> str:
    """Classify the dumping type for reporting."""
    if _vehicle_present(tracks):
        return "vehicle_dumping"
    if pair.held_frames >= RULE_MIN_HELD_FRAMES:
        return "pedestrian_dumping"
    return "anomaly"


# ══════════════════════════════════════════════════════════
# Rule-based scorer
# ══════════════════════════════════════════════════════════

def evaluate_rules(
    pair:       PairState,
    tracks:     List[TrackedObject],
    feature_dict: Dict,
) -> Tuple[float, List[str]]:
    """
    Evaluate a set of deterministic rules and return a composite score [0,1]
    plus a list of human-readable reason strings.

    Scoring:
        Each rule contributes an additive weight.
        Total is clipped to [0, 1].

    Args:
        pair:         PairState from Layer 3 MemoryEngine
        tracks:       All TrackedObject from Layer 2 (for vehicle detection)
        feature_dict: structured feature dict (from build_feature_dict())

    Returns:
        (rule_score, reasons)
    """
    score   = 0.0
    reasons = []

    distance         = feature_dict.get("distance",         999.0)
    velocity         = feature_dict.get("velocity",         0.0)
    interaction_time = feature_dict.get("interaction_time", 0)
    stationary_time  = feature_dict.get("stationary_time",  0)
    object_area      = feature_dict.get("object_area",      0.0)
    vehicle_present  = bool(feature_dict.get("vehicle_present", 0))
    event_count      = feature_dict.get("event_count",      0)

    # ── Rule 0: minimum interaction — ignore transient pairings ──
    if interaction_time < RULE_MIN_INTERACTION:
        return 0.0, ["interaction_too_short"]

    # ── Rule 1: object was held ───────────────────────────────────
    if pair.held_frames >= RULE_MIN_HELD_FRAMES:
        score   += 0.25
        reasons.append(f"held_for_{pair.held_frames}_frames")

    # ── Rule 2: object released / separated from person ──────────
    if pair.released_frames >= RULE_MIN_RELEASED:
        score   += 0.30
        reasons.append(f"released_for_{pair.released_frames}_frames")
    elif pair.ever_held and pair.released_frames > 0:
        score   += 0.10
        reasons.append(f"recently_released_{pair.released_frames}_frames")

    # ── Rule 3: object is now stationary ─────────────────────────
    if velocity < RULE_MAX_VELOCITY and stationary_time > 3:
        score   += 0.15
        reasons.append(f"object_stationary_{stationary_time}f")

    # ── Rule 4: distance increasing (person walking away) ─────────
    seq = pair.get_sequence()
    if len(pair.sequence) >= 4:
        recent_dist  = seq[-1, 0]   # feature 0 = dist_norm
        earlier_dist = seq[-4, 0]
        if recent_dist > earlier_dist + 0.05:
            score   += 0.15
            reasons.append("person_moving_away")

    # ── Rule 5: vehicle nearby bonus ──────────────────────────────
    if vehicle_present:
        score   += 0.10
        reasons.append("vehicle_present")

    # ── Rule 6: object not too tiny ───────────────────────────────
    if object_area < RULE_AREA_MIN:
        score   -= 0.10
        reasons.append("object_too_small_penalty")

    # ── Rule 7: trash already flagged by Layer 1 ──────────────────
    obj_track = _find_object_track(pair.object_id, tracks)
    if obj_track is not None and obj_track.is_trash:
        score   += 0.20
        reasons.append(f"layer1_trash_flag_{obj_track.trash_how}")

    return float(np.clip(score, 0.0, 1.0)), reasons


def _find_object_track(
    object_id: int,
    tracks:    List[TrackedObject],
) -> Optional[TrackedObject]:
    for t in tracks:
        if t.track_id == object_id:
            return t
    return None


# ══════════════════════════════════════════════════════════
# Feature dict builder  (feeds both rules and ML model)
# ══════════════════════════════════════════════════════════

def build_feature_dict(
    pair:   PairState,
    tracks: List[TrackedObject],
) -> Dict:
    """
    Build the structured feature dict from PairState + current tracks.
    Used by both evaluate_rules() and MLScoringModel.predict().
    """
    person_track = _find_object_track(pair.person_id, tracks)
    obj_track    = _find_object_track(pair.object_id, tracks)

    # Distance
    if person_track is not None and obj_track is not None:
        distance = float(np.linalg.norm(
            _centroid(person_track.bbox) - _centroid(obj_track.bbox)
        ))
    else:
        distance = 999.0

    # IoU
    iou = 0.0
    if person_track is not None and obj_track is not None:
        a, b = person_track.bbox, obj_track.bbox
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter > 0:
            aa = (a[2]-a[0])*(a[3]-a[1])
            ab = (b[2]-b[0])*(b[3]-b[1])
            iou = inter / (aa + ab - inter)

    # Object velocity (from sequence)
    seq = pair.get_sequence()
    velocity = float(seq[-1, 2]) * 50.0 if len(pair.sequence) > 0 else 0.0  # denorm

    # Stationary frames (frames where velocity was near zero)
    stationary_time = 0
    if len(pair.sequence) >= 2:
        vels = seq[:, 2]
        stationary_time = int((vels < RULE_STATIONARY_THRESH).sum())

    # Object area
    object_area = _area(obj_track.bbox) if obj_track is not None else 0.0

    # Vehicle present
    vehicle_present = 1 if _vehicle_present(tracks) else 0

    # Event count
    event_count = pair.held_frames + pair.released_frames

    return {
        "distance":         distance,
        "iou":              iou,
        "velocity":         velocity,
        "interaction_time": pair.frames_seen,
        "stationary_time":  stationary_time,
        "object_area":      object_area,
        "vehicle_present":  vehicle_present,
        "event_count":      event_count,
    }


# ══════════════════════════════════════════════════════════
# Hybrid Agent
# ══════════════════════════════════════════════════════════

class HybridAgent:
    """
    Combines transformer, ML, and rule-based scores into a single decision.

    Usage:
        agent = HybridAgent(alpha=0.5, beta=0.3, gamma=0.2)
        decision = agent.decide(pair, tracks, transformer_prob, ml_output)
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        beta:  float = BETA,
        gamma: float = GAMMA,
        threshold: float = ALERT_THRESHOLD,
    ):
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.threshold = threshold

        # Normalise weights so they sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta  /= total
        self.gamma /= total

    def decide(
        self,
        pair:              PairState,
        tracks:            List[TrackedObject],
        transformer_prob:  float,                  # from DumpingInference.predict()
        ml_output:         Dict,                   # from MLScoringModel.predict()
    ) -> Dict:
        """
        Main decision entry point.

        Args:
            pair:             PairState from Layer 3
            tracks:           All TrackedObject from Layer 2
            transformer_prob: float from DumpingInference (-1 = not ready)
            ml_output:        dict {"probability": float, "confidence": float}

        Returns:
            Decision dict (see module docstring).
        """
        # Build shared feature dict once
        feature_dict = build_feature_dict(pair, tracks)

        # ── Rule-based score ──────────────────────────────────────
        rule_score, reasons = evaluate_rules(pair, tracks, feature_dict)

        # ── ML score ──────────────────────────────────────────────
        ml_score = ml_output.get("probability", 0.5)

        # ── Transformer score ──────────────────────────────────────
        # If not ready (-1), fall back to 0.5 (neutral)
        t_score = transformer_prob if transformer_prob >= 0 else 0.5

        # ── Blend ─────────────────────────────────────────────────
        final_score = (
            self.alpha * t_score
            + self.beta  * ml_score
            + self.gamma * rule_score
        )
        final_score = float(np.clip(final_score, 0.0, 1.0))

        alert = final_score >= self.threshold

        # ── Classify type ─────────────────────────────────────────
        dump_type = _classify_type(pair, tracks) if alert else "none"

        return {
            "alert":             alert,
            "confidence":        final_score,
            "type":              dump_type,
            "transformer_score": round(t_score,    4),
            "ml_score":          round(ml_score,   4),
            "rule_score":        round(rule_score,  4),
            "final_score":       round(final_score, 4),
            "reasons":           reasons,
            "pair_key":          pair.pair_key,
            "feature_dict":      feature_dict,
        }

    def combine(
        self,
        ml_output:    Dict,
        rule_output:  Tuple[float, List[str]],
        transformer_prob: float = 0.5,
    ) -> Dict:
        """
        Alternative entry when you have pre-computed rule_output tuple.
        Mirrors the API mentioned in the task spec.
        """
        rule_score, reasons = rule_output
        ml_score = ml_output.get("probability", 0.5)
        t_score  = transformer_prob if transformer_prob >= 0 else 0.5

        final_score = float(np.clip(
            self.alpha * t_score + self.beta * ml_score + self.gamma * rule_score,
            0.0, 1.0,
        ))
        alert = final_score >= self.threshold

        return {
            "alert":             alert,
            "confidence":        final_score,
            "type":              "pedestrian_dumping" if alert else "none",
            "transformer_score": round(t_score,    4),
            "ml_score":          round(ml_score,   4),
            "rule_score":        round(rule_score,  4),
            "final_score":       round(final_score, 4),
            "reasons":           reasons,
        }