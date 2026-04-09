"""
Layer 3/feature.py — Feature Extractor  (v2)

Two public functions:

1. extract_features()  — UNCHANGED from v1
   Produces the 9-dim float32 vector fed into the transformer sequence buffer.
   Layer 4 DumpingClassifier reads these via PairState.get_sequence().

2. extract_feature_dict()  — NEW in v2
   Produces a structured dict for the ML scoring model and hybrid agent.
   Called by Layer4/agent.py::build_feature_dict() each frame.

Feature vector  (dimension = 9, for transformer — unchanged):
  0  distance_norm
  1  delta_distance_norm
  2  obj_velocity_norm
  3  person_velocity_norm
  4  direction_x
  5  direction_y
  6  is_holding
  7  obj_area_norm
  8  visibility_score

Feature dict  (for ML model + rules — new):
  distance          float  — raw pixel distance
  iou               float  — person/object bbox IoU
  velocity          float  — object velocity px/frame
  interaction_time  int    — frames pair has been observed
  stationary_time   int    — frames object was near-stationary
  object_area       float  — raw bbox area (px²)
  vehicle_present   int    — 1 if vehicle track nearby
  event_count       int    — held_frames + released_frames
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from .config import NORM_DISTANCE, NORM_VELOCITY, NORM_AREA, HOLD_DISTANCE_PX

# ── Public constant — Layer 4 needs to know this ──────────
FEATURE_DIM = 9


def _centroid(bbox: np.ndarray) -> np.ndarray:
    """Returns [cx, cy] as float32 array."""
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
        dtype=np.float32,
    )


def _bbox_area(bbox: np.ndarray) -> float:
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


# ══════════════════════════════════════════════════════════
# 1. Transformer feature vector (unchanged from v1)
# ══════════════════════════════════════════════════════════

def extract_features(
    person_bbox:      np.ndarray,
    object_bbox:      np.ndarray,
    prev_person_bbox: Optional[np.ndarray],
    prev_object_bbox: Optional[np.ndarray],
    prev_distance:    Optional[float],
    both_visible:     bool = True,
) -> np.ndarray:
    """
    Extract a single feature vector for one (person, object) pair.

    Args:
        person_bbox:      [x1,y1,x2,y2] — current person bbox
        object_bbox:      [x1,y1,x2,y2] — current object bbox
        prev_person_bbox: bbox from previous frame (None on first frame)
        prev_object_bbox: bbox from previous frame (None on first frame)
        prev_distance:    centroid distance from previous frame (None on first)
        both_visible:     False if either track was missing this frame

    Returns:
        np.ndarray of shape (FEATURE_DIM,), dtype float32
    """
    pc = _centroid(person_bbox)
    oc = _centroid(object_bbox)

    # ── 0. Distance ───────────────────────────────────────
    diff       = pc - oc
    distance   = float(np.linalg.norm(diff))
    dist_norm  = np.clip(distance / NORM_DISTANCE, 0.0, 1.0)

    # ── 1. Delta distance ─────────────────────────────────
    if prev_distance is not None:
        delta_dist = (distance - prev_distance) / NORM_DISTANCE
        delta_dist = np.clip(delta_dist, -1.0, 1.0)
    else:
        delta_dist = 0.0

    # ── 2 & 3. Velocities ────────────────────────────────
    if prev_object_bbox is not None:
        prev_oc      = _centroid(prev_object_bbox)
        obj_vel      = float(np.linalg.norm(oc - prev_oc))
        obj_vel_norm = np.clip(obj_vel / NORM_VELOCITY, 0.0, 1.0)
    else:
        obj_vel_norm = 0.0

    if prev_person_bbox is not None:
        prev_pc         = _centroid(prev_person_bbox)
        person_vel      = float(np.linalg.norm(pc - prev_pc))
        person_vel_norm = np.clip(person_vel / NORM_VELOCITY, 0.0, 1.0)
    else:
        person_vel_norm = 0.0

    # ── 4 & 5. Direction (object → person unit vector) ───
    if distance > 1e-3:
        dir_vec = diff / distance
        dir_x   = float(np.clip(dir_vec[0], -1.0, 1.0))
        dir_y   = float(np.clip(dir_vec[1], -1.0, 1.0))
    else:
        dir_x, dir_y = 0.0, 0.0

    # ── 6. Is holding ─────────────────────────────────────
    is_holding = 1.0 if distance < HOLD_DISTANCE_PX else 0.0

    # ── 7. Object area ────────────────────────────────────
    obj_area_norm = np.clip(_bbox_area(object_bbox) / NORM_AREA, 0.0, 1.0)

    # ── 8. Visibility score ───────────────────────────────
    visibility = 1.0 if both_visible else 0.5

    return np.array([
        dist_norm,
        delta_dist,
        obj_vel_norm,
        person_vel_norm,
        dir_x,
        dir_y,
        is_holding,
        obj_area_norm,
        visibility,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════
# 2. Structured feature dict (NEW — for ML model + agent)
# ══════════════════════════════════════════════════════════

def extract_feature_dict(
    person_bbox:      np.ndarray,
    object_bbox:      np.ndarray,
    prev_object_bbox: Optional[np.ndarray],
    interaction_time: int,
    held_frames:      int,
    released_frames:  int,
    sequence:         Optional[np.ndarray] = None,
    vehicle_present:  bool = False,
) -> Dict:
    """
    Produce the structured feature dict consumed by:
        - MLScoringModel.predict()
        - HybridAgent.evaluate_rules()

    Args:
        person_bbox:      current person [x1,y1,x2,y2]
        object_bbox:      current object [x1,y1,x2,y2]
        prev_object_bbox: object bbox from previous frame
        interaction_time: pair.frames_seen
        held_frames:      pair.held_frames
        released_frames:  pair.released_frames
        sequence:         optional (T, FEATURE_DIM) array for stationary count
        vehicle_present:  True if vehicle track visible in scene

    Returns:
        Dict with keys matching FEATURE_KEYS in ml_model.py
    """
    pc = _centroid(person_bbox)
    oc = _centroid(object_bbox)

    # Raw distance
    distance = float(np.linalg.norm(pc - oc))

    # IoU
    a, b = person_bbox, object_bbox
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter > 0:
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        iou    = inter / (area_a + area_b - inter)
    else:
        iou = 0.0

    # Object velocity (raw px/frame)
    if prev_object_bbox is not None:
        prev_oc  = _centroid(prev_object_bbox)
        velocity = float(np.linalg.norm(oc - prev_oc))
    else:
        velocity = 0.0

    # Stationary time — count frames in sequence where obj velocity was low
    stationary_time = 0
    if sequence is not None and sequence.shape[0] > 0:
        obj_vels        = sequence[:, 2]   # feature index 2 = obj_velocity_norm
        stationary_time = int((obj_vels < 0.3).sum())

    # Object area (raw px²)
    object_area = _bbox_area(object_bbox)

    # Event count = sum of interaction events
    event_count = held_frames + released_frames

    return {
        "distance":         distance,
        "iou":              iou,
        "velocity":         velocity,
        "interaction_time": interaction_time,
        "stationary_time":  stationary_time,
        "object_area":      object_area,
        "vehicle_present":  1 if vehicle_present else 0,
        "event_count":      event_count,
    }