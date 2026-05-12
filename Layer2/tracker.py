"""
Layer 2 — Production ByteTrack with ReID + Failure Detection & ROI Recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes applied:
  - Partial visibility (hand-from-car) handled via centroid-proximity fallback
  - Motion-aware matching: velocity-extrapolated bbox used for lost tracks
  - Duplicate person suppression: overlapping person tracks → older ID wins
  - ReID EMA embedding: stabilises ID across viewpoint changes
  - 3-stage ByteTrack cascade preserved exactly
  - All Layer 3 output fields preserved (backward-compatible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from scipy.optimize import linear_sum_assignment

from Layer1.detector       import Detection
from Layer1.trash_detector import TrashDetection
from .track_state  import TrackedObject, KalmanFilter2D
from .roi_recovery import ROIRecoveryModule, RecoveryResult
from .reid         import ReIDEmbedder
from .config import (
    TRACK_HIGH_THRESH, TRACK_LOW_THRESH,
    TRACK_MATCH_THRESH, TRACK_SECOND_THRESH,
    NEW_TRACK_THRESH,
    MAX_TIME_LOST, MIN_CONFIRM_FRAMES, MIN_TRACK_FRAMES,
    PREDICT_FRAMES,
    MAX_MATCH_DISTANCE_PX,
    GHOST_COOLDOWN_FRAMES,
    LOCK_CLASS_ON_CONFIRM,
    UNCERTAINTY_THRESHOLD,
    MAX_RECOVERABLE_FRAMES,
    REID_ENABLED, REID_WEIGHT, MOTION_WEIGHT, IOU_WEIGHT,
    REID_MAX_COSINE_DIST, REID_FALLBACK_IOU, REID_MIN_CROP_SIZE,
)

logger = logging.getLogger(__name__)


# ── Track lifecycle states ─────────────────────────────────────────────────────

class _State:
    TENTATIVE   = "tentative"
    CONFIRMED   = "confirmed"
    LOST        = "lost"
    RECOVERABLE = "recoverable"
    DELETED     = "deleted"


# ── Internal track ─────────────────────────────────────────────────────────────

class _InternalTrack:
    _next_id: int = 1

    def __init__(self, bbox: np.ndarray, conf: float, class_name: str):
        self.track_id    = _InternalTrack._next_id
        _InternalTrack._next_id += 1

        self.bbox         = bbox.copy()
        self.conf         = conf
        self.class_name   = class_name
        self.locked_class: Optional[str] = None
        self.frames_seen  = 1
        self.frames_lost  = 0
        self.is_active    = True

        self.state        = _State.TENTATIVE
        self.age          = 1
        self.hits         = 1
        self.missed       = 0
        self._predict_age = 0

        self.velocity       = np.zeros(4, dtype=np.float32)
        self._velocity      = np.zeros(2, dtype=np.float32)   # (vx, vy) centroid velocity
        self._prev_centroid: Optional[np.ndarray] = None
        self.prev_bbox      = bbox.copy()
        self.size_history   = [self._area(bbox)]

        cx, cy = self._centroid_from_bbox(bbox)
        self._kalman = KalmanFilter2D(cx, cy)

        self._uncertainty_score:  float = 0.0
        self._sudden_loss_flag:   bool  = False
        self._recoverable_frames: int   = 0

        self.embedding: Optional[np.ndarray] = None

    @staticmethod
    def _centroid_from_bbox(bbox: np.ndarray) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def _area(bbox: np.ndarray) -> float:
        return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

    def centroid(self) -> np.ndarray:
        cx, cy = self._centroid_from_bbox(self.bbox)
        return np.array([cx, cy], dtype=np.float32)

    def predicted_bbox(self) -> np.ndarray:
        """Velocity-extrapolated bbox for use during lost frames."""
        return self.bbox + self.velocity

    def max_historical_area(self) -> float:
        return max(self.size_history) if self.size_history else 1.0

    def update(self, bbox: np.ndarray, conf: float, class_name: str) -> None:
        new_cx, new_cy = self._centroid_from_bbox(bbox)
        old_cx, old_cy = self._centroid_from_bbox(self.bbox)
        self._velocity  = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float32)
        self.velocity   = bbox - self.bbox
        self.prev_bbox  = self.bbox.copy()
        self.bbox       = bbox.copy()
        self.conf       = conf

        # Class lock: once confirmed, class_name cannot flip
        if self.locked_class is None:
            self.class_name = class_name
        else:
            self.class_name = self.locked_class

        self.frames_seen  += 1
        self.frames_lost   = 0
        self.is_active     = True
        self.hits         += 1
        self.missed        = 0
        self.age          += 1
        self._predict_age  = 0
        self._sudden_loss_flag = False

        self.size_history.append(self._area(bbox))
        if len(self.size_history) > 30:
            self.size_history.pop(0)

        self._kalman.update(new_cx, new_cy)

        # Promote tentative → confirmed after enough hits
        if self.state == _State.TENTATIVE and self.hits >= MIN_CONFIRM_FRAMES:
            self.state = _State.CONFIRMED
            if LOCK_CLASS_ON_CONFIRM and self.locked_class is None:
                self.locked_class = self.class_name

        # Re-activate recovered track
        if self.state in (_State.LOST, _State.RECOVERABLE):
            self.state = _State.CONFIRMED
            self._recoverable_frames = 0

    def mark_lost(self, is_sudden: bool = False) -> None:
        self.missed       += 1
        self.age          += 1
        self.hits          = 0
        self._predict_age += 1
        self.bbox          = self.bbox + self.velocity * 0.5   # decay-extrapolate
        self.frames_lost  += 1
        self.is_active     = False

        if is_sudden:
            self._sudden_loss_flag = True

        self._kalman.predict()

        if self.missed > MAX_TIME_LOST:
            self.state = _State.DELETED
        elif self.state == _State.CONFIRMED:
            self.state = _State.LOST
        elif self.state == _State.TENTATIVE:
            self.state = _State.DELETED
        elif self.state == _State.RECOVERABLE:
            self._recoverable_frames += 1
            if self._recoverable_frames > MAX_RECOVERABLE_FRAMES:
                self.state = _State.LOST

    def compute_uncertainty(self) -> float:
        from .config import (
            MOTION_ERROR_WEIGHT, MISSED_FRAMES_WEIGHT, SUDDEN_LOSS_WEIGHT,
            MISSED_FRAMES_SPIKE, MOTION_ERROR_THRESH_PX,
        )
        motion_component  = min(self._kalman.last_innovation / (MOTION_ERROR_THRESH_PX + 1e-6), 1.0)
        missed_component  = min(self.missed / max(MISSED_FRAMES_SPIKE, 1), 1.0)
        sudden_component  = 1.0 if self._sudden_loss_flag else 0.0

        score = (
            MOTION_ERROR_WEIGHT  * motion_component
            + MISSED_FRAMES_WEIGHT * missed_component
            + SUDDEN_LOSS_WEIGHT   * sudden_component
        )
        self._uncertainty_score = float(np.clip(score, 0.0, 1.0))
        return self._uncertainty_score

    @property
    def is_active(self) -> bool:
        return self.state != _State.DELETED

    @is_active.setter
    def is_active(self, value: bool) -> None:
        # Allow external code to set is_active = False to force deletion
        if not value:
            if self.state not in (_State.DELETED,):
                pass  # actual state transition handled by mark_lost / suppress

    @property
    def is_confirmed(self) -> bool:
        return self.state in (_State.CONFIRMED, _State.LOST, _State.RECOVERABLE)

    @property
    def is_lost(self) -> bool:
        return self.state in (_State.LOST, _State.RECOVERABLE)

    @property
    def use_prediction(self) -> bool:
        return self.is_lost and self._predict_age <= PREDICT_FRAMES

    @property
    def is_uncertain(self) -> bool:
        return self._uncertainty_score >= UNCERTAINTY_THRESHOLD


# ── Geometry utilities ─────────────────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i] for i in range(4))
    ix1   = np.maximum(ax1[:, None], bx1[None, :])
    iy1   = np.maximum(ay1[:, None], by1[None, :])
    ix2   = np.minimum(ax2[:, None], bx2[None, :])
    iy2   = np.minimum(ay2[:, None], by2[None, :])
    inter  = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _single_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1   = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2   = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


def _centroid_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ca   = np.stack([(boxes_a[:,0]+boxes_a[:,2])/2, (boxes_a[:,1]+boxes_a[:,3])/2], axis=1)
    cb   = np.stack([(boxes_b[:,0]+boxes_b[:,2])/2, (boxes_b[:,1]+boxes_b[:,3])/2], axis=1)
    diff = ca[:, None, :] - cb[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def _crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < REID_MIN_CROP_SIZE or y2 - y1 < REID_MIN_CROP_SIZE:
        return None
    return frame[y1:y2, x1:x2]



# ── Hungarian matching with combined cost ──────────────────────────────────────

def _match_hungarian(
    tracks:         List[_InternalTrack],
    dets:           List[Detection],
    iou_thresh:     float,
    use_prediction: bool                        = False,
    same_class:     bool                        = True,
    det_embeddings: Optional[List]              = None,
    track_embeddings: Optional[List]            = None,
) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Multi-cue Hungarian matching.

    Cost = IOU_WEIGHT*(1-iou) + MOTION_WEIGHT*centroid_norm + REID_WEIGHT*reid_dist

    For partially-visible detections (hand-from-car):
      - centroid proximity acts as fallback when IoU is near zero
      - motion_ok flag allows match purely on centroid for person class
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    track_boxes = np.array([
        t.predicted_bbox() if (use_prediction and t.frames_lost > 0) else t.bbox
        for t in tracks
    ])
    det_boxes = np.array([d.bbox for d in dets])

    iou_mat  = _iou_batch(track_boxes, det_boxes)
    cdist    = _centroid_distance_matrix(track_boxes, det_boxes)

    # Normalise centroid distance to [0,1] using diagonal of frame
    frame_diag = float(np.sqrt(track_boxes[:,2].max()**2 + track_boxes[:,3].max()**2)) + 1e-6
    motion_mat = np.clip(cdist / (frame_diag * 0.3), 0.0, 1.0)
    motion_score = 1.0 - motion_mat   # higher = closer

    # ReID distance matrix
    if REID_ENABLED and det_embeddings and track_embeddings:
        reid_mat = ReIDEmbedder.embedding_matrix(track_embeddings, det_embeddings)
    else:
        reid_mat = np.full((len(tracks), len(dets)), 0.5, dtype=np.float32)

    # Combined cost (lower = better match)
    # For zero-IoU partial detections, fall back to motion + reid
    iou_contribution    = IOU_WEIGHT    * (1.0 - iou_mat)
    motion_contribution = MOTION_WEIGHT * motion_mat
    reid_contribution   = REID_WEIGHT   * reid_mat

    cost_mat = iou_contribution + motion_contribution + reid_contribution

    # Gate: never match across classes if same_class=True
    if same_class:
        for i, t in enumerate(tracks):
            t_cls = t.locked_class or t.class_name
            for j, d in enumerate(dets):
                if t_cls != d.class_name:
                    cost_mat[i, j] = 1e9

    # Gate: never match beyond max pixel distance
    cost_mat[cdist > MAX_MATCH_DISTANCE_PX] = 1e9

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_r: Set[int] = set()
    matched_c: Set[int] = set()
    matched:   List[Tuple[int,int]] = []

    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] >= 1e9:
            continue
        iou_ok    = iou_mat[r, c] >= iou_thresh
        # Partial-visibility fallback: centroid close enough + same person class
        motion_ok = (
            motion_score[r, c] >= 0.5
            and (tracks[r].locked_class or tracks[r].class_name) == "person"
            and (det_embeddings is None or reid_mat[r, c] < REID_MAX_COSINE_DIST + 0.2)
        )
        # ReID-primary match for lost tracks (even with low IoU)
        reid_ok = (
            REID_ENABLED
            and reid_mat[r, c] < REID_MAX_COSINE_DIST
            and iou_mat[r, c] >= REID_FALLBACK_IOU
        )

        if iou_ok or motion_ok or reid_ok:
            matched.append((r, c))
            matched_r.add(r)
            matched_c.add(c)

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_r]
    unmatched_d = [j for j in range(len(dets))   if j not in matched_c]
    return matched, unmatched_t, unmatched_d


# ── Failure Detector ───────────────────────────────────────────────────────────

class _FailureDetector:
    def __init__(self, frame_w: int, frame_h: int):
        self._frame_w = frame_w
        self._frame_h = frame_h

    def update_frame_size(self, frame_w: int, frame_h: int) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h

    def score_tracks(self, tracks: List[_InternalTrack]) -> List[_InternalTrack]:
        uncertain = []
        for t in tracks:
            if not t.is_confirmed:
                continue
            score = t.compute_uncertainty()
            if score >= UNCERTAINTY_THRESHOLD:
                uncertain.append(t)
        return uncertain

    def detect_sudden_loss(self, track: _InternalTrack, frame_w: int, frame_h: int) -> bool:
        EXIT_MARGIN = 30
        x1, y1, x2, y2 = track.bbox
        vx, vy = float(track._velocity[0]), float(track._velocity[1])
        near_left   = x1 < EXIT_MARGIN       and vx < 0
        near_right  = x2 > frame_w - EXIT_MARGIN and vx > 0
        near_top    = y1 < EXIT_MARGIN        and vy < 0
        near_bottom = y2 > frame_h - EXIT_MARGIN and vy > 0
        at_edge     = near_left or near_right or near_top or near_bottom
        return not at_edge   # True = sudden, unexpected loss


# ── Duplicate Suppressor ───────────────────────────────────────────────────────

def _suppress_duplicate_persons(tracks: List[_InternalTrack]) -> None:
    """
    If two active person tracks overlap (IoU > 0.5), suppress the newer one.
    Newer = higher track_id. Sets state → DELETED so it is purged next cycle.
    """
    person_tracks = [
        t for t in tracks
        if (t.locked_class or t.class_name) == "person"
        and t.state in (_State.CONFIRMED, _State.TENTATIVE)
        and t.frames_lost == 0
    ]
    if len(person_tracks) < 2:
        return

    boxes   = np.array([t.bbox for t in person_tracks])
    iou_mat = _iou_batch(boxes, boxes)
    np.fill_diagonal(iou_mat, 0.0)

    to_delete: Set[int] = set()
    for i in range(len(person_tracks)):
        for j in range(i + 1, len(person_tracks)):
            if iou_mat[i, j] > 0.5:
                newer_idx = i if person_tracks[i].track_id > person_tracks[j].track_id else j
                to_delete.add(newer_idx)

    for idx in to_delete:
        person_tracks[idx].state = _State.DELETED


# ── Main Tracker ───────────────────────────────────────────────────────────────

class ByteTrackWrapper:
    """
    Production ByteTrack + ReID + Failure Detection + ROI Recovery.

    Public interface (backward-compatible with original Layer 2):
        tracker = ByteTrackWrapper(detector=rtdetr_detector)
        tracks  = tracker.update(detections, trash_detections, frame, frame_shape)
    """

    def __init__(self, detector=None):
        self._tracks:            List[_InternalTrack]     = []
        self._public_tracks:     Dict[int, TrackedObject] = {}
        self.total_trash_events: int                      = 0
        self._trash_track_ids:   Set[int]                 = set()
        self._ghost_cooldown:    int                      = 0
        self._frame_count:       int                      = 0

        self._roi_recovery: Optional[ROIRecoveryModule] = (
            ROIRecoveryModule(detector) if detector is not None else None
        )
        self._failure_detector: Optional[_FailureDetector] = None
        self._reid = ReIDEmbedder()

        logger.info(
            "[Layer2] ByteTrackWrapper initialised. ROI recovery: %s  ReID: %s",
            "ENABLED" if self._roi_recovery else "DISABLED",
            "ENABLED" if REID_ENABLED else "DISABLED",
        )

    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        detections:       List[Detection],
        trash_detections: List[TrashDetection],
        frame_or_shape,
        frame_shape:      Optional[Tuple] = None,
    ) -> List[TrackedObject]:

        self._frame_count += 1
        if self._ghost_cooldown > 0:
            self._ghost_cooldown -= 1

        # ── Resolve frame / shape ─────────────────────────────────────────────
        if isinstance(frame_or_shape, np.ndarray):
            frame = frame_or_shape
            H, W  = frame.shape[:2]
        else:
            frame = None
            H, W  = int(frame_or_shape[0]), int(frame_or_shape[1])

        if self._failure_detector is None:
            self._failure_detector = _FailureDetector(W, H)
        else:
            self._failure_detector.update_frame_size(W, H)

        # ── STAGE 0: Extract ReID embeddings for all detections (once/frame) ──
        det_embeddings: List[Optional[np.ndarray]] = []
        if REID_ENABLED and frame is not None:
            for d in detections:
                crop = _crop(frame, d.bbox)
                emb  = self._reid.extract(crop) if crop is not None else None
                det_embeddings.append(emb)
        else:
            det_embeddings = [None] * len(detections)

        # ── Split by confidence ───────────────────────────────────────────────
        high_idx  = [i for i, d in enumerate(detections) if d.confidence >= TRACK_HIGH_THRESH]
        low_idx   = [i for i, d in enumerate(detections)
                     if TRACK_LOW_THRESH <= d.confidence < TRACK_HIGH_THRESH]
        high_dets = [detections[i] for i in high_idx]
        low_dets  = [detections[i] for i in low_idx]
        high_embs = [det_embeddings[i] for i in high_idx]
        low_embs  = [det_embeddings[i] for i in low_idx]

        confirmed_active = [t for t in self._tracks if t.state == _State.CONFIRMED]
        lost_tracks      = [t for t in self._tracks if t.state in (_State.LOST, _State.RECOVERABLE)]
        tentative_tracks = [t for t in self._tracks if t.state == _State.TENTATIVE]

        # Gather track-side embeddings for ReID
        conf_embs = [self._reid.get_embedding(t.track_id) for t in confirmed_active]
        lost_embs = [self._reid.get_embedding(t.track_id) for t in lost_tracks]
        tent_embs = [self._reid.get_embedding(t.track_id) for t in tentative_tracks]

        # ════ STAGE 1: High-conf dets ↔ Confirmed active ═════════════════════
        matched_1, unmatched_conf, unmatched_high = _match_hungarian(
            confirmed_active, high_dets,
            iou_thresh=TRACK_MATCH_THRESH,
            use_prediction=False,
            same_class=True,
            det_embeddings=high_embs,
            track_embeddings=conf_embs,
        )
        for ti, di in matched_1:
            confirmed_active[ti].update(high_dets[di].bbox, high_dets[di].confidence, high_dets[di].class_name)
            if REID_ENABLED and frame is not None and high_embs[di] is not None:
                crop = _crop(frame, high_dets[di].bbox)
                if crop is not None:
                    confirmed_active[ti].embedding = self._reid.update_track(
                        confirmed_active[ti].track_id, crop
                    )

        # ════ STAGE 2: Low-conf dets ↔ Unmatched confirmed ═══════════════════
        remaining_conf      = [confirmed_active[i] for i in unmatched_conf]
        remaining_conf_embs = [conf_embs[i]         for i in unmatched_conf]

        matched_2, still_unmatched_conf, _ = _match_hungarian(
            remaining_conf, low_dets,
            iou_thresh=TRACK_SECOND_THRESH,
            use_prediction=False,
            same_class=True,
            det_embeddings=low_embs,
            track_embeddings=remaining_conf_embs,
        )
        for ti, di in matched_2:
            remaining_conf[ti].update(low_dets[di].bbox, low_dets[di].confidence, low_dets[di].class_name)

        for i in still_unmatched_conf:
            t         = remaining_conf[i]
            is_sudden = self._failure_detector.detect_sudden_loss(t, W, H)
            t.mark_lost(is_sudden=is_sudden)

        # ════ STAGE 3: Unmatched high-conf dets ↔ Lost tracks (ReID!) ════════
        unmatched_high_dets = [high_dets[i] for i in unmatched_high]
        unmatched_high_embs = [high_embs[i] for i in unmatched_high]

        matched_3, unmatched_lost, unmatched_new = _match_hungarian(
            lost_tracks, unmatched_high_dets,
            iou_thresh=TRACK_MATCH_THRESH,
            use_prediction=True,
            same_class=True,
            det_embeddings=unmatched_high_embs,
            track_embeddings=lost_embs,
        )
        for ti, di in matched_3:
            lost_tracks[ti].update(
                unmatched_high_dets[di].bbox,
                unmatched_high_dets[di].confidence,
                unmatched_high_dets[di].class_name,
            )
            if REID_ENABLED and frame is not None and unmatched_high_embs[di] is not None:
                crop = _crop(frame, unmatched_high_dets[di].bbox)
                if crop is not None:
                    lost_tracks[ti].embedding = self._reid.update_track(
                        lost_tracks[ti].track_id, crop
                    )

        for i in unmatched_lost:
            lost_tracks[i].mark_lost()

        # ════ STAGE 4: Tentative tracks ↔ Remaining truly-new dets ══════════
        truly_new_dets = [unmatched_high_dets[i] for i in unmatched_new]
        truly_new_embs = [unmatched_high_embs[i] for i in unmatched_new]

        matched_4, unmatched_tent, unmatched_birth = _match_hungarian(
            tentative_tracks, truly_new_dets,
            iou_thresh=TRACK_MATCH_THRESH,
            use_prediction=False,
            same_class=False,
            det_embeddings=truly_new_embs,
            track_embeddings=tent_embs,
        )
        for ti, di in matched_4:
            tentative_tracks[ti].update(truly_new_dets[di].bbox, truly_new_dets[di].confidence, truly_new_dets[di].class_name)

        for i in unmatched_tent:
            tentative_tracks[i].mark_lost()

        # ════ STAGE 5: Birth new tracks ═══════════════════════════════════════
        for i in unmatched_birth:
            d = truly_new_dets[i]
            if d.confidence >= NEW_TRACK_THRESH:
                new_t = _InternalTrack(d.bbox, d.confidence, d.class_name)
                if REID_ENABLED and frame is not None and truly_new_embs[i] is not None:
                    crop = _crop(frame, d.bbox)
                    if crop is not None:
                        new_t.embedding = self._reid.update_track(new_t.track_id, crop)
                self._tracks.append(new_t)

        # ════ STAGE 5.5: Duplicate person suppression ═════════════════════════
        _suppress_duplicate_persons(self._tracks)

        # ════ STAGE 6: Failure detection ══════════════════════════════════════
        uncertain_internal = self._failure_detector.score_tracks(self._tracks)

        # ════ STAGE 7: ROI Re-Scan (event-triggered) ══════════════════════════
        if self._roi_recovery and frame is not None and uncertain_internal:
            uncertain_proxies = self._build_uncertain_proxies(uncertain_internal)
            recovery_results  = self._roi_recovery.recover(uncertain_proxies, frame)
            self._apply_recovery(recovery_results, uncertain_internal)

        # ════ STAGE 8: Purge deleted ═══════════════════════════════════════════
        deleted_ids = {t.track_id for t in self._tracks if t.state == _State.DELETED}
        for tid in deleted_ids:
            self._reid.remove_track(tid)
        self._tracks = [t for t in self._tracks if t.state != _State.DELETED]

        # ════ STAGE 9: Build public output ════════════════════════════════════
        output:   List[TrackedObject] = []
        seen_ids: Set[int]            = set()

        for t in self._tracks:
            if not t.is_confirmed:
                continue
            if t.age < MIN_TRACK_FRAMES:
                continue
            if not self._is_valid_bbox(t.bbox, (H, W)):  # ← ADD HERE
                continue

            tid = t.track_id
            seen_ids.add(tid)

            pub_status = (
                "ACTIVE"      if t.state == _State.CONFIRMED   else
                "RECOVERABLE" if t.state == _State.RECOVERABLE else
                "LOST"
            )

            if tid in self._public_tracks:
                pub                   = self._public_tracks[tid]
                pub.bbox              = t.bbox.copy()
                pub.confidence        = t.conf
                pub.class_name        = t.locked_class or t.class_name
                pub.age               = t.age
                pub.missed_frames     = t.missed
                pub.confirmed         = t.is_confirmed
                pub.velocity          = t._velocity.copy()
                pub.locked_class      = t.locked_class
                pub.status            = pub_status
                pub.uncertainty_score = t._uncertainty_score
            else:
                pub = TrackedObject(
                    track_id          = tid,
                    bbox              = t.bbox.copy(),
                    class_name        = t.locked_class or t.class_name,
                    confidence        = t.conf,
                    age               = t.age,
                    missed_frames     = t.missed,
                    confirmed         = t.is_confirmed,
                    locked_class      = t.locked_class,
                    velocity          = t._velocity.copy(),
                    status            = pub_status,
                    uncertainty_score = t._uncertainty_score,
                )
                self._public_tracks[tid] = pub

            pub.update_trail()
            output.append(pub)

        for tid in list(self._public_tracks):
            if tid not in seen_ids:
                del self._public_tracks[tid]

        # ════ STAGE 10: Trash tagging ══════════════════════════════════════════
        self._tag_trash(output, trash_detections)

        return output

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_uncertain_proxies(
        self, uncertain_tracks: List[_InternalTrack]
    ) -> List[TrackedObject]:
        proxies = []
        for t in uncertain_tracks:
            if t.track_id in self._public_tracks:
                proxies.append(self._public_tracks[t.track_id])
            else:
                proxy = TrackedObject(
                    track_id     = t.track_id,
                    bbox         = t.bbox.copy(),
                    class_name   = t.locked_class or t.class_name,
                    confidence   = t.conf,
                    velocity     = t._velocity.copy(),
                    locked_class = t.locked_class,
                )
                proxies.append(proxy)
        return proxies

    def _apply_recovery(
        self,
        results:          List[RecoveryResult],
        uncertain_tracks: List[_InternalTrack],
    ) -> None:
        id_to_track = {t.track_id: t for t in uncertain_tracks}
        for result in results:
            t = id_to_track.get(result.track_id)
            if t is None:
                continue
            if result.recovered and result.new_bbox is not None:
                t.update(result.new_bbox, result.confidence, t.locked_class or t.class_name)
                logger.info("[Layer2] ROI recovery SUCCESS  track_id=%d", t.track_id)
            else:
                if t.state == _State.CONFIRMED:
                    t.state = _State.RECOVERABLE
                logger.debug("[Layer2] ROI recovery FAILED  track_id=%d", t.track_id)
    @staticmethod
    def _is_valid_bbox(bbox: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
            x1, y1, x2, y2 = bbox
            h, w = frame_shape[:2]
            if x2 <= x1 or y2 <= y1:
                return False
            if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
                return False
            if (x2 - x1) * (y2 - y1) < 100:
                return False
            return True

    def _tag_trash(
        self,
        tracks:           List[TrackedObject],
        trash_detections: List[TrashDetection],
    ) -> None:
        for trash in trash_detections:
            best_iou, best_track = 0.0, None
            for t in tracks:
                if t.class_name == "person":
                    continue
                iou = _single_iou(trash.bbox, t.bbox)
                if iou > best_iou:
                    best_iou, best_track = iou, t

            if best_track is not None and best_iou > 0.15:
                best_track.is_trash    = True
                best_track.trash_label = trash.label
                best_track.trash_how   = trash.how
                if best_track.track_id not in self._trash_track_ids:
                    self._trash_track_ids.add(best_track.track_id)
                    self.total_trash_events += 1
            else:
                if self._ghost_cooldown == 0:
                    self.total_trash_events += 1
                    self._ghost_cooldown = GHOST_COOLDOWN_FRAMES

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.state == _State.CONFIRMED)

    @property
    def lost_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_lost)