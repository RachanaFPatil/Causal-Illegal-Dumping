"""
Layer 2 — Production ByteTrack with ReID + Failure Detection & ROI Recovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key upgrade over previous tracker.py:

  NEW — ReID Appearance Embedding (fixes viewpoint-change ID switching)
       Each detection crop → 128D feature vector via ReIDEmbedder.
       Combined cost = α·appearance_dist + β·(1-IoU) + γ·motion_dist
       Embedding is EMA-smoothed per track → stable across viewpoint change.
       This is the PRIMARY fix for "same person → two IDs after angle change".

  All other modules unchanged:
       ByteTrack 3-stage cascade, Kalman filter, Failure Detection, ROI Re-Scan,
       class-lock, ghost-throw cooldown, Layer 3 compatible output.
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

        self.bbox        = bbox.astype(np.float32)
        self.conf        = conf
        self.class_name  = class_name
        self.locked_class: Optional[str] = None

        self.state       = _State.TENTATIVE
        self.age         = 1
        self.hits        = 1
        self.missed      = 0
        self._predict_age = 0

        self._velocity   = np.zeros(2, dtype=np.float32)
        self._prev_centroid: Optional[np.ndarray] = None

        cx, cy = self._centroid_from_bbox(bbox)
        self._kalman     = KalmanFilter2D(cx, cy)

        self._uncertainty_score:  float = 0.0
        self._sudden_loss_flag:   bool  = False
        self._recoverable_frames: int   = 0

        # ReID embedding (set externally by tracker after crop extraction)
        self.embedding: Optional[np.ndarray] = None

    @staticmethod
    def _centroid_from_bbox(bbox: np.ndarray) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def centroid(self) -> np.ndarray:
        cx, cy = self._centroid_from_bbox(self.bbox)
        return np.array([cx, cy], dtype=np.float32)

    def predicted_bbox(self) -> np.ndarray:
        pred_pos = self._kalman.predict()
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return np.array([
            pred_pos[0] - w / 2, pred_pos[1] - h / 2,
            pred_pos[0] + w / 2, pred_pos[1] + h / 2,
        ], dtype=np.float32)

    def update(self, bbox: np.ndarray, conf: float, class_name: str) -> None:
        curr = np.array(self._centroid_from_bbox(bbox), dtype=np.float32)
        self._kalman.update(curr[0], curr[1])

        if self._prev_centroid is not None:
            raw_vel        = curr - self._prev_centroid
            self._velocity = 0.4 * raw_vel + 0.6 * self._velocity
        self._prev_centroid = curr

        self.bbox = bbox.astype(np.float32)
        self.conf = conf
        if not (LOCK_CLASS_ON_CONFIRM and self.locked_class):
            self.class_name = class_name

        self.age         += 1
        self.hits        += 1
        self.missed       = 0
        self._predict_age = 0
        self._sudden_loss_flag   = False
        self._recoverable_frames = 0
        self._uncertainty_score  = 0.0

        if self.state == _State.TENTATIVE and self.hits >= MIN_CONFIRM_FRAMES:
            self.state        = _State.CONFIRMED
            self.locked_class = self.class_name
        elif self.state in (_State.LOST, _State.RECOVERABLE):
            self.state = _State.CONFIRMED

    def mark_lost(self, is_sudden: bool = False) -> None:
        self.missed      += 1
        self.age         += 1
        self.hits         = 0
        self._predict_age += 1

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
        motion_component = min(
            self._kalman.last_innovation / (MOTION_ERROR_THRESH_PX + 1e-6), 1.0
        )
        missed_component = min(self.missed / max(MISSED_FRAMES_SPIKE, 1), 1.0)
        sudden_component = 1.0 if self._sudden_loss_flag else 0.0

        score = (
            MOTION_ERROR_WEIGHT  * motion_component
            + MISSED_FRAMES_WEIGHT * missed_component
            + SUDDEN_LOSS_WEIGHT   * sudden_component
        )
        self._uncertainty_score = float(np.clip(score, 0.0, 1.0))
        return self._uncertainty_score

    @property
    def is_active(self)    -> bool: return self.state != _State.DELETED
    @property
    def is_confirmed(self) -> bool: return self.state in (_State.CONFIRMED, _State.LOST, _State.RECOVERABLE)
    @property
    def is_lost(self)      -> bool: return self.state in (_State.LOST, _State.RECOVERABLE)
    @property
    def use_prediction(self) -> bool: return self.is_lost and self._predict_age <= PREDICT_FRAMES
    @property
    def is_uncertain(self) -> bool: return self._uncertainty_score >= UNCERTAINTY_THRESHOLD


# ── Geometry utilities ─────────────────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (a[:, i] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i] for i in range(4))
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter  = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _single_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


def _centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
    cb = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
    return float(np.linalg.norm(ca - cb))


def _crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Safely crop bbox from frame. Returns None if out-of-bounds."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < REID_MIN_CROP_SIZE or y2 - y1 < REID_MIN_CROP_SIZE:
        return None
    return frame[y1:y2, x1:x2]


# ── Hungarian matching with combined cost ──────────────────────────────────────

def _match_hungarian(
    tracks:          List[_InternalTrack],
    dets:            List[Detection],
    iou_thresh:      float,
    use_prediction:  bool = False,
    same_class:      bool = True,
    det_embeddings:  Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian assignment using combined cost:
        cost = REID_WEIGHT * appearance_dist
             + MOTION_WEIGHT * (1 - iou)
             + IOU_WEIGHT    * (1 - iou)

    When REID_ENABLED=False or embeddings are missing, falls back to IoU-only.
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    if use_prediction:
        track_boxes = np.array([
            t.predicted_bbox() if t.use_prediction else t.bbox
            for t in tracks
        ], dtype=np.float32)
    else:
        track_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)

    det_boxes = np.array([d.bbox for d in dets], dtype=np.float32)
    iou_mat   = _iou_batch(track_boxes, det_boxes)   # (T, D)

    # ── Build appearance distance matrix ──────────────────────────────────────
    if REID_ENABLED and det_embeddings is not None:
        track_embs = [t.embedding for t in tracks]
        appear_mat = ReIDEmbedder.embedding_matrix(track_embs, det_embeddings)
    else:
        appear_mat = np.full((len(tracks), len(dets)), 0.5, dtype=np.float32)

    # ── Combined cost ─────────────────────────────────────────────────────────
    motion_cost  = 1.0 - iou_mat
    cost_mat     = (
        REID_WEIGHT    * appear_mat
        + MOTION_WEIGHT  * motion_cost
        + IOU_WEIGHT     * motion_cost
    )

    # ── Apply pixel-distance guard + class-lock mask ──────────────────────────
    for ti, t in enumerate(tracks):
        for di, d in enumerate(dets):
            dist = _centroid_distance(track_boxes[ti], det_boxes[di])
            if dist > MAX_MATCH_DISTANCE_PX:
                cost_mat[ti, di] = 1.0
                iou_mat[ti, di]  = 0.0
                continue
            if same_class and LOCK_CLASS_ON_CONFIRM and t.locked_class:
                if t.locked_class != d.class_name:
                    cost_mat[ti, di] = 1.0
                    iou_mat[ti, di]  = 0.0
                    continue
            # Reject if appearance is too dissimilar (only if embedding exists)
            if (REID_ENABLED
                    and det_embeddings is not None
                    and t.embedding is not None
                    and det_embeddings[di] is not None):
                if appear_mat[ti, di] > REID_MAX_COSINE_DIST:
                    # Only hard-reject if IoU is also weak (belt-and-suspenders)
                    if iou_mat[ti, di] < REID_FALLBACK_IOU:
                        cost_mat[ti, di] = 1.0
                        iou_mat[ti, di]  = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_t: Set[int] = set()
    matched_d: Set[int] = set()
    matched:   List[Tuple[int, int]] = []

    for r, c in zip(row_ind, col_ind):
        # Accept if IoU is sufficient OR appearance is close enough
        iou_ok    = iou_mat[r, c] >= iou_thresh
        appear_ok = (REID_ENABLED
                     and det_embeddings is not None
                     and tracks[r].embedding is not None
                     and det_embeddings[c] is not None
                     and appear_mat[r, c] <= REID_MAX_COSINE_DIST)
        if iou_ok or appear_ok:
            matched.append((r, c))
            matched_t.add(r)
            matched_d.add(c)

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [j for j in range(len(dets))   if j not in matched_d]
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
        vx, vy = track._velocity
        near_left   = x1 < EXIT_MARGIN and vx < 0
        near_right  = x2 > frame_w - EXIT_MARGIN and vx > 0
        near_top    = y1 < EXIT_MARGIN and vy < 0
        near_bottom = y2 > frame_h - EXIT_MARGIN and vy > 0
        return not (near_left or near_right or near_top or near_bottom)


# ── Main Tracker ───────────────────────────────────────────────────────────────

class ByteTrackWrapper:
    """
    Production ByteTrack + ReID + Failure Detection + ROI Recovery.

    Public interface (backward-compatible):
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

    # ─────────────────────────────────────────────────────────────────────────

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
            H, W  = frame_or_shape[0], frame_or_shape[1]

        if self._failure_detector is None:
            self._failure_detector = _FailureDetector(W, H)
        else:
            self._failure_detector.update_frame_size(W, H)

        # ── STAGE 0: Extract ReID embeddings for all detections ───────────────
        # Done once per frame — O(D) crops, not per-track.
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

        # ════ STAGE 1: High-conf dets ↔ Confirmed active (with ReID) ═════════
        matched_1, unmatched_conf, unmatched_high = _match_hungarian(
            confirmed_active, high_dets,
            iou_thresh=TRACK_MATCH_THRESH,
            use_prediction=False,
            same_class=True,
            det_embeddings=high_embs,
        )
        for ti, di in matched_1:
            confirmed_active[ti].update(high_dets[di].bbox, high_dets[di].confidence, high_dets[di].class_name)
            # Update track's EMA embedding with matched detection's crop
            if REID_ENABLED and frame is not None and high_embs[di] is not None:
                confirmed_active[ti].embedding = self._reid.update_track(
                    confirmed_active[ti].track_id,
                    _crop(frame, high_dets[di].bbox) or np.zeros((1,1,3), dtype=np.uint8)
                )

        # ════ STAGE 2: Low-conf dets ↔ Unmatched confirmed ═══════════════════
        remaining_conf = [confirmed_active[i] for i in unmatched_conf]
        matched_2, still_unmatched_conf, _ = _match_hungarian(
            remaining_conf, low_dets,
            iou_thresh=TRACK_SECOND_THRESH,
            use_prediction=False,
            same_class=True,
            det_embeddings=low_embs,
        )
        for ti, di in matched_2:
            remaining_conf[ti].update(low_dets[di].bbox, low_dets[di].confidence, low_dets[di].class_name)

        for i in still_unmatched_conf:
            t = remaining_conf[i]
            is_sudden = self._failure_detector.detect_sudden_loss(t, W, H)
            t.mark_lost(is_sudden=is_sudden)

        # ════ STAGE 3: Unmatched high-conf dets ↔ Lost tracks (ReID!) ════════
        unmatched_high_dets = [high_dets[i] for i in unmatched_high]
        unmatched_high_embs = [high_embs[i] for i in unmatched_high]
        matched_3, unmatched_lost, unmatched_new = _match_hungarian(
            lost_tracks, unmatched_high_dets,
            iou_thresh=TRACK_SECOND_THRESH,
            use_prediction=True,
            same_class=True,
            det_embeddings=unmatched_high_embs,
        )
        for ti, di in matched_3:
            lost_tracks[ti].update(
                unmatched_high_dets[di].bbox,
                unmatched_high_dets[di].confidence,
                unmatched_high_dets[di].class_name,
            )
            if REID_ENABLED and frame is not None and unmatched_high_embs[di] is not None:
                lost_tracks[ti].embedding = self._reid.update_track(
                    lost_tracks[ti].track_id,
                    _crop(frame, unmatched_high_dets[di].bbox) or np.zeros((1,1,3), dtype=np.uint8)
                )

        for i in unmatched_lost:
            lost_tracks[i].mark_lost()

        # ════ STAGE 4: Tentative tracks ↔ Remaining unmatched high-conf ══════
        truly_new_dets = [unmatched_high_dets[i] for i in unmatched_new]
        truly_new_embs = [unmatched_high_embs[i] for i in unmatched_new]
        matched_4, unmatched_tent, unmatched_birth = _match_hungarian(
            tentative_tracks, truly_new_dets,
            iou_thresh=TRACK_MATCH_THRESH,
            use_prediction=False,
            same_class=False,
            det_embeddings=truly_new_embs,
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
                # Seed embedding immediately on birth
                if REID_ENABLED and frame is not None and truly_new_embs[i] is not None:
                    new_t.embedding = self._reid.update_track(
                        new_t.track_id,
                        _crop(frame, d.bbox) or np.zeros((1,1,3), dtype=np.uint8)
                    )
                self._tracks.append(new_t)

        # ════ STAGE 6: Failure detection ══════════════════════════════════════
        uncertain_internal = self._failure_detector.score_tracks(self._tracks)

        # ════ STAGE 7: ROI Re-Scan (event-triggered) ══════════════════════════
        if self._roi_recovery and frame is not None and uncertain_internal:
            uncertain_proxies = self._build_uncertain_proxies(uncertain_internal)
            recovery_results  = self._roi_recovery.recover(uncertain_proxies, frame)
            self._apply_recovery(recovery_results, uncertain_internal)

        # ════ STAGE 8: Purge deleted ═══════════════════════════════════════════
        deleted_ids = {t.track_id for t in self._tracks if not t.is_active}
        for tid in deleted_ids:
            self._reid.remove_track(tid)
        self._tracks = [t for t in self._tracks if t.is_active]

        # ════ STAGE 9: Build public output ════════════════════════════════════
        output:   List[TrackedObject] = []
        seen_ids: Set[int]            = set()

        for t in self._tracks:
            if not t.is_confirmed:
                continue
            if t.age < MIN_TRACK_FRAMES:
                continue

            tid = t.track_id
            seen_ids.add(tid)

            pub_status = (
                "ACTIVE"      if t.state == _State.CONFIRMED   else
                "RECOVERABLE" if t.state == _State.RECOVERABLE else
                "LOST"
            )

            if tid in self._public_tracks:
                pub               = self._public_tracks[tid]
                pub.bbox          = t.bbox.copy()
                pub.confidence    = t.conf
                pub.class_name    = t.locked_class or t.class_name
                pub.age           = t.age
                pub.missed_frames = t.missed
                pub.confirmed     = t.is_confirmed
                pub.velocity      = t._velocity.copy()
                pub.locked_class  = t.locked_class
                pub.status        = pub_status
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

    def _build_uncertain_proxies(self, uncertain_tracks):
        proxies = []
        for t in uncertain_tracks:
            if t.track_id in self._public_tracks:
                proxies.append(self._public_tracks[t.track_id])
            else:
                proxy = TrackedObject(
                    track_id=t.track_id, bbox=t.bbox.copy(),
                    class_name=t.locked_class or t.class_name,
                    confidence=t.conf, velocity=t._velocity.copy(),
                    locked_class=t.locked_class,
                )
                proxies.append(proxy)
        return proxies

    def _apply_recovery(self, results, uncertain_tracks):
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

    def _tag_trash(self, tracks, trash_detections):
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

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.state == _State.CONFIRMED)

    @property
    def lost_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_lost)