"""
Layer 1 — Robust CCTV AI Engine/detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Upgrades over baseline detector:

  1. Lighting Analyzer      — measures brightness & contrast per frame
  2. Adaptive Preprocessing — applies CLAHE only when lighting is poor
  3. Dual-Stream Detection  — runs RT-DETR on original + CLAHE frame
  4. Detection Fusion       — IoU-based merging, keeps highest-confidence box

Public API is 100% backward-compatible with Layer 2 and all downstream layers.
The only change visible externally is that detections are now more robust under
poor lighting — the output structure (List[Detection]) is identical.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import logging
import numpy as np
import torchvision
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ultralytics import RTDETR

from .config import (
    MODEL_NAME, IMGSZ, CONF_THRESH, IOU_THRESH, DEVICE, KEEP_CLASSES,
    LIGHTING_BRIGHTNESS_THRESH, LIGHTING_CONTRAST_THRESH,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    FUSION_IOU_THRESH,
)

logger = logging.getLogger(__name__)


# ── Data contract (unchanged — Layer 2 depends on this) ─────────────────────

@dataclass
class Detection:
    """Single detection output. Structure is identical to the original."""
    bbox:       np.ndarray   # [x1, y1, x2, y2]  pixel coords
    class_name: str
    confidence: float
    class_id:   int


# ── Lighting analysis ────────────────────────────────────────────────────────

@dataclass
class LightingReport:
    """Holds per-frame lighting metrics and the decision to apply CLAHE."""
    brightness:  float   # mean grayscale intensity  (0–255)
    contrast:    float   # std-dev of grayscale       (0–128 typical)
    needs_clahe: bool    # True when either metric is below its threshold


def analyze_lighting(frame: np.ndarray) -> LightingReport:
    """
    Compute brightness (mean) and contrast (std-dev) from the grayscale frame.

    Decision rule (fully learned-data-driven thresholds, set in config.py):
      needs_clahe = brightness < THRESH  OR  contrast < THRESH

    Args:
        frame: BGR uint8 numpy array (H, W, 3).

    Returns:
        LightingReport with brightness, contrast, and CLAHE decision.
    """
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    needs_clahe = (
        brightness < LIGHTING_BRIGHTNESS_THRESH
        or contrast < LIGHTING_CONTRAST_THRESH
    )
    logger.debug(
        "[Lighting] brightness=%.1f  contrast=%.1f  clahe=%s",
        brightness, contrast, needs_clahe,
    )
    return LightingReport(brightness=brightness, contrast=contrast,
                          needs_clahe=needs_clahe)


# ── Adaptive CLAHE preprocessing ─────────────────────────────────────────────

# Module-level singleton so the cv2.CLAHE object is not recreated every frame.
_clahe_engine: Optional[cv2.CLAHE] = None


def _get_clahe() -> cv2.CLAHE:
    global _clahe_engine
    if _clahe_engine is None:
        _clahe_engine = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE,
        )
    return _clahe_engine


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE per-channel in LAB colour space to boost local contrast
    without shifting hue.  Only the L (luminance) channel is equalised,
    preserving natural colour balance for the downstream detector.

    Args:
        frame: BGR uint8 numpy array.

    Returns:
        CLAHE-enhanced BGR uint8 numpy array (same shape/dtype).
    """
    lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    l_eq       = _get_clahe().apply(l)
    lab_eq     = cv2.merge([l_eq, a, b])
    enhanced   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ── Detection fusion ──────────────────────────────────────────────────────────

def fuse_detections(
    dets_orig:  List[Detection],
    dets_clahe: List[Detection],
    iou_thresh: float = FUSION_IOU_THRESH,
) -> List[Detection]:
    """
    Merge two detection lists (original stream + CLAHE stream) using greedy
    IoU matching.  For each matched pair keep the box with higher confidence.
    Unmatched detections from either stream are kept as-is.

    Matching only happens between detections of the **same class** to avoid
    cross-class suppression (e.g. a person box should never eat a bottle box).

    Args:
        dets_orig:  Detections from the unmodified frame.
        dets_clahe: Detections from the CLAHE-enhanced frame.
        iou_thresh: IoU above which two detections are considered duplicates.

    Returns:
        Fused List[Detection] with no redundant boxes.
    """
    if not dets_clahe:
        return dets_orig
    if not dets_orig:
        return dets_clahe

    # Work on a mutable copy so we don't alter the caller's lists
    merged: List[Detection] = list(dets_orig)
    used_orig  = [False] * len(dets_orig)

    for det_c in dets_clahe:
        best_iou   = 0.0
        best_idx   = -1

        for i, det_o in enumerate(dets_orig):
            if det_o.class_name != det_c.class_name:
                continue
            iou = _iou(det_o.bbox, det_c.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= iou_thresh:
            # Duplicate — keep whichever has higher confidence in merged list
            if det_c.confidence > dets_orig[best_idx].confidence:
                # Replace the orig entry in merged with the clahe detection.
                # Find it by identity (index offset tracking):
                for j, m in enumerate(merged):
                    if m is dets_orig[best_idx]:
                        merged[j] = det_c
                        break
            # else: orig is already in merged and has higher conf → leave it
        else:
            # Genuinely new detection from CLAHE stream — add it
            merged.append(det_c)

    logger.debug(
        "[Fusion] orig=%d  clahe=%d  fused=%d",
        len(dets_orig), len(dets_clahe), len(merged),
    )
    return merged


# ── Main detector ─────────────────────────────────────────────────────────────

class RTDETRDetector:
    """
    Robust CCTV AI Engine wrapping RT-DETR.

    Public interface (unchanged from baseline):
        detector = RTDETRDetector()
        detections: List[Detection] = detector.detect(frame)

    Internal pipeline per frame:
        1. analyze_lighting(frame)          → decides if CLAHE is needed
        2. apply_clahe(frame)               → only when lighting is poor
        3. RT-DETR(original frame)          → stream A detections
        4. RT-DETR(clahe frame)             → stream B detections (if needed)
        5. fuse_detections(A, B)            → merged, deduplicated list
        6. _nms_filter(fused)               → final NMS pass
    """

    def __init__(self):
        logger.info("[Layer1] Loading %s on %s …", MODEL_NAME, DEVICE)
        print(f"[Layer1] Loading {MODEL_NAME} on {DEVICE} …")
        self.model      = RTDETR(MODEL_NAME)
        self.device     = DEVICE
        self.names      = self.model.names          # {id: class_name}
        self._keep_ids  = self._resolve_keep_ids()
        print(f"[Layer1] Tracking classes : {sorted(KEEP_CLASSES)}")
        print(f"[Layer1] Lighting thresh  : brightness<{LIGHTING_BRIGHTNESS_THRESH}"
              f"  contrast<{LIGHTING_CONTRAST_THRESH}")
        print(f"[Layer1] Fusion IoU thresh: {FUSION_IOU_THRESH}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_keep_ids(self) -> set:
        return {cid for cid, name in self.names.items() if name in KEEP_CLASSES}

    def _nms_filter(
        self,
        detections: List[Detection],
        iou_thresh: float = 0.40,
    ) -> List[Detection]:
        """Extra NMS pass — removes any residual duplicates after fusion."""
        if len(detections) < 2:
            return detections

        boxes  = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        keep   = torchvision.ops.nms(boxes, scores, iou_thresh)
        return [detections[i] for i in keep.tolist()]

    def _run_rtdetr(self, frame: np.ndarray) -> List[Detection]:
        """
        Run RT-DETR on a single BGR frame and return filtered detections.
        This is the core inference call — kept pure so it can be called for
        both the original and the CLAHE stream without code duplication.
        """
        results = self.model.predict(
            source  = frame,
            imgsz   = IMGSZ,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = self.device,
            verbose = False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid not in self._keep_ids:
                    continue
                detections.append(Detection(
                    bbox       = box.xyxy[0].cpu().numpy(),
                    class_name = self.names[cid],
                    confidence = float(box.conf[0]),
                    class_id   = cid,
                ))
        return detections

    # ── Public API (backward-compatible) ─────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run the full robust detection pipeline on a single BGR frame.

        Steps:
            1. Lighting analysis   → LightingReport
            2. Adaptive CLAHE      → only if report.needs_clahe
            3. Dual-stream RT-DETR → original stream always; CLAHE stream when enhanced
            4. Fusion              → IoU-based merging
            5. Final NMS           → dedup

        Returns:
            List[Detection] — identical structure to the original detector.
            Layer 2 (ByteTrack) and all downstream layers are unaffected.
        """
        # ── Step 1: lighting check ────────────────────────────────────
        report = analyze_lighting(frame)

        # ── Step 2: adaptive CLAHE ────────────────────────────────────
        clahe_frame: Optional[np.ndarray] = None
        if report.needs_clahe:
            clahe_frame = apply_clahe(frame)
            logger.debug(
                "[CLAHE] applied  brightness=%.1f  contrast=%.1f",
                report.brightness, report.contrast,
            )

        # ── Step 3: stream A — always run on the original frame ───────
        dets_orig = self._run_rtdetr(frame)

        # ── Step 4: stream B — only run when CLAHE was applied ────────
        dets_clahe: List[Detection] = []
        if clahe_frame is not None:
            dets_clahe = self._run_rtdetr(clahe_frame)

        # ── Step 5: fuse both streams ─────────────────────────────────
        fused = fuse_detections(dets_orig, dets_clahe)

        # ── Step 6: final NMS pass ────────────────────────────────────
        final = self._nms_filter(fused)

        logger.debug(
            "[Detect] orig=%d  clahe=%d  fused=%d  final=%d",
            len(dets_orig), len(dets_clahe), len(fused), len(final),
        )
        return final

    # ── Diagnostic helper (optional, does not affect pipeline) ───────────────

    def get_lighting_report(self, frame: np.ndarray) -> LightingReport:
        """
        Expose lighting metrics for external visualisation / logging.
        Layer 2 never calls this; it's available for run.py overlays.
        """
        return analyze_lighting(frame)