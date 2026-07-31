"""
Layer 1 — Video Calibrator (Universal, All-Layer)
═══════════════════════════════════════════════════
Place this file at the PROJECT ROOT (same level as run_pipeline.py).

HOW TO USE:
  run_pipeline.py already imports and calls this automatically when
  --no-calibrate is NOT passed. No manual action needed.

WHAT IT AUTO-ADJUSTS:
  Layer 1  — CONF_THRESH, IOU_THRESH, TRASH_MIN_AREA, TRASH_DIST2THRESHOLD,
              LIGHTING_BRIGHTNESS_THRESH, LIGHTING_CONTRAST_THRESH
  Layer 2  — TRACK_HIGH_THRESH, TRACK_LOW_THRESH, MATCH_THRESH,
              CENTROID_FB_PX, MAX_TIME_LOST, MIN_TRACK_FRAMES
  Layer 3  — MAX_PAIR_DISTANCE, HOLD_DISTANCE_PX, MAX_MISSING_FRAMES
  Layer 4  — DUMP_THRESHOLD, INFER_EVERY_N
  Layer 5  — BIN_LEGAL_RADIUS_PX

WHAT IT NEVER TOUCHES:
  - Model names / device / class lists  (architectural choices)
  - Layer 4 model architecture params   (D_MODEL, NHEAD, EPOCHS …)
  - Visualisation colours / toggles
  - Normalisation constants             (NORM_DISTANCE etc.)

FALSE-POSITIVE NOTE:
  The calibrator raises CONF_THRESH for well-lit scenes (common in outdoor
  CCTV). This directly helps reduce the false positive issue in the
  kurta-girl and scooter videos.
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── Baseline defaults (used when calibration is skipped or fails) ─────────────
# These mirror Layer1/config.py so no-calibrate runs behave identically.
_DEFAULTS: dict = {
    "layer1": {
        "CONF_THRESH":                 0.30,   # updated from 0.35 — see config.py
        "IOU_THRESH":                  0.45,
        "TRASH_MIN_AREA":              1200,   # updated from 600 — see config.py
        "TRASH_DIST2THRESHOLD":        50.0,
        "LIGHTING_BRIGHTNESS_THRESH":  80,
        "LIGHTING_CONTRAST_THRESH":    40,
        "CLAHE_CLIP_LIMIT":            2.5,
        "FUSION_IOU_THRESH":           0.45,
    },
    "layer2": {
        "TRACK_HIGH_THRESH":   0.35,
        "TRACK_LOW_THRESH":    0.15,
        "MATCH_THRESH":        0.35,
        "CENTROID_FB_PX":      80,
        "MAX_TIME_LOST":       40,
        "MIN_TRACK_FRAMES":    2,
    },
    "layer3": {
        "MAX_PAIR_DISTANCE":   300,
        "HOLD_DISTANCE_PX":    90,
        "MAX_MISSING_FRAMES":  20,
        "SEQUENCE_LENGTH":     24,
    },
    "layer4": {
        "DUMP_THRESHOLD":      0.40,
        "INFER_EVERY_N":       5,
        "MIN_HELD_FRAMES":     8,     # raised from 3 → 8 to kill flickering fp
        "RELEASE_CONFIRM":     6,     # raised from 4 → 6
        "BIN_NEAR_PX":         200,
        "NEAR_PERSON_PX":      150,
    },
    "layer5": {
        "BIN_LEGAL_RADIUS_PX": 210,
    },
}

logger = logging.getLogger("calibrator")

CAL_FRAMES = 60
CAL_SKIP   = 2

_REF_W, _REF_H = 960, 540


@dataclass
class CalibrationResult:
    cfg:            dict
    fps:            float = 30.0
    resolution:     tuple = (960, 540)
    scale:          float = 1.0
    brightness:     float = 128.0
    contrast:       float = 50.0
    median_conf:    float = 0.50
    median_iou:     float = 0.50
    p75_jump_px:    float = 30.0
    took_sec:       float = 0.0
    warnings:       list  = field(default_factory=list)


def calibrate(
    source,
    detector,
    cal_frames: int  = CAL_FRAMES,
    cal_skip:   int  = CAL_SKIP,
    verbose:    bool = True,
) -> CalibrationResult:
    """
    Run the calibration pass and return a CalibrationResult.

    Parameters
    ----------
    source    : video path (str) or camera index (int)
    detector  : Layer1 RTDETRDetector instance (already loaded)
    cal_frames: how many frames to sample
    cal_skip  : read every nth frame
    verbose   : print calibration report
    """
    t0 = time.time()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.warning("[Calibrator] Cannot open source — using defaults.")
        return CalibrationResult(cfg=_clone_defaults())

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    diag     = math.hypot(W, H)
    ref_diag = math.hypot(_REF_W, _REF_H)
    scale    = diag / ref_diag

    brightnesses:   list = []
    contrasts:      list = []
    confs:          list = []
    ious:           list = []
    centroid_jumps: list = []
    obj_areas:      list = []
    pair_dists:     list = []

    cap2 = cv2.VideoCapture(source)
    prev_dets: dict = {}
    sampled = 0

    # Skip to 20% to avoid static intro frames
    total_frames_cap = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames_cap * 0.20)
    if start_frame > 0:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while sampled < cal_frames:
        for _ in range(cal_skip):
            ret, frame = cap2.read()
            if not ret:
                break
        if not ret:
            break

        sampled += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(gray.mean()))
        contrasts.append(float(gray.std()))

        if sampled % 2 == 0:
            try:
                dets = detector.detect(frame)
                if dets is None:
                    dets = []
                if not isinstance(dets, (list, tuple)):
                    dets = [dets]
            except Exception as e:
                logger.debug(f"[Calibrator] detector failed: {e}")
                dets = []

            for d in dets:
                if not (hasattr(d, "bbox") and hasattr(d, "class_name")
                        and hasattr(d, "confidence")):
                    continue
                bbox = d.bbox
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox[:4]
                if x2 <= x1 or y2 <= y1:
                    continue
                confs.append(float(d.confidence))
                if d.class_name != "person":
                    obj_areas.append(float((x2 - x1) * (y2 - y1)))

            # IoU between consecutive frames
            curr_dets: dict = {}
            for d in dets:
                if hasattr(d, "class_name") and hasattr(d, "bbox"):
                    curr_dets.setdefault(d.class_name, []).append(d.bbox)
            for cls, bboxes in curr_dets.items():
                if cls in prev_dets:
                    for b1 in bboxes:
                        for b2 in prev_dets[cls]:
                            iou = _iou(b1, b2)
                            if iou > 0.30:
                                area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                                area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                                if area1 < 0.002*(W*H) or area2 < 0.002*(W*H):
                                    continue
                                cx1 = (b1[0]+b1[2])/2; cy1 = (b1[1]+b1[3])/2
                                cx2 = (b2[0]+b2[2])/2; cy2 = (b2[1]+b2[3])/2
                                jump = math.hypot(cx1-cx2, cy1-cy2)
                                ious.append(iou)
                                centroid_jumps.append(jump)
            prev_dets = curr_dets

            persons = [d for d in dets if hasattr(d,"class_name") and d.class_name=="person"]
            objects = [d for d in dets if hasattr(d,"class_name") and d.class_name!="person"]
            for p in persons:
                pcx = (p.bbox[0]+p.bbox[2])/2; pcy = (p.bbox[1]+p.bbox[3])/2
                nearest = None
                for o in objects:
                    ocx = (o.bbox[0]+o.bbox[2])/2; ocy = (o.bbox[1]+o.bbox[3])/2
                    dist = math.hypot(pcx-ocx, pcy-ocy)
                    if nearest is None or dist < nearest:
                        nearest = dist
                if nearest is not None:
                    pair_dists.append(nearest)

    cap2.release()

    # ── Derived statistics ────────────────────────────────────────────────────
    brightness   = float(np.median(brightnesses)) if brightnesses else 128.0
    contrast     = float(np.median(contrasts))    if contrasts    else 50.0
    is_dark      = brightness < 80
    is_glary     = contrast   < 35

    median_conf  = float(np.percentile(confs, 25))          if confs          else 0.50
    median_iou   = float(np.median(ious))                    if ious           else 0.60
    p75_jump     = float(np.percentile(centroid_jumps, 75))  if centroid_jumps else 30.0
    median_pdist = float(np.median(pair_dists))              if pair_dists     else 200.0

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    # For well-lit scenes (not dark, not glary) cap at 0.45 to kill fp.
    # NOTE: never go below 0.28 — any lower and background starts firing.
    if is_dark or is_glary:
        conf_thresh = max(0.25, median_conf * 0.55)
    else:
        conf_thresh = float(np.clip(median_conf * 0.60, 0.28, 0.45))

    iou_thresh     = float(np.clip(0.45 - 0.1*(median_pdist/(scale*300)), 0.25, 0.50))
    # TRASH_MIN_AREA: at least 1200px² — raised baseline from 600
    trash_min_area = max(1200, int(600 * scale**2))
    trash_d2       = float(np.clip(50.0*(contrast/50.0), 20.0, 100.0))
    light_brightness_thresh = int(np.clip(brightness*0.65, 50, 110))
    light_contrast_thresh   = int(np.clip(contrast*0.70, 25, 60))
    clahe_clip     = float(np.clip(3.5 - contrast/25.0, 1.5, 4.0))

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    track_high    = float(np.clip(median_conf*0.65, 0.35, 0.50))
    track_low     = float(np.clip(track_high*0.50, 0.10, 0.22))
    match_thresh  = float(np.clip(median_iou*0.38, 0.20, 0.36))
    centroid_fb   = int(np.clip(p75_jump*3.5, 60, 220))
    max_time_lost = int(fps*2.5)
    min_track_frames = max(5, int(fps*0.20))

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    max_pair_dist  = int(300*scale)
    hold_dist      = int(np.clip(90*scale, 70, 160))
    max_missing    = int(fps*0.70)
    sequence_length = 24

    # ── Layer 4 ───────────────────────────────────────────────────────────────
    dump_thresh   = 0.35 if (is_dark or is_glary) else 0.40
    infer_n       = max(1, min(8, int(fps/6)))
    min_held      = max(8, int(fps*0.30))    # raised: 0.30s minimum
    release_confirm = max(6, int(fps*0.25))  # raised: 0.25s minimum
    bin_near_px   = int(200*scale)
    near_person_px = int(150*scale)

    # ── Layer 5 ───────────────────────────────────────────────────────────────
    bin_legal_radius = int(210*scale)

    cfg = {
        "layer1": {
            "CONF_THRESH":                conf_thresh,
            "IOU_THRESH":                 iou_thresh,
            "TRASH_MIN_AREA":             trash_min_area,
            "TRASH_DIST2THRESHOLD":       trash_d2,
            "LIGHTING_BRIGHTNESS_THRESH": light_brightness_thresh,
            "LIGHTING_CONTRAST_THRESH":   light_contrast_thresh,
            "CLAHE_CLIP_LIMIT":           clahe_clip,
            "FUSION_IOU_THRESH":          float(np.clip(iou_thresh+0.05, 0.30, 0.55)),
        },
        "layer2": {
            "TRACK_HIGH_THRESH":  track_high,
            "TRACK_LOW_THRESH":   track_low,
            "MATCH_THRESH":       match_thresh,
            "CENTROID_FB_PX":     centroid_fb,
            "MAX_TIME_LOST":      max_time_lost,
            "MIN_TRACK_FRAMES":   min_track_frames,
        },
        "layer3": {
            "MAX_PAIR_DISTANCE":  max_pair_dist,
            "HOLD_DISTANCE_PX":   hold_dist,
            "MAX_MISSING_FRAMES": max_missing,
            "SEQUENCE_LENGTH":    sequence_length,
        },
        "layer4": {
            "DUMP_THRESHOLD":     dump_thresh,
            "INFER_EVERY_N":      infer_n,
            "MIN_HELD_FRAMES":    min_held,
            "RELEASE_CONFIRM":    release_confirm,
            "BIN_NEAR_PX":        bin_near_px,
            "NEAR_PERSON_PX":     near_person_px,
        },
        "layer5": {
            "BIN_LEGAL_RADIUS_PX": bin_legal_radius,
        },
    }

    took = time.time() - t0

    result = CalibrationResult(
        cfg=cfg, fps=fps, resolution=(W,H), scale=scale,
        brightness=brightness, contrast=contrast,
        median_conf=median_conf, median_iou=median_iou,
        p75_jump_px=p75_jump, took_sec=took,
    )

    if verbose:
        _print_report(result, is_dark, is_glary)

    return result


def apply(cfg: dict) -> None:
    """
    Patch each layer's module-level config variables in-place.
    Call once, immediately after calibrate(), before constructing any Layer objects.
    """
    _patch_module("Layer1.config", cfg.get("layer1", {}))
    _patch_module("Layer2.config", cfg.get("layer2", {}))
    _patch_module("Layer3.config", cfg.get("layer3", {}))
    _patch_module("Layer4.config", cfg.get("layer4", {}))
    _patch_module("Layer5.config", cfg.get("layer5", {}))

    try:
        import Layer2.tracker as _trk
        l2 = cfg.get("layer2", {})
        if hasattr(_trk, "MATCH_THRESH"):
            _trk.MATCH_THRESH = l2.get("MATCH_THRESH", _trk.MATCH_THRESH)
        if hasattr(_trk, "CENTROID_FALLBACK_PX"):
            _trk.CENTROID_FALLBACK_PX = l2.get("CENTROID_FB_PX",
                                                _trk.CENTROID_FALLBACK_PX)
    except Exception:
        pass

    try:
        import Layer5.agent as _l5
        l5 = cfg.get("layer5", {})
        if hasattr(_l5, "BIN_LEGAL_RADIUS_PX"):
            _l5.BIN_LEGAL_RADIUS_PX = l5.get("BIN_LEGAL_RADIUS_PX",
                                              _l5.BIN_LEGAL_RADIUS_PX)
    except Exception:
        pass


def defaults() -> dict:
    return _clone_defaults()


def _patch_module(module_path: str, overrides: dict) -> None:
    try:
        import importlib, sys
        if module_path in sys.modules:
            mod = sys.modules[module_path]
        else:
            mod = importlib.import_module(module_path)
        for k, v in overrides.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
    except Exception as e:
        logger.debug(f"[Calibrator] Could not patch {module_path}: {e}")


def _clone_defaults() -> dict:
    import copy
    return copy.deepcopy(_DEFAULTS)


def _iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1)*max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-6)


def _print_report(r: CalibrationResult, is_dark: bool, is_glary: bool) -> None:
    w, h = r.resolution
    flags = ""
    if is_dark:  flags += "  ⚠ DARK"
    if is_glary: flags += "  ⚠ LOW-CONTRAST"
    c = r.cfg
    print("[Calibrator] " + "─"*56)
    print(f"[Calibrator] Source     : {w}×{h}  FPS={r.fps:.1f}  scale={r.scale:.2f}×{flags}")
    print(f"[Calibrator] Lighting   : brightness={r.brightness:.0f}  contrast={r.contrast:.0f}")
    print(f"[Calibrator] Detections : median_conf={r.median_conf:.2f}  "
          f"median_iou={r.median_iou:.3f}  p75_jump={r.p75_jump_px:.1f}px")
    print("[Calibrator] " + "─"*56)
    l1,l2,l3,l4,l5 = c["layer1"],c["layer2"],c["layer3"],c["layer4"],c["layer5"]
    print(f"[Calibrator] L1 CONF_THRESH           → {l1['CONF_THRESH']:.2f}")
    print(f"[Calibrator] L1 IOU_THRESH            → {l1['IOU_THRESH']:.2f}")
    print(f"[Calibrator] L1 TRASH_MIN_AREA        → {l1['TRASH_MIN_AREA']} px²")
    print(f"[Calibrator] L1 CLAHE_CLIP            → {l1['CLAHE_CLIP_LIMIT']:.1f}")
    print(f"[Calibrator] L2 TRACK_HIGH_THRESH     → {l2['TRACK_HIGH_THRESH']:.2f}")
    print(f"[Calibrator] L2 MATCH_THRESH          → {l2['MATCH_THRESH']:.2f}")
    print(f"[Calibrator] L2 MAX_TIME_LOST         → {l2['MAX_TIME_LOST']} frames")
    print(f"[Calibrator] L3 MAX_PAIR_DISTANCE     → {l3['MAX_PAIR_DISTANCE']}px")
    print(f"[Calibrator] L4 MIN_HELD_FRAMES       → {l4['MIN_HELD_FRAMES']} frames")
    print(f"[Calibrator] L4 RELEASE_CONFIRM       → {l4['RELEASE_CONFIRM']} frames")
    print(f"[Calibrator] L5 BIN_LEGAL_RADIUS_PX   → {l5['BIN_LEGAL_RADIUS_PX']}px")
    print(f"[Calibrator] Took {r.took_sec:.1f}s")
    print("[Calibrator] " + "─"*56)