"""
Layer 1.5 — Enhancement & Plate Detection
==========================================
Sits between Layer 4 (Dumping Inference) and Layer 5 (Agent).
Triggered ONLY when Layer 4 fires an "illegal_dumping" event.

Pipeline per event:
  1. Scan a best-frame window using Laplacian blur scoring
  2. Run fast-alpr on the best frame — detects plate bbox + reads text
  3. Re-run OCR on a 2x-upscaled tight plate crop to fix two-line plate errors
  4. If no plate found in full frame, fall back to person-anchored crops
  5. Return EnhancementResult with plate text, confidence, and saved evidence

Install:
  pip install fast-alpr onnxruntime     # CPU — works on M1 via CoreML EP
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Indian plate pattern: KA 05K K5546 → KA05KK5546 ─────────────────────────
INDIAN_PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}$')

# ── Lazy fast-alpr singleton ──────────────────────────────────────────────────
_alpr = None

def _get_alpr():
    global _alpr
    if _alpr is None:
        from fast_alpr import ALPR
        _alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
        )
        logger.info("[Enhancer] fast-alpr loaded (detector + OCR)")
    return _alpr


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EnhancementResult:
    pair_id:           str
    plate_text:        Optional[str] = None
    plate_conf:        float         = 0.0
    blur_score:        float         = 0.0
    best_frame_offset: int           = 0
    frames_scanned:    int           = 0
    saved_paths:       list[str]     = field(default_factory=list)
    elapsed_ms:        int           = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def blur_score(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cap_frame_generator(
    cap: cv2.VideoCapture, n_frames: int = 150
) -> Iterator[np.ndarray]:
    """Yield the next n_frames from cap, then restore the position."""
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    yielded   = 0
    while yielded < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        yielded += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)


def _safe_crop(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    pad: int = 0
) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _is_indian(text: str) -> bool:
    return bool(INDIAN_PLATE_RE.match(_normalise(text)))


# ── Person-anchored crops (fallback) ─────────────────────────────────────────

def _person_anchored_crops(
    frame: np.ndarray, person_bbox: tuple
) -> list[np.ndarray]:
    """
    Three progressively-wider horizontal strips in the lower half of the
    frame — where a nearby vehicle's plate typically appears.
    """
    fh, fw = frame.shape[:2]
    crops  = []
    for top_frac, bot_frac, x_frac in [
        (0.45, 0.75, 0.20),
        (0.50, 0.80, 0.25),
        (0.55, 0.85, 0.30),
    ]:
        c = _safe_crop(
            frame,
            int(fw * x_frac),       int(fh * top_frac),
            int(fw * (1 - x_frac)), int(fh * bot_frac),
        )
        if c is not None and c.size > 0:
            crops.append(c)
    return crops


# ── fast-plate-ocr singleton (OCR-only, no detector) ─────────────────────────
_plate_ocr = None

def _get_plate_ocr():
    """
    Load fast-plate-ocr's LicensePlateRecognizer once.
    This runs OCR directly on a pre-cropped plate image — no detector needed.
    Much more reliable than running alpr.predict() on a tiny crop.
    """
    global _plate_ocr
    if _plate_ocr is None:
        from fast_plate_ocr import LicensePlateRecognizer
        _plate_ocr = LicensePlateRecognizer("global-plates-mobile-vit-v2-model")
        logger.info("[Enhancer] fast-plate-ocr loaded (OCR-only)")
    return _plate_ocr


def _ocr_img(img: np.ndarray) -> tuple[Optional[str], float]:
    ocr = _get_plate_ocr()
    try:
        # fast-plate-ocr expects GRAYSCALE, not RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        pred = ocr.run(gray, return_confidence=True)
        if not pred:
            return None, 0.0
        p    = pred[0]
        text = _normalise(p.plate if hasattr(p, "plate") else str(p))
        raw_conf = p.confidence if hasattr(p, "confidence") else []
        conf = float(np.mean(raw_conf)) if raw_conf else 0.5
        return (text, conf) if text else (None, 0.0)
    except Exception as e:
        logger.debug(f"[Enhancer] _ocr_img error: {e}")
        return None, 0.0


# ── Two-line plate OCR ────────────────────────────────────────────────────────

def _ocr_tight_crop(crop: np.ndarray, tag: str) -> list[tuple[str, float]]:
    """
    Run OCR directly on a pre-cropped plate region using fast-plate-ocr
    (bypasses the detector — crop is already isolated).

    Strategy:
      A) Full crop  — works for single-line plates
      B) Top + bottom halves merged — handles two-line Indian plates
         KA05K (top line) + K5546 (bottom line) → KA05KK5546
    """
    results = []

    # Upscale so the OCR model has enough pixels (target: ~64px height)
    h, w    = crop.shape[:2]
    scale = max(4, int(128 / max(h, 1)))
    upscale = cv2.resize(
        crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
    )
    print(f"[ALPR-TIGHT] {tag} crop={h}x{w} → upscaled={h*scale}x{w*scale}")

    # ── A: full upscaled crop ─────────────────────────────────────────────
    text, conf = _ocr_img(upscale)
    if text and len(text) >= 4:
        print(f"[ALPR-TIGHT] {tag} full → '{text}' conf={conf:.2f}")
        results.append((text, conf))

    # ── B: split into top / bottom halves ────────────────────────────────
    mid        = upscale.shape[0] // 2
    top_half   = upscale[:mid, :]
    bot_half   = upscale[mid:, :]
    merged     = ""
    conf_sum   = 0.0
    hit        = 0

    for hlabel, half in [("top", top_half), ("bot", bot_half)]:
        if half.shape[0] < 8:
            continue
        t, c = _ocr_img(half)
        if t and len(t) >= 2:
            print(f"[ALPR-TIGHT] {tag} {hlabel} → '{t}' conf={c:.2f}")
            merged   += t
            conf_sum += c
            hit      += 1

    if hit > 0 and len(merged) >= 6:
        avg = conf_sum / hit
        print(f"[ALPR-TIGHT] {tag} merged → '{merged}' conf={avg:.2f}")
        results.append((merged, avg))

    return results


# ── Main ALPR runner ──────────────────────────────────────────────────────────

def _run_alpr(
    frame: np.ndarray,
    save_dir: str,
    tag: str,
) -> list[tuple[str, float]]:
    """
    Run fast-alpr on `frame` using alpr.predict() (not draw_predictions).
    For every detected plate bbox, also re-runs OCR on the upscaled tight
    crop to handle two-line Indian plates.
    """
    alpr    = _get_alpr()
    results = []

    try:
        preds = alpr.predict(frame)

        if not preds:
            logger.info(f"[Enhancer] No plates detected [{tag}]")
            return []

        # Save annotated debug image once

        for i, r in enumerate(preds):
            if r.ocr is None:
                continue

            raw_text = _normalise(r.ocr.text)
            raw_conf = float(np.mean(r.ocr.confidence))
            print(f"[ALPR] tag={tag} raw → '{raw_text}' conf={raw_conf:.2f}")

            # Keep the raw full-frame read
            if len(raw_text) >= 4:
                results.append((raw_text, raw_conf))

            # Re-run OCR on the tight plate crop
            try:
                bb = r.detection.bounding_box
                # Handle both .x1/.y1/.x2/.y2 and (x,y,w,h) formats
                if hasattr(bb, "x1"):
                    x1, y1, x2, y2 = (
                        int(bb.x1), int(bb.y1),
                        int(bb.x2), int(bb.y2),
                    )
                else:
                    x1 = int(bb[0]); y1 = int(bb[1])
                    x2 = int(bb[0] + bb[2]); y2 = int(bb[1] + bb[3])

                plate_crop = _safe_crop(frame, x1, y1, x2, y2, pad=4)
                if plate_crop is not None and plate_crop.size > 0:
                    tight_results = _ocr_tight_crop(plate_crop, tag=f"{tag}_{i}")
                    # Replace tight-crop's 0.50 placeholder with the real ALPR confidence
                    tight_results = [(text, raw_conf) for text, _ in tight_results]
                    results.extend(tight_results)

            except Exception as e:
                logger.debug(f"[Enhancer] plate crop error [{tag}]: {e}")

    except Exception as e:
        logger.warning(f"[Enhancer] fast-alpr predict error [{tag}]: {e}")

    return results


# ── Best result selector ──────────────────────────────────────────────────────

def _best_result(candidates):
    if not candidates:
        return None, 0.0
    indian = [(t, c) for t, c in candidates if _is_indian(t)]
    if indian:
        return max(indian, key=lambda x: x[1])
    return max(candidates, key=lambda x: x[1])


# ── Core Enhancer ─────────────────────────────────────────────────────────────

class Enhancer:

    def process_event(
        self,
        frame:       np.ndarray,
        person_bbox: tuple,
        person_id:   str,
        pair_id:     str,
        save_dir:    str                             = "evidence",
        frame_iter:  Optional[Iterator[np.ndarray]] = None,
    ) -> EnhancementResult:

        t0 = time.time()
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        result = EnhancementResult(pair_id=pair_id)

        # ── Step 1: build candidate frames ───────────────────────────────
        frames_to_scan: list[tuple[int, np.ndarray]] = [(0, frame)]
        if frame_iter is not None:
            for offset, f in enumerate(frame_iter, start=1):
                frames_to_scan.append((offset, f))

        result.frames_scanned = len(frames_to_scan)
        logger.info(
            f"[Enhancer] Scanning {result.frames_scanned} frames for "
            f"pair={pair_id}"
        )

        # ── Step 2: find sharpest frame ───────────────────────────────────
        best_blur   = -1.0
        best_offset = 0
        best_frame  = frame

        for offset, scan_frame in frames_to_scan:
            score = blur_score(scan_frame)
            if score > best_blur:
                best_blur   = score
                best_offset = offset
                best_frame  = scan_frame
            if best_blur >= 800:
                logger.info(
                    f"[Enhancer] Sharp frame at +{best_offset}, "
                    f"stopping early"
                )
                break

        result.blur_score        = best_blur
        result.best_frame_offset = best_offset
        print(
            f"[BEST FRAME] offset=+{best_offset} blur={best_blur:.1f} "
            f"shape={best_frame.shape}"
        )

        # ── Step 3: fast-alpr on full best frame ──────────────────────────
        all_candidates: list[tuple[str, float]] = []

        full_results = _run_alpr(
            best_frame, save_dir, tag=f"full_{best_offset}"
        )
        all_candidates.extend(full_results)

        # ── Step 4: person-anchored crop fallback ─────────────────────────
        best_full_conf = max((c for _, c in full_results), default=0.0)

        if not full_results or best_full_conf < 0.50:
            logger.info(
                "[Enhancer] Full-frame conf low — trying person-anchored crops"
            )
            crops = _person_anchored_crops(best_frame, person_bbox)
            for ci, crop in enumerate(crops):
                if crop.shape[0] < 30 or crop.shape[1] < 80:
                    continue
                cv2.imwrite(
                    str(Path(save_dir) / f"debug_crop_{ci}.jpg"), crop
                )
                crop_results = _run_alpr(
                    crop, save_dir, tag=f"crop{ci}_{best_offset}"
                )
                all_candidates.extend(crop_results)

        # ── Step 5: pick best plate ───────────────────────────────────────
        # ── Step 5: pick best plate ───────────────────────────────────────
        plate_text, plate_conf = _best_result(all_candidates)
        result.plate_text = plate_text
        result.plate_conf = plate_conf

        # ── Step 5b: save corrected debug_alpr image ──────────────────────
        # Redraw the ALPR debug image using the CORRECTED plate text,
        # replacing the raw KA018554 label with KA05KK5546
        try:
            alpr        = _get_alpr()
            preds       = alpr.predict(best_frame)
            debug_frame = best_frame.copy()
            for r in preds:
                if r.detection is None:
                    continue
                bb = r.detection.bounding_box
                if hasattr(bb, "x1"):
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                else:
                    x1, y1 = int(bb[0]), int(bb[1])
                    x2, y2 = int(bb[0]+bb[2]), int(bb[1]+bb[3])
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                corrected_label = f"{plate_text or 'NO PLATE'} {plate_conf:.0%}"
                cv2.putText(
                    debug_frame, corrected_label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            debug_path = str(Path(save_dir) / f"debug_alpr_corrected_{best_offset}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            result.saved_paths.append(debug_path)
        except Exception as e:
            logger.debug(f"[Enhancer] corrected debug image error: {e}")

        # ── Step 6: save annotated evidence frame ─────────────────────────
        evidence_frame = best_frame.copy()
        px1, py1, px2, py2 = [int(v) for v in person_bbox]
        cv2.rectangle(evidence_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
        label = plate_text if plate_text else "NO PLATE"
        label_y = py1 - 8 if py1 > 20 else py1 + 25
        cv2.putText(
            evidence_frame, label, (px1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        ev_path = str(Path(save_dir) / f"{pair_id}_evidence.jpg")
        cv2.imwrite(ev_path, evidence_frame)
        result.saved_paths.append(ev_path)

        result.elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[Enhancer] {'✅' if plate_text else '⚠️ '} "
            f"pair={pair_id} | plate={plate_text} conf={plate_conf:.2f} | "
            f"blur={best_blur:.1f} best_frame=+{best_offset} "
            f"scanned={result.frames_scanned} | {result.elapsed_ms}ms"
        )
        return result