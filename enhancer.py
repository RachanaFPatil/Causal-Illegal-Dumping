"""
Layer 1.5 — Enhancement & Plate Detection
==========================================
Sits between Layer 4 (Dumping Inference) and Layer 5 (Agent).
Triggered ONLY when Layer 4 fires an "illegal_dumping" event.

FIXES in this version
=====================
FIX-A  Blurry plate multi-pass preprocessing
  When plate_region_blur < BLUR_VERY_LOW (85), standard CLAHE+unsharp
  is insufficient. Now runs 5 preprocessing variants (stronger CLAHE,
  adaptive threshold, bilateral+CLAHE, Otsu, morphological sharpening)
  through both fast-plate-ocr models AND EasyOCR.

FIX-B  EasyOCR on in-memory crops (not file glob)
  Runs directly on numpy arrays returned by _run_alpr, not saved files.

FIX-C  Person-proximity plate selection
  When person_bbox is known, ranks ALPR detections by distance from
  plate center to person center — the offender's vehicle plate wins.
  Fallback to largest Indian-format plate when person_bbox is None.

FIX-D  Watermark region exclusion
  Plates detected in the bottom-center horizontal band (not corners)
  are suppressed — real vehicle plates are in corners, watermarks are
  centered. Parameterized so tuning is easy.

FIX-E  Extended Indian plate regex
  Covers KA03NB4648, DL3CDB9940, DL3SDY425 (2-4 letter series group).

FIX-F  EasyOCR paragraph mode for blurry plates
  paragraph=True + width_ths/ycenter_ths groups fragmented characters
  that blur together on low-sharpness crops.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Indian plate patterns ─────────────────────────────────────────────────────
INDIAN_PLATE_RE      = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,4}\d{3,5}$')
INDIAN_PLATE_RELAXED = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z0-9]{4,7}$')

# ── Thresholds ────────────────────────────────────────────────────────────────
BLUR_VERY_LOW        = 85.0   # below this → aggressive multi-pass preprocessing
EASYOCR_TRIGGER_CONF = 0.75   # below this → run EasyOCR fallback

# ── EDSR model path ───────────────────────────────────────────────────────────
EDSR_MODEL_PATH = "EDSR_x4.pb"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_alpr           = None
_plate_ocr_vit  = None
_plate_ocr_cct  = None
_edsr_sr        = None
_easyocr_reader = None


def _get_alpr():
    global _alpr
    if _alpr is None:
        from fast_alpr import ALPR
        _alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
        )
        logger.info("[Enhancer] fast-alpr loaded")
    return _alpr


def _get_plate_ocr_vit():
    global _plate_ocr_vit
    if _plate_ocr_vit is None:
        from fast_plate_ocr import LicensePlateRecognizer
        _plate_ocr_vit = LicensePlateRecognizer("global-plates-mobile-vit-v2-model")
    return _plate_ocr_vit


def _get_plate_ocr_cct():
    global _plate_ocr_cct
    if _plate_ocr_cct is None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            _plate_ocr_cct = LicensePlateRecognizer("cct-s-v2-global-model")
        except Exception as e:
            logger.warning(f"[Enhancer] CCT-S unavailable: {e}")
            _plate_ocr_cct = False
    return _plate_ocr_cct if _plate_ocr_cct else None


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("[Enhancer] EasyOCR loaded")
        except Exception as e:
            logger.warning(f"[Enhancer] EasyOCR unavailable: {e}")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader else None


def _get_edsr() -> Optional[object]:
    global _edsr_sr
    if _edsr_sr is not None:
        return _edsr_sr if _edsr_sr is not False else None
    if not Path(EDSR_MODEL_PATH).exists():
        logger.warning(f"[Enhancer] EDSR not found at '{EDSR_MODEL_PATH}'. Using INTER_CUBIC.")
        _edsr_sr = False
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(EDSR_MODEL_PATH)
        sr.setModel("edsr", 4)
        _edsr_sr = sr
        return sr
    except Exception as e:
        logger.warning(f"[Enhancer] EDSR load failed ({e}), using INTER_CUBIC.")
        _edsr_sr = False
        return None


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class EnhancementResult:
    pair_id:           str
    plate_text:        Optional[str] = None
    plate_conf:        float         = 0.0
    blur_score:        float         = 0.0
    best_frame_offset: int           = 0
    frames_scanned:    int           = 0
    saved_paths:       list          = field(default_factory=list)
    elapsed_ms:        int           = 0


# ── Scoring helpers ───────────────────────────────────────────────────────────

def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _plate_region_blur(frame: np.ndarray) -> Tuple[float, float]:
    """Portrait-aware: bottom 45% height, right 45% width."""
    h, w    = frame.shape[:2]
    x_start = int(w * 0.45)
    y_start = int(h * 0.55)
    region  = frame[y_start:, x_start:]
    if region.size < 100:
        return blur_score(frame), float(h * w)
    return blur_score(region), float(region.shape[0] * region.shape[1])


# ── Frame generator ───────────────────────────────────────────────────────────

def cap_frame_generator(cap: cv2.VideoCapture, n_frames: int = 150) -> Iterator[np.ndarray]:
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    yielded   = 0
    while yielded < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        yielded += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 0) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad);  y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _is_indian(text: str) -> bool:
    n = _normalise(text)
    return bool(INDIAN_PLATE_RE.match(n)) or (
        bool(INDIAN_PLATE_RELAXED.match(n)) and len(n) >= 8
    )


def _indian_score(text: str) -> int:
    n = _normalise(text)
    if INDIAN_PLATE_RE.match(n):
        return 2
    if INDIAN_PLATE_RELAXED.match(n) and len(n) >= 8:
        return 1
    return 0


def _bbox_center(bbox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _pt_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)


# ── FIX-D: Watermark zone exclusion ──────────────────────────────────────────

def _is_watermark_zone(bbox: Tuple[int, int, int, int], frame_shape: Tuple) -> bool:
    """
    Returns True if this detection bbox is in the watermark band.
    Watermarks occupy the bottom-center of frames.
    Real vehicle plates are near the bottom corners.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # Bottom half AND center horizontal (not near left/right edges)
    in_bottom   = cy > 0.45 * h
    in_center_x = (0.20 * w) < cx < (0.80 * w)
    near_edge   = cx < 0.25 * w or cx > 0.75 * w
    if in_bottom and in_center_x and not near_edge:
        return True
    return False


# ── FIX-A: Multi-pass preprocessing ──────────────────────────────────────────

def _preprocess_variants(crop_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    5 preprocessing variants for blurry plates.
    Input can be BGR or grayscale.
    All outputs are grayscale.
    """
    if crop_img.ndim == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()

    variants: List[Tuple[str, np.ndarray]] = []

    # 1. Standard CLAHE + unsharp
    clahe1 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    g1 = clahe1.apply(gray)
    b1 = cv2.GaussianBlur(g1, (0, 0), 1.5)
    g1 = cv2.addWeighted(g1, 1.8, b1, -0.8, 0)
    variants.append(("std_clahe", g1))

    # 2. Strong CLAHE (FIX-A)
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    g2 = clahe2.apply(gray)
    b2 = cv2.GaussianBlur(g2, (0, 0), 2.0)
    g2 = cv2.addWeighted(g2, 2.2, b2, -1.2, 0)
    variants.append(("strong_clahe", g2))

    # 3. Bilateral deblur + adaptive threshold (FIX-A)
    g3 = cv2.bilateralFilter(gray, 9, 75, 75)
    g3 = cv2.adaptiveThreshold(
        g3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(("adaptive_thresh", g3))

    # 4. Histogram equalization + Otsu (FIX-A)
    g4 = cv2.equalizeHist(gray)
    _, g4 = cv2.threshold(g4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_eq", g4))

    # 5. Morphological top-hat sharpening (FIX-A)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g5 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    g5 = cv2.addWeighted(gray, 1.0, g5, 1.5, 0)
    clahe5 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    g5 = clahe5.apply(g5)
    variants.append(("morph_sharp", g5))

    return variants


def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """Standard single-pass preprocessing."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)


# ── Upscale ───────────────────────────────────────────────────────────────────

def _upscale_crop(crop: np.ndarray) -> np.ndarray:
    h, w   = crop.shape[:2]
    aspect = w / max(h, 1)
    sr     = _get_edsr()
    if sr:
        inp = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
        try:
            up = sr.upsample(inp)
            nh, nw = up.shape[:2]
            if aspect >= 2.5 and nw > 480:
                nw = 480; nh = max(int(h * (480 / w)), 64)
                up = cv2.resize(up, (nw, nh), interpolation=cv2.INTER_AREA)
            return up
        except Exception as e:
            logger.debug(f"[Enhancer] EDSR failed: {e}")
    scale = max(4, int(128 / max(h, 1)))
    nw = w * scale; nh = h * scale
    if aspect >= 2.5 and nw > 480:
        nw = 480; nh = max(int(h * (480 / w)), 64)
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _ocr_img_single(ocr_model, img: np.ndarray, alpr_conf: float, label: str) -> List[Tuple[str, float]]:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        pred = ocr_model.run(gray, return_confidence=True)
        if not pred:
            return []
        p    = pred[0]
        text = _normalise(p.plate if hasattr(p, "plate") else str(p))
        if text and len(text) >= 4:
            return [(text, alpr_conf)]
    except Exception as e:
        logger.debug(f"[OCR-{label}] {e}")
    return []


def _ocr_img(img: np.ndarray, alpr_conf: float = 0.5) -> List[Tuple[str, float]]:
    results = []
    vit = _get_plate_ocr_vit()
    results.extend(_ocr_img_single(vit, img, alpr_conf, "VIT"))
    cct = _get_plate_ocr_cct()
    if cct:
        results.extend(_ocr_img_single(cct, img, alpr_conf, "CCT"))
    return results


# ── FIX-F: EasyOCR with paragraph mode ───────────────────────────────────────

def _run_easyocr(crop: np.ndarray, alpr_conf: float, is_blurry: bool = False) -> List[Tuple[str, float]]:
    reader = _get_easyocr()
    if reader is None:
        return []
    try:
        crop_in = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
        kwargs: dict = dict(
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            detail=1,
            paragraph=is_blurry,
        )
        if is_blurry:
            kwargs['width_ths']   = 0.9
            kwargs['ycenter_ths'] = 0.8
        results = reader.readtext(crop_in, **kwargs)
        out = []
        for _, text, conf in results:
            if conf < 0.25:
                continue
            text = _normalise(text)
            if text and len(text) >= 4:
                print(f"[EasyOCR] '{text}' conf={conf:.2f}")
                out.append((text, max(float(conf), alpr_conf)))
        return out
    except Exception as e:
        logger.warning(f"[EasyOCR] failed: {e}")
        return []


# ── Tight-crop OCR ────────────────────────────────────────────────────────────

def _ocr_tight_crop(crop: np.ndarray, tag: str, alpr_conf: float, is_blurry: bool = False) -> List[Tuple[str, float]]:
    results        = []
    h_orig, w_orig = crop.shape[:2]
    aspect         = w_orig / max(h_orig, 1)
    upscale        = _upscale_crop(crop)
    new_h, new_w   = upscale.shape[:2]
    sr_mode        = "EDSR" if (_edsr_sr and _edsr_sr is not False) else "INTER_CUBIC"

    print(f"[ALPR-TIGHT] {tag} crop={h_orig}x{w_orig} aspect={aspect:.1f} "
          f"→ upscaled={new_h}x{new_w} [{sr_mode}]")

    # Preprocessing variants
    if is_blurry:
        variants = _preprocess_variants(upscale)
    else:
        variants = [("std_clahe", _preprocess_plate(upscale)),
                    ("raw_upscale", upscale if upscale.ndim == 2 else
                     cv2.cvtColor(upscale, cv2.COLOR_BGR2GRAY))]

    for label, img in variants:
        cands = _ocr_img(img, alpr_conf)
        for text, conf in cands:
            print(f"[ALPR-TIGHT] {tag} {label} → '{text}' conf={conf:.2f}")
        results.extend(cands)

    # Two-line plate split
    if aspect < 2.0:
        print(f"[ALPR-TIGHT] {tag} two-line plate — splitting top/bot")
        mid = upscale.shape[0] // 2
        merged = ""
        hit = 0
        for hlabel, half in [("top", upscale[:mid, :]), ("bot", upscale[mid:, :])]:
            if half.shape[0] < 8:
                continue
            half_pre  = _preprocess_plate(half)
            half_cands = _ocr_img(half_pre, alpr_conf) or _ocr_img(half, alpr_conf)
            if half_cands:
                best_text, _ = max(half_cands, key=lambda x: x[1])
                if len(best_text) >= 2:
                    print(f"[ALPR-TIGHT] {tag} {hlabel} → '{best_text}'")
                    merged += best_text
                    hit += 1
        if hit > 0 and len(merged) >= 6:
            print(f"[ALPR-TIGHT] {tag} merged → '{merged}'")
            results.append((merged, alpr_conf))
    else:
        print(f"[ALPR-TIGHT] {tag} single-line (aspect={aspect:.1f}) — skipping split")

    # EasyOCR on upscaled crop (FIX-F: paragraph mode for blurry)
    easy = _run_easyocr(upscale, alpr_conf, is_blurry=is_blurry)
    results.extend(easy)

    return results


# ── Main ALPR runner ──────────────────────────────────────────────────────────

def _run_alpr(
    frame: np.ndarray,
    save_dir: str,
    tag: str,
    person_bbox: Optional[tuple] = None,
    is_blurry: bool = False,
) -> Tuple[List[Tuple[str, float]], List[np.ndarray]]:
    """
    FIX-B: returns in-memory crops for direct EasyOCR use.
    FIX-C: sorts by distance to person when person_bbox is known.
    FIX-D: skips watermark-zone detections.
    """
    alpr      = _get_alpr()
    results   = []
    crops_out = []
    try:
        preds = alpr.predict(frame)
        if not preds:
            return [], []

        scored = []
        for r in preds:
            if r.ocr is None:
                continue
            raw_text = _normalise(r.ocr.text)
            raw_conf = float(np.mean(r.ocr.confidence))
            try:
                bb = r.detection.bounding_box
                if hasattr(bb, "x1"):
                    x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                else:
                    x1, y1 = int(bb[0]), int(bb[1])
                    x2, y2 = int(bb[0]+bb[2]), int(bb[1]+bb[3])
            except Exception:
                continue

            # FIX-D: suppress watermark-zone plates
            if _is_watermark_zone((x1, y1, x2, y2), frame.shape):
                print(f"[ALPR] SKIP watermark zone at ({x1},{y1}): '{raw_text}'")
                continue

            area      = max(0, x2-x1) * max(0, y2-y1)
            ind_score = _indian_score(raw_text)

            # FIX-C: proximity to person
            if person_bbox is not None:
                plate_c  = ((x1+x2)/2.0, (y1+y2)/2.0)
                person_c = _bbox_center(person_bbox)
                pdist    = _pt_dist(plate_c, person_c)
            else:
                pdist = float("inf")

            scored.append((ind_score, area, raw_conf, pdist, raw_text, r, x1, y1, x2, y2))

        if not scored:
            return [], []

        # FIX-C: when person known → sort by proximity first
        if person_bbox is not None:
            scored.sort(key=lambda x: (x[3], -x[0]))   # ascending dist, then Indian
        else:
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        for ind_score, area, raw_conf, pdist, raw_text, r, x1, y1, x2, y2 in scored:
            dist_label = f"{pdist:.0f}px" if pdist != float("inf") else "?"
            print(f"[ALPR] tag={tag} raw='{raw_text}' conf={raw_conf:.2f} "
                  f"person_dist={dist_label} ind={ind_score}")
            if len(raw_text) >= 4:
                results.append((raw_text, raw_conf))

            plate_crop = _safe_crop(frame, x1, y1, x2, y2, pad=4)
            if plate_crop is not None and plate_crop.size > 0:
                crops_out.append(plate_crop)
                idx = len(crops_out) - 1
                cv2.imwrite(str(Path(save_dir)/f"plate_crop_{tag}_{idx}.jpg"), plate_crop)
                up = _upscale_crop(plate_crop)
                cv2.imwrite(str(Path(save_dir)/f"plate_crop_upscaled_{tag}_{idx}.jpg"), up)
                cv2.imwrite(str(Path(save_dir)/f"plate_crop_pre_{tag}_{idx}.jpg"),
                            _preprocess_plate(up))
                results.extend(
                    _ocr_tight_crop(plate_crop, f"{tag}_{idx}", raw_conf,
                                    is_blurry=is_blurry)
                )

    except Exception as e:
        logger.warning(f"[Enhancer] ALPR error [{tag}]: {e}")

    return results, crops_out


# ── Best result picker ────────────────────────────────────────────────────────

def _best_result(candidates: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
    if not candidates:
        return None, 0.0
    seen, dedup = set(), []
    for text, conf in candidates:
        if text not in seen:
            seen.add(text)
            dedup.append((text, conf))
    strict  = [(t, c) for t, c in dedup if INDIAN_PLATE_RE.match(_normalise(t))]
    if strict:
        return max(strict, key=lambda x: x[1])
    relaxed = [(t, c) for t, c in dedup if _indian_score(t) > 0]
    if relaxed:
        return max(relaxed, key=lambda x: x[1])
    return max(dedup, key=lambda x: x[1])


# ── Person-anchored fallback crops ────────────────────────────────────────────

def _person_anchored_crops(frame: np.ndarray, person_bbox: tuple) -> List[np.ndarray]:
    fh, fw = frame.shape[:2]
    crops  = []
    for top_frac, bot_frac, x_frac in [
        (0.45, 0.75, 0.20),
        (0.50, 0.80, 0.25),
        (0.55, 0.85, 0.30),
    ]:
        c = _safe_crop(frame, int(fw*x_frac), int(fh*top_frac),
                       int(fw*(1-x_frac)), int(fh*bot_frac))
        if c is not None and c.size > 0:
            crops.append(c)
    return crops


# ── Core Enhancer ─────────────────────────────────────────────────────────────

class Enhancer:

    def process_event(
        self,
        frame:       np.ndarray,
        person_bbox: Optional[tuple],
        person_id:   str,
        pair_id:     str,
        save_dir:    str                             = "evidence",
        frame_iter:  Optional[Iterator[np.ndarray]] = None,
    ) -> EnhancementResult:

        t0 = time.time()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        result = EnhancementResult(pair_id=pair_id)

        # ── Step 1: build candidate frames ───────────────────────────────
        frames_to_scan: List[Tuple[int, np.ndarray]] = [(0, frame)]
        if frame_iter is not None:
            for offset, f in enumerate(frame_iter, start=1):
                frames_to_scan.append((offset, f))
        result.frames_scanned = len(frames_to_scan)

        # ── Step 2: best frame by composite plate sharpness ──────────────
        best_composite = -1.0
        best_blur_val  = -1.0
        best_offset    = 0
        best_frame     = frame

        for offset, scan_frame in frames_to_scan:
            region_blur, region_area = _plate_region_blur(scan_frame)
            composite = region_blur * np.log1p(region_area)
            if composite > best_composite:
                best_composite = composite
                best_blur_val  = region_blur
                best_offset    = offset
                best_frame     = scan_frame

        result.blur_score        = best_blur_val
        result.best_frame_offset = best_offset
        is_blurry = best_blur_val < BLUR_VERY_LOW

        print(
            f"[BEST FRAME] offset=+{best_offset} "
            f"plate_region_blur={best_blur_val:.1f} "
            f"composite={best_composite:.1f} "
            f"shape={best_frame.shape}"
            + (" [BLURRY — multi-pass preprocessing]" if is_blurry else "")
        )

        # ── Step 3: ALPR on best frame ────────────────────────────────────
        all_candidates: List[Tuple[str, float]] = []
        in_memory_crops: List[np.ndarray] = []

        full_results, full_crops = _run_alpr(
            best_frame, save_dir, tag=f"full_{best_offset}",
            person_bbox=person_bbox, is_blurry=is_blurry
        )
        all_candidates.extend(full_results)
        in_memory_crops.extend(full_crops)
        best_full_conf = max((c for _, c in full_results), default=0.0)

        # ── Step 3b: EasyOCR fallback (FIX-B: in-memory crops) ───────────
        if not full_results or best_full_conf < EASYOCR_TRIGGER_CONF:
            print(f"[Enhancer] EasyOCR fallback (fast-alpr conf={best_full_conf:.2f})")
            if in_memory_crops:
                for crop_img in in_memory_crops:
                    # Raw crop
                    all_candidates.extend(
                        _run_easyocr(crop_img, best_full_conf, is_blurry=is_blurry)
                    )
                    # Upscaled crop
                    up_img = _upscale_crop(crop_img)
                    all_candidates.extend(
                        _run_easyocr(up_img, best_full_conf, is_blurry=is_blurry)
                    )
                    # FIX-A: all preprocessing variants when blurry
                    if is_blurry:
                        for vlabel, vimg in _preprocess_variants(up_img):
                            easy_v = _run_easyocr(vimg, best_full_conf, is_blurry=True)
                            for t, c in easy_v:
                                print(f"[EasyOCR-{vlabel}] '{t}' conf={c:.2f}")
                            all_candidates.extend(easy_v)
            else:
                # No ALPR detections at all — try plate region directly
                h, w = best_frame.shape[:2]
                plate_region = best_frame[int(h*0.55):, int(w*0.40):]
                if plate_region.size > 0:
                    all_candidates.extend(
                        _run_easyocr(plate_region, best_full_conf, is_blurry=is_blurry)
                    )
                    if is_blurry:
                        for vlabel, vimg in _preprocess_variants(plate_region):
                            all_candidates.extend(
                                _run_easyocr(vimg, best_full_conf, is_blurry=True)
                            )

        # ── Step 4: person-anchored fallback ──────────────────────────────
        best_conf_so_far = max((c for _, c in all_candidates), default=0.0)
        if (not all_candidates or best_conf_so_far < 0.50) and person_bbox is not None:
            logger.info("[Enhancer] person-anchored crop fallback")
            for ci, crop in enumerate(_person_anchored_crops(best_frame, person_bbox)):
                if crop.shape[0] < 30 or crop.shape[1] < 80:
                    continue
                cv2.imwrite(str(Path(save_dir)/f"debug_crop_{ci}.jpg"), crop)
                c_results, c_crops = _run_alpr(
                    crop, save_dir, tag=f"crop{ci}_{best_offset}",
                    person_bbox=person_bbox, is_blurry=is_blurry
                )
                all_candidates.extend(c_results)
                for cc in c_crops:
                    all_candidates.extend(
                        _run_easyocr(cc, best_full_conf, is_blurry=is_blurry)
                    )

        # ── Step 5: pick best plate ───────────────────────────────────────
        plate_text, plate_conf = _best_result(all_candidates)
        result.plate_text = plate_text
        result.plate_conf = plate_conf

        # ── Step 5b: debug annotated frame ───────────────────────────────
        try:
            preds       = _get_alpr().predict(best_frame)
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
                label_y = y1 - 6 if y1 > 20 else y1 + 20
                cv2.putText(debug_frame,
                            f"{plate_text or 'NO PLATE'} {plate_conf:.0%}",
                            (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            debug_path = str(Path(save_dir)/f"debug_alpr_{best_offset}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            result.saved_paths.append(debug_path)
        except Exception as e:
            logger.debug(f"[Enhancer] debug frame error: {e}")

        # ── Step 6: evidence frame ────────────────────────────────────────
        ev = best_frame.copy()
        if person_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in person_bbox]
            cv2.rectangle(ev, (px1, py1), (px2, py2), (0, 0, 255), 2)
            label_y = py1 - 8 if py1 > 20 else py1 + 25
            cv2.putText(ev, plate_text or "NO PLATE", (px1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(ev, plate_text or "NO PLATE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ev_path = str(Path(save_dir)/f"{pair_id}_evidence.jpg")
        cv2.imwrite(ev_path, ev)
        result.saved_paths.append(ev_path)
        result.saved_paths.extend([
            str(p) for p in sorted(Path(save_dir).glob("plate_crop_*.jpg"))
        ])

        result.elapsed_ms = int((time.time() - t0) * 1000)
        if plate_text:
            print(f"[Enhancer] 🚗 PLATE: {plate_text} (conf={plate_conf:.2f})\n")
        else:
            print(f"[Enhancer] ⚠️  No plate read | "
                  f"blur={best_blur_val:.1f} | scanned={result.frames_scanned}f\n")
        logger.info(
            f"[Enhancer] pair={pair_id} plate={plate_text} conf={plate_conf:.2f} "
            f"blur={best_blur_val:.1f} blurry={is_blurry} "
            f"best_frame=+{best_offset} scanned={result.frames_scanned} "
            f"{result.elapsed_ms}ms"
        )
        return result