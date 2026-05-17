"""
VidTrace — Full Pipeline Runner (Layers 1–5 + ALPR via enhancer.py)
====================================================================

ROOT CAUSE FIX for Windows "No confirmed events" bug
------------------------------------------------------
The previous version called _patch_l1_device() AFTER `from Layer1.detector import ...`
Python caches modules on first import — so by the time the patch ran, RTDETRDetector
had already read DEVICE="mps" from config.py. On Windows, mps does not exist →
RT-DETR produced ZERO detections every frame → Layer5 saw nothing.

The fix: patch Layer1.config.DEVICE using importlib BEFORE any Layer1 import.
This is done in Steps 1+2 below. All Layer imports follow in Step 3.

Also fixed: broken indentation in agent.py ghost-debug print (leftover comment
turned into a real line). See agent.py fix notes.

Install:
    pip install fast-alpr fast-plate-ocr easyocr onnxruntime
    pip install reportlab pillow          # for Penalty/Challan PDF generation
    # OR for NVIDIA GPU:
    pip install onnxruntime-gpu

Usage:
    python run_pipeline.py --source test7.mp4
    python run_pipeline.py --source test7.mp4 --save
    python run_pipeline.py --source test7.mp4 --location "Outer Ring Road, Bengaluru"
    python run_pipeline.py --source test7.mp4 --no-challan   # skip challan generation
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import cv2
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — compute correct device for THIS platform
# ══════════════════════════════════════════════════════════════════════════════

def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"


DEVICE = _best_device()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — patch Layer1.config BEFORE importing anything from Layer1
#  This is the critical fix. importlib.import_module caches the module with
#  the correct DEVICE so detector.py reads it correctly at class-definition time.
# ══════════════════════════════════════════════════════════════════════════════

def _force_patch_l1_device(device: str) -> None:
    try:
        if "Layer1.config" in sys.modules:
            sys.modules["Layer1.config"].DEVICE = device
        else:
            mod = importlib.import_module("Layer1.config")
            mod.DEVICE = device
        print(f"[Pipeline] Device → {device}")
    except Exception as exc:
        print(f"[Pipeline] WARNING: could not patch Layer1.config: {exc}")


_force_patch_l1_device(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — safe to import Layer1 now (DEVICE is already correct)
# ══════════════════════════════════════════════════════════════════════════════

from Layer1.detector        import RTDETRDetector       # noqa: E402
from Layer1.trash_detector  import TrashDetector        # noqa: E402
from Layer1.bin_detector    import BinDetector          # noqa: E402

from Layer2.tracker         import ByteTrackWrapper     # noqa: E402
from Layer2.bin_tracker     import BinTracker           # noqa: E402
from Layer2.visualizer      import draw_tracks          # noqa: E402
from Layer2.bin_visualizer  import draw_bins            # noqa: E402

from Layer3.feature_extractor import BinInteractionFeatureExtractor  # noqa: E402

from Layer4.dumping_inference import DumpingInference   # noqa: E402

from Layer5.agent           import DumpingAgent         # noqa: E402
from Layer5.visualizer      import (                    # noqa: E402
    draw_l5_verdicts,
    draw_l5_reasoning,
    draw_l5_evidence_bars,
    draw_l5_summary_box,
)

# ── Enhancer (ALPR) ───────────────────────────────────────────────────────────
try:
    from enhancer import Enhancer, cap_frame_generator
    _enhancer      = Enhancer()
    ALPR_AVAILABLE = True
    print("[ALPR] enhancer.py loaded — fast-alpr + EasyOCR pipeline active")
except ImportError as _e:
    ALPR_AVAILABLE = False
    _enhancer      = None
    print(f"[ALPR] enhancer.py not available: {_e}")
    print("       Run: pip install fast-alpr fast-plate-ocr easyocr onnxruntime")
except Exception as _e:
    ALPR_AVAILABLE = False
    _enhancer      = None
    print(f"[ALPR] enhancer.py error on load: {_e}")

# ── Penalty & Challan Manager ─────────────────────────────────────────────────
try:
    from penalty_manager import PenaltyManager
    _penalty_manager   = PenaltyManager()
    CHALLAN_AVAILABLE  = True
    print("[Challan] penalty_manager.py loaded — challan issuance active")
except ImportError as _e:
    CHALLAN_AVAILABLE  = False
    _penalty_manager   = None
    print(f"[Challan] penalty_manager.py not available: {_e}")
    print("          Run: pip install reportlab pillow")
except Exception as _e:
    CHALLAN_AVAILABLE  = False
    _penalty_manager   = None
    print(f"[Challan] penalty_manager.py error on load: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plate overlay
# ══════════════════════════════════════════════════════════════════════════════

def _draw_plate(frame: np.ndarray, plate_text: str, conf: float) -> np.ndarray:
    if not plate_text:
        return frame
    H, W    = frame.shape[:2]
    label   = f"PLATE: {plate_text}  ({conf:.2f})"
    font    = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.7, 2
    (tw, th), _  = cv2.getTextSize(label, font, scale, thick)
    x, y = W - tw - 18, H - 18
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + 6), (15, 15, 15), -1)
    cv2.putText(frame, label, (x, y), font, scale, (0, 220, 255), thick, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling frame buffer (for best-frame selection at ALPR time)
# ══════════════════════════════════════════════════════════════════════════════

def _sharpness(f: np.ndarray) -> float:
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _FrameBuffer:
    def __init__(self, maxlen: int = 120):
        self._buf:    list[tuple[int, np.ndarray]] = []
        self._maxlen: int                          = maxlen

    def push(self, idx: int, frame: np.ndarray) -> None:
        self._buf.append((idx, frame.copy()))
        if len(self._buf) > self._maxlen:
            self._buf.pop(0)

    def best_near(self, target: int, window: int = 40) -> tuple[int, np.ndarray | None]:
        cands = [(i, f) for i, f in self._buf if abs(i - target) <= window]
        if not cands:
            return 0, None
        best_i, best_f = max(cands, key=lambda x: _sharpness(x[1]))
        print(
            f"[BEST FRAME] offset={best_i - target:+d} "
            f"blur={_sharpness(best_f):.1f} shape={best_f.shape}"
        )
        return best_i, best_f


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(source: str, save: bool = False, location: str = "",
        enable_challan: bool = True) -> None:

    print("[Pipeline] Booting VidTrace...")

    # ── Init all layers ───────────────────────────────────────────────────────
    detector          = RTDETRDetector()
    trash_detector    = TrashDetector()
    bin_detector      = BinDetector(device="cpu")

    tracker           = ByteTrackWrapper()
    bin_tracker       = BinTracker()

    feature_extractor = BinInteractionFeatureExtractor(debug=False)
    dumping_inference = DumpingInference()
    agent             = DumpingAgent()

    # ── Video ─────────────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"[Pipeline] Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Derive a sensible location from the source filename if none provided
    if not location:
        import os
        location = os.path.splitext(os.path.basename(str(source)))[0]

    # Output video name derived from source
    out_path = f"vidtrace_output.mp4"

    writer = None
    if save:
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        print(f"[Pipeline] Saving → {out_path}")

    print("[Pipeline] Running — Q:quit P:pause D:L3-debug R:L5-reasoning")

    # ── State ─────────────────────────────────────────────────────────────────
    frame_idx         = 0
    paused            = False
    show_l3_debug     = False
    show_l5_reason    = False
    frame_buf         = _FrameBuffer(maxlen=120)
    all_verdicts:     list[dict]      = []
    last_plate:       tuple[str, float] = ("", 0.0)
    alpr_done_events: set[str]        = set()

    # Challan tracking — one challan per confirmed violation event
    # Maps event pair_id → challan_id so we never double-issue
    challan_issued:   dict[str, str]  = {}

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frame_buf.push(frame_idx, frame)

        # Layer 1
        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)
        bin_detections   = bin_detector.detect(frame)

        # Layer 2
        tracked_objects = tracker.update(detections, trash_detections, frame, (H, W))
        tracked_bins    = bin_tracker.update(bin_detections)

        # Layer 3
        sequences = feature_extractor.update(
            tracked_objects, tracked_bins, timestamp=frame_idx / fps
        )

        # Layer 4
        l4_events = dumping_inference.update(tracked_objects, tracked_bins)

        # Layer 5
        new_verdicts = agent.update(frame_idx, tracked_objects, tracked_bins, l4_events)
        all_verdicts.extend(new_verdicts)

        # ── ALPR + Challan — run once per confirmed violation ─────────────────
        for verdict in new_verdicts:
            if not verdict.get("violation"):
                continue

            event_key = verdict.get("pair_id", str(frame_idx))

            # ── ALPR (plate reading) ──────────────────────────────────────────
            if event_key not in alpr_done_events:
                alpr_done_events.add(event_key)

                alpr_result      = None
                best_frame_alpr  = frame

                if ALPR_AVAILABLE and _enhancer is not None:
                    best_idx, best_frame_alpr = frame_buf.best_near(
                        frame_idx, window=40
                    )
                    if best_frame_alpr is None:
                        best_frame_alpr = frame

                    person_id   = verdict.get("person_id")
                    person_bbox = None
                    if person_id is not None:
                        for obj in tracked_objects:
                            if obj.track_id == person_id:
                                person_bbox = tuple(obj.bbox.tolist())
                                break

                    saved_pos  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_iter = cap_frame_generator(cap, n_frames=150)

                    alpr_result = _enhancer.process_event(
                        frame       = best_frame_alpr,
                        person_bbox = person_bbox,
                        person_id   = str(person_id),
                        pair_id     = event_key,
                        save_dir    = "evidence",
                        frame_iter  = frame_iter,
                    )

                    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

                    if alpr_result.plate_text:
                        last_plate = (alpr_result.plate_text, alpr_result.plate_conf)

            # ── Challan issuance (only after ALPR is done for this event) ─────
            # We issue the challan on the first pass where we have ALPR data.
            # If ALPR is unavailable, we still issue (plate_number=None →
            # pedestrian-style challan with "Owner Not Found").
            if (enable_challan
                    and CHALLAN_AVAILABLE
                    and _penalty_manager is not None
                    and event_key not in challan_issued):

                # Use plate from this event's ALPR result if available,
                # otherwise fall back to last_plate read earlier in the video.
                if alpr_result is not None and alpr_result.plate_text:
                    plate_for_challan = alpr_result.plate_text
                    plate_conf        = alpr_result.plate_conf
                    evidence_plate    = (alpr_result.saved_paths[0]
                                         if alpr_result.saved_paths else None)
                elif last_plate[0]:
                    plate_for_challan = last_plate[0]
                    plate_conf        = last_plate[1]
                    evidence_plate    = None
                else:
                    plate_for_challan = None
                    plate_conf        = 0.0
                    evidence_plate    = None

                confidence = verdict.get("confidence", 0.0)

                try:
                    challan_id = _penalty_manager.create_violation(
                        plate_number              = plate_for_challan,
                        evidence_video_path       = out_path if save else None,
                        evidence_plate_image_path = evidence_plate,
                        location                  = location,
                        confidence                = confidence,
                    )
                    pdf_path = _penalty_manager.generate_challan(challan_id)

                    challan_issued[event_key] = challan_id

                    print(
                        f"[Challan] ✅ Issued | {challan_id} | "
                        f"plate={plate_for_challan or 'N/A'} | "
                        f"conf={confidence:.2f}"
                    )
                    if pdf_path:
                        print(f"[Challan] 📄 PDF → {pdf_path}")

                except Exception as _ce:
                    print(f"[Challan] ⚠️  Failed to issue challan for "
                          f"{event_key}: {_ce}")

        # Visualise
        vis = frame.copy()
        vis = draw_tracks(vis, tracked_objects, tracker.total_trash_events)
        vis = draw_bins(vis, tracked_bins)

        if show_l3_debug:
            vis = feature_extractor.draw_debug(vis, tracked_objects, tracked_bins)

        locked_results = agent.get_all_results()
        vis = draw_l5_verdicts(vis, locked_results)
        vis = draw_l5_evidence_bars(vis, locked_results)
        vis = draw_l5_summary_box(vis, locked_results)

        if show_l5_reason:
            vis = draw_l5_reasoning(vis, agent)

        if last_plate[0]:
            vis = _draw_plate(vis, last_plate[0], last_plate[1])

        cv2.putText(
            vis, f"F:{frame_idx}",
            (W - 90, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
        )

        cv2.imshow("VidTrace \u2014 Illegal Dumping Detection", vis)
        if writer:
            writer.write(vis)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"[Pipeline] {'Paused' if paused else 'Resumed'}")
        elif key == ord("d"):
            show_l3_debug = not show_l3_debug
            print(f"[Pipeline] L3 debug: {'ON' if show_l3_debug else 'OFF'}")
        elif key == ord("r"):
            show_l5_reason = not show_l5_reason
            print(f"[Pipeline] L5 reasoning: {'ON' if show_l5_reason else 'OFF'}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[Pipeline] Done. Frames: {frame_idx}")

    # ── Final Layer 5 summary ─────────────────────────────────────────────────
    final_results = agent.get_all_results()
    if final_results:
        print(f"\n[Layer5] ══ Final Confirmed Events ({len(final_results)}) ══")
        for r in final_results:
            tag = "🚨 VIOLATION" if r["violation"] else "✅ LEGAL    "
            print(
                f"  {tag}  {r['event']:<22} conf={r['confidence']:.2f}  "
                f"P{r['person_id']} T{r['object_id']}  "
                f"coupling={r.get('coupling_frames','?')}f  "
                f"cos={r.get('peak_coupling',0):.2f}  "
                f"intent={r.get('intent_score',0):.2f}  "
                f"frames={r['frames']}"
            )
            print(f"       reason: {r.get('reason','')}")
            log_str = " -> ".join(r.get("reasoning_log", []))
            if log_str:
                print(f"       L5 log: {log_str}")
    else:
        print("\n[Layer5] No confirmed events.")

    if last_plate[0]:
        print(f"\n[Enhancer] 🚗 Final plate: {last_plate[0]} (conf={last_plate[1]:.2f})")

    # ── Final Challan summary ─────────────────────────────────────────────────
    if challan_issued:
        print(f"\n[Challan] ══ Challans Issued This Run ({len(challan_issued)}) ══")
        for ev_key, cid in challan_issued.items():
            print(f"  Event {ev_key} → {cid}")

        # Run escalation check in case any existing violations are overdue
        if CHALLAN_AVAILABLE and _penalty_manager is not None:
            try:
                escalated = _penalty_manager.check_and_escalate()
                if escalated:
                    print(f"[Challan] ⚡ {escalated} existing violation(s) escalated")
            except Exception as _esc_e:
                print(f"[Challan] Escalation check failed: {_esc_e}")

        # Print summary from DB
        if CHALLAN_AVAILABLE and _penalty_manager is not None:
            try:
                s = _penalty_manager.summary()
                print(
                    f"[Challan] DB Summary — "
                    f"total={s['total']} | "
                    f"pending={s['pending']} | "
                    f"paid={s['paid']} | "
                    f"escalated={s['escalated']} | "
                    f"collected=Rs.{s['collected'] or 0:.2f}"
                )
            except Exception:
                pass
    else:
        print("\n[Challan] No challans issued this run.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidTrace — Illegal Dumping Detection")
    parser.add_argument("--source",      default="0",
                        help="Video file path or camera index")
    parser.add_argument("--save",        action="store_true",
                        help="Save annotated output video as vidtrace_output.mp4")
    parser.add_argument("--location",    default="",
                        help="Location string for challan (e.g. 'Outer Ring Road, Bengaluru')")
    parser.add_argument("--no-challan",  action="store_true",
                        help="Disable automatic challan issuance")
    args = parser.parse_args()

    run(
        source         = args.source,
        save           = args.save,
        location       = args.location,
        enable_challan = not args.no_challan,
    )