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
    # OR for NVIDIA GPU:
    pip install onnxruntime-gpu

Usage:
    python run_pipeline.py --source test7.mp4
    python run_pipeline.py --source test7.mp4 --save
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

def run(source: str, save: bool = False) -> None:

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

    writer = None
    if save:
        out_path = "vidtrace_output.mp4"
        writer   = cv2.VideoWriter(
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

        # ALPR — run once per confirmed violation
        for verdict in new_verdicts:
            if not verdict.get("violation"):
                continue
            event_key = verdict.get("pair_id", str(frame_idx))
            if event_key in alpr_done_events:
                continue
            alpr_done_events.add(event_key)

            if not ALPR_AVAILABLE or _enhancer is None:
                continue

            best_idx, best_frame_alpr = frame_buf.best_near(frame_idx, window=40)
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

            result = _enhancer.process_event(
                frame       = best_frame_alpr,
                person_bbox = person_bbox,
                person_id   = str(person_id),
                pair_id     = event_key,
                save_dir    = "evidence",
                frame_iter  = frame_iter,
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

            if result.plate_text:
                last_plate = (result.plate_text, result.plate_conf)

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

        cv2.imshow("VidTrace — Illegal Dumping Detection", vis)
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

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[Pipeline] Done. Frames: {frame_idx}")

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


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidTrace — Illegal Dumping Detection")
    parser.add_argument("--source", default="0")
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    run(args.source, args.save)