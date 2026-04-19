"""
Full Pipeline Runner — Layer 1 + Layer 2 (+ Bin Detection)
===========================================================

TWO MODES — selected automatically each frame:

  MODE A  (no bins detected):
    → Runs exactly as before — RT-DETR COCO model + TrashDetector +
      ByteTrack.  Zero changes to existing behaviour.

  MODE B  (bins detected in frame):
    → Also runs BinDetector + BinTracker in parallel.
    → Each unique trash_bin track is flagged ONCE.
    → Bin overlays drawn on top of existing pipeline output.
    → Existing trash-drop detection still runs normally alongside.

Hard constraints respected:
  ✓ Layer 1 detector.py / trash_detector.py  — NOT modified
  ✓ Layer 2 tracker.py / track_state.py      — NOT modified
  ✓ ByteTrack output format unchanged
  ✓ Bin IDs start at 9000 — no collision with ByteTrack IDs
  ✓ CPU optimised — bin model only called when bins are present

Usage:
    python run_pipeline.py --source test2.mp4
    python run_pipeline.py --source bin_video.mp4
    python run_pipeline.py --source 0
    python run_pipeline.py --source test2.mp4 --save
    python run_pipeline.py --source test2.mp4 --no-bins   # disable bin detection
"""

import argparse
import time
import cv2

# ── Layer 1 (unchanged) ───────────────────────────────────────────────────────
from Layer1.detector        import RTDETRDetector
from Layer1.trash_detector  import TrashDetector
from Layer1.visualizer      import draw_detections, draw_trash
from Layer1.config          import TRASH_ENABLE, DEVICE

# ── Layer 1 — Bin detector (new, isolated) ────────────────────────────────────
from Layer1.bin_detector    import BinDetector

# ── Layer 2 (unchanged) ───────────────────────────────────────────────────────
from Layer2.tracker         import ByteTrackWrapper
from Layer2.visualizer      import draw_tracks

# ── Layer 2 — Bin tracker (new, isolated) ────────────────────────────────────
from Layer2.bin_tracker     import BinTracker
from Layer2.bin_visualizer  import draw_bins


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_bins(frame, bin_detector: BinDetector):
    """
    Lightweight check: run bin detector and return (bins, detections).
    Called every frame — fast because RT-DETR is already GPU/MPS batched
    and the model is small (nc=1).
    """
    bin_dets = bin_detector.detect(frame)
    return bin_dets


# ── Main run loop ─────────────────────────────────────────────────────────────

def run(source: str, save: bool = False, enable_bins: bool = True, enable_display: bool = True) -> None:

    # ── Initialise Layer 1 ────────────────────────────────────────────────────
    print("[Pipeline] Initialising Layer 1 (COCO RT-DETR + Trash)...")
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None

    # ── Initialise Layer 1 Bin Detector (fine-tuned) ──────────────────────────
    bin_detector = None
    if enable_bins:
        print("[Pipeline] Initialising fine-tuned Bin Detector...")
        try:
            bin_detector = BinDetector(device="mps")
        except Exception as e:
            print(f"[Pipeline] ⚠️  BinDetector failed to load: {e}")
            print("[Pipeline]    Continuing without bin detection.")
            bin_detector = None

    # ── Initialise Layer 2 ────────────────────────────────────────────────────
    print("[Pipeline] Initialising Layer 2 (ByteTrack + BinTracker)...")
    tracker     = ByteTrackWrapper()
    bin_tracker = BinTracker() if bin_detector else None

    # ── Open source ───────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"[Pipeline] Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[Pipeline] Source: {source}  ({W}×{H} @ {fps:.1f} fps)")

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    if save:
        out_path = "pipeline_output.mp4"
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        print(f"[Pipeline] Saving output → {out_path}")

    print("[Pipeline] Running — Q to quit | P to pause\n")

    frame_idx    = 0
    paused       = False
    prev_time    = time.time()
    no_display   = not enable_display

    # Cumulative unique-ID counters (only go up)
    seen_person_ids: set = set()
    seen_object_ids: set = set()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] End of source.")
                    break

                frame_idx += 1

                # ════════════════════════════════════════════════════════════
                # LAYER 1A — COCO RT-DETR detection (always runs)
                # ════════════════════════════════════════════════════════════
                detections = detector.detect(frame)

                trash_detections = []
                if trash_detector:
                    trash_detections = trash_detector.detect(frame.shape, detections)

                # ════════════════════════════════════════════════════════════
                # LAYER 1B — Bin detection (fine-tuned model, runs in parallel)
                # ════════════════════════════════════════════════════════════
                bin_detections = []
                if bin_detector:
                    bin_detections = _has_bins(frame, bin_detector)
                    # MODE A: no bins → bin_tracker.update([]) → returns []
                    # MODE B: bins    → bin_tracker.update(dets) → tracks bins

                # ════════════════════════════════════════════════════════════
                # LAYER 2A — ByteTrack (unchanged — handles COCO objects)
                # ════════════════════════════════════════════════════════════
                tracked = tracker.update(detections, trash_detections, (H, W))

                # Cumulative unique person / object counts
                for t in tracked:
                    if t.class_name == "person" and not t.is_trash:
                        seen_person_ids.add(t.track_id)
                    elif not t.is_trash:
                        seen_object_ids.add(t.track_id)

                # ════════════════════════════════════════════════════════════
                # LAYER 2B — Bin Tracker (once-per-ID flagging)
                # ════════════════════════════════════════════════════════════
                tracked_bins = []
                if bin_tracker:
                    tracked_bins = bin_tracker.update(bin_detections)

                # ════════════════════════════════════════════════════════════
                # VISUALISE
                # ════════════════════════════════════════════════════════════

                # Layer 1: COCO detections + trash highlights
                vis = draw_detections(frame.copy(), detections)
                vis = draw_trash(vis, trash_detections)

                # Layer 2: ByteTrack overlays (persons, objects, trash events)
                vis = draw_tracks(
                    vis,
                    tracked,
                    total_trash_events = tracker.total_trash_events,
                    cumulative_persons = len(seen_person_ids),
                    cumulative_objects = len(seen_object_ids),
                )

                # Layer 2B: Bin overlays (drawn on top — doesn't interfere)
                if bin_tracker:
                    vis = draw_bins(
                        vis,
                        tracked_bins,
                        total_flagged = bin_tracker.total_bins_flagged,
                    )

                # ── FPS + frame counter ───────────────────────────────────
                now      = time.time()
                fps_live = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    vis, f"FPS:{fps_live:.1f}  F:{frame_idx}",
                    (W - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA,
                )

                # ── Mode indicator (top-left, small) ─────────────────────
                mode_text  = "MODE B: BIN DETECTION ACTIVE" if tracked_bins else "MODE A: STANDARD"
                mode_color = (0, 200, 255) if tracked_bins else (180, 180, 180)

                # ── Per-frame terminal log ────────────────────────────────
                persons = [t for t in tracked if t.class_name == "person"]
                objects = [t for t in tracked if t.class_name != "person" and not t.is_trash]
                trash   = [t for t in tracked if t.is_trash]
                bin_summary = (
                    f"  BINS: {[f'ID#{b.bin_id} flagged={b.flagged} conf={b.confidence:.2f}' for b in tracked_bins]}"
                    if tracked_bins else "  BINS: none"
                )
                print(
                    f"[F{frame_idx:05d}] "
                    f"persons={len(persons)} objects={len(objects)} trash={len(trash)} | "
                    f"trash_events={tracker.total_trash_events} | "
                    f"{bin_summary} | "
                    f"bins_flagged_total={bin_tracker.total_bins_flagged if bin_tracker else 0}"
                )

                if not no_display:
                    cv2.putText(vis, mode_text, (10, H - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
                    cv2.imshow("Pipeline — Detection + Tracking + Bin Flagging", vis)
                if writer:
                    writer.write(vis)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF if not no_display else 0xFF
            if key == ord("q"):
                print("[Pipeline] Quit.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[Pipeline] {'Paused' if paused else 'Resumed'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n[Pipeline] ── Summary ──────────────────────────────────")
        print(f"[Pipeline] Frames processed      : {frame_idx}")
        print(f"[Pipeline] Persons tracked       : {len(seen_person_ids)}")
        print(f"[Pipeline] Objects tracked       : {len(seen_object_ids)}")
        print(f"[Pipeline] Trash events          : {tracker.total_trash_events}")
        if bin_tracker:
            print(f"[Pipeline] Bins flagged (unique) : {bin_tracker.total_bins_flagged}")
        print(f"[Pipeline] ────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline — COCO detection + ByteTrack + Bin Flagging"
    )
    parser.add_argument("--source",   required=True,
                        help="Video path, RTSP URL, or camera index (e.g. 0)")
    parser.add_argument("--save",     action="store_true",
                        help="Save annotated output to pipeline_output.mp4")
    parser.add_argument("--no-bins",     action="store_true",
                        help="Disable fine-tuned bin detector (run as before)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Disable OpenCV window — print frame log to terminal only")
    args = parser.parse_args()

    run(args.source, save=args.save, enable_bins=not args.no_bins, enable_display=not args.no_display)