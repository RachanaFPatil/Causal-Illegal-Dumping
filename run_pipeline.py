"""
VidTrace: Full Pipeline Runner — Layer 1 + 2 + 3 + 4
Integrated with DumpingInference (Layer 4 Fix)
"""

import argparse
import time
import cv2
import numpy as np

# ── Layer 1 Imports ──────────────────────────────────────────────────────────
from Layer1.detector        import RTDETRDetector
from Layer1.trash_detector  import TrashDetector
from Layer1.bin_detector    import BinDetector
from Layer1.visualizer      import draw_detections, draw_trash

# ── Layer 2 & 3 Imports ──────────────────────────────────────────────────────
from Layer2.tracker         import ByteTrackWrapper
from Layer2.bin_tracker     import BinTracker
from Layer2.visualizer      import draw_tracks
from Layer2.bin_visualizer  import draw_bins
from Layer3.feature_extractor import BinInteractionFeatureExtractor

# ── Layer 4 Fix ─────────────────────────────────────────────────────────────
from Layer4.dumping_inference import DumpingInference, DumpingEvent

# ── Global UI Config ─────────────────────────────────────────────────────────
_COL_ILLEGAL = (0, 50, 255)   # Red
_COL_LEGAL   = (50, 220, 50)  # Green
_COL_BG      = (20, 20, 20)   # Dark Gray

def _draw_event_banners(frame: np.ndarray, events: list, W: int) -> np.ndarray:
    """Draws the final Layer 4 decisions as banners in the top-right."""
    visible = [e for e in events if e.event != "pending"]
    if not visible:
        return frame

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    pad   = 6

    for i, ev in enumerate(visible):
        color = _COL_ILLEGAL if ev.event == "illegal_dumping" else _COL_LEGAL
        label = f"ID:{ev.pair_id} | {ev.event.upper()} | {ev.confidence:.2f}"
        
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        x = W - tw - 15
        y = 40 + i * 30
        
        # Background box for readability
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), _COL_BG, -1)
        cv2.putText(frame, label, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return frame

def run(source: str, save: bool = False, debug: bool = False) -> None:
    # ── 1. Initialize System ──────────────────────────────────────────────────
    print("[Pipeline] Booting VidTrace Architecture...")
    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    bin_detector   = BinDetector()
    
    tracker        = ByteTrackWrapper()
    bin_tracker    = BinTracker()
    
    # Layer 3 & 4
    extractor      = BinInteractionFeatureExtractor(debug=debug)
    inference      = DumpingInference()

    # ── 2. Setup Video ────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if save:
        out_name = "vidtrace_output.mp4"
        writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        print(f"[Pipeline] Recording to: {out_name}")

    print("\n[Pipeline] Active. Press 'Q' to quit | 'D' for Layer 3 debug overlays.")
    
    prev_time   = time.time()
    frame_idx   = 0
    paused      = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                ts = time.time()

                # ── 3. Processing Pipeline ────────────────────────────────────
                # Layer 1: Detect
                dets       = detector.detect(frame)
                trash_dets = trash_detector.detect(frame.shape, dets)
                bin_dets   = bin_detector.detect(frame)

                # Layer 2: Track
                tracked_objs = tracker.update(dets, trash_dets, (H, W))
                tracked_bins = bin_tracker.update(bin_dets)

                # Layer 3: Feature Extraction (Hidden State)
                extractor.update(tracked_objs, tracked_bins, ts)

                # Layer 4: Final Inference (Fixes early triggers & flip-flopping)
                events = inference.update(tracked_objs, tracked_bins)

                # ── 4. Visualization ──────────────────────────────────────────
                vis = frame.copy()
                
                # Layer 1 & 2 Overlays
                vis = draw_tracks(vis, tracked_objs, tracker.total_trash_events)
                vis = draw_bins(vis, tracked_bins, bin_tracker.total_bins_flagged)
                
                # Layer 4 Decision Banners
                vis = _draw_event_banners(vis, events, W)

                # Layer 3 Debug (Distance zones/trajectories)
                if debug:
                    vis = extractor.draw_debug(vis, tracked_objs, tracked_bins)

                # ── 5. HUD & UI ───────────────────────────────────────────────
                # Black Background for Stats
                cv2.rectangle(vis, (0, 0), (220, 85), (0, 0, 0), -1)
                
                persons  = sum(1 for t in tracked_objs if t.class_name == "person")
                objects  = sum(1 for t in tracked_objs if t.is_trash)
                labels = [
                    f"Persons Tracked: {persons}",
                    f"Trash Objects : {objects}",
                    f"Flagged Bins  : {bin_tracker.total_bins_flagged}"
                ]
                for i, txt in enumerate(labels):
                    cv2.putText(vis, txt, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Performance Info
                now = time.time()
                fps_live = 1.0 / (now - prev_time + 1e-9)
                prev_time = now
                cv2.putText(vis, f"FPS: {fps_live:.1f} | Frame: {frame_idx}", (W - 200, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("VidTrace - Illegal Dumping Detection", vis)
                if writer: writer.write(vis)

            # ── 6. Controls ───────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print("[Pipeline] Paused" if paused else "[Pipeline] Resumed")
            elif key == ord("d"):
                debug = not debug
                extractor._debug = debug
                print(f"[Debug] L3 Overlay: {'ON' if debug else 'OFF'}")

    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"\n[Pipeline] Shutdown. Total Frames: {frame_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidTrace Illegal Dumping Detection")
    parser.add_argument("--source", default="0", help="Video path or camera ID")
    parser.add_argument("--save",   action="store_true", help="Save output to file")
    parser.add_argument("--debug",  action="store_true", help="Show L3 debug trajectories")
    args = parser.parse_args()
    run(args.source, args.save, args.debug)