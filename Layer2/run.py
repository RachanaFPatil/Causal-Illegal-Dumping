"""
Layer 2 — Standalone Run Script / run.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs Layer 1 (RT-DETR detection) + Layer 2 (ByteTrack) only.
Layer 3 and Layer 4 are NOT loaded.

Fixes applied:
  1. Cumulative counts — tracked persons / objects use sets of unique track
     IDs, so the count only ever goes UP (never resets between frames).
  2. No overlapping stats — Layer 1's draw_detections() no longer draws its
     own stats block.  A single unified panel is drawn by draw_tracks() at
     y=24/46/68 (top-left).  No black-rectangle overwrite hack needed.

Usage:
    python3 -m Layer2.run --source test2.mp4
    python3 -m Layer2.run --source test2.mp4 --save
    python3 -m Layer2.run --source 0              # webcam

Controls:
    Q — quit
    P — pause / unpause
    S — save current frame to frame_dump/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import pathlib
import time

import cv2

# ── Layer 1 ───────────────────────────────────────────────────────────────────
from Layer1.detector       import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer1.visualizer     import draw_detections, draw_trash
from Layer1.config         import TRASH_ENABLE

# ── Layer 2 ───────────────────────────────────────────────────────────────────
from Layer2.tracker    import ByteTrackWrapper
from Layer2.visualizer import draw_tracks


# ── Main run loop ─────────────────────────────────────────────────────────────

def run(source: str, save: bool = False) -> None:
    # ── Initialise ────────────────────────────────────────────────────────────
    print("[Layer2] Initialising Layer 1 (detection)...")
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None

    print("[Layer2] Initialising Layer 2 (tracking)...")
    tracker = ByteTrackWrapper()

    # ── Open source ───────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Layer2] Source: {source}  ({W}×{H} @ {fps:.1f} fps)")

    # ── Video writer (optional) ───────────────────────────────────────────────
    writer = None
    if save:
        out_path = "layer2_output.mp4"
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        print(f"[Layer2] Saving output to '{out_path}'")

    print("[Layer2] Running — Q to quit | P to pause | S to save frame\n")

    frame_idx = 0
    paused    = False
    prev_time = time.time()

    # ── Cumulative counters — sets of unique confirmed track IDs ──────────────
    # Once a track ID enters this set it stays forever → count only goes up.
    seen_person_ids: set = set()   # unique confirmed person track IDs
    seen_object_ids: set = set()   # unique confirmed non-person track IDs

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Layer2] End of source.")
                    break

                frame_idx += 1

                # ── Layer 1: Detection ────────────────────────────────────────
                detections = detector.detect(frame)

                trash_detections = []
                if trash_detector:
                    trash_detections = trash_detector.detect(
                        frame.shape, detections
                    )

                # ── Layer 2: Tracking ─────────────────────────────────────────
                tracks = tracker.update(
                    detections, trash_detections, frame.shape[:2]
                )

                # ── Update cumulative unique-ID counters ──────────────────────
                # Only confirmed tracks appear in `tracks`; add their IDs to
                # the appropriate set.  Sets never shrink → counts only go up.
                for t in tracks:
                    if t.class_name == "person" and not t.is_trash:
                        seen_person_ids.add(t.track_id)
                    elif not t.is_trash:
                        seen_object_ids.add(t.track_id)

                cumulative_persons = len(seen_person_ids)
                cumulative_objects = len(seen_object_ids)

                # ── Visualise ─────────────────────────────────────────────────
                # Layer 1: draw bounding boxes only (no stats text)
                vis = draw_detections(frame, detections)
                vis = draw_trash(vis, trash_detections)

                # Layer 2: draw tracks + unified cumulative stats panel
                vis = draw_tracks(
                    vis,
                    tracks,
                    total_trash_events  = tracker.total_trash_events,
                    cumulative_persons  = cumulative_persons,
                    cumulative_objects  = cumulative_objects,
                )

                # ── Tracker HUD (bottom-left) — active/lost/total ─────────────
                hud_stats = [
                    f"Active : {tracker.active_count}",
                    f"Lost   : {tracker.lost_count}",
                    f"Total  : {len(tracks)}",
                ]
                for i, txt in enumerate(hud_stats):
                    cv2.putText(
                        vis, txt,
                        (10, H - 20 - (len(hud_stats) - 1 - i) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 255, 180), 1, cv2.LINE_AA,
                    )

                # ── FPS counter (top-right) ───────────────────────────────────
                now      = time.time()
                fps_live = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    vis, f"FPS:{fps_live:.1f}  F:{frame_idx}",
                    (W - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA,
                )

                cv2.imshow("Layer 1+2 — Detection + ByteTrack", vis)

                if writer:
                    writer.write(vis)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Layer2] Quit.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[Layer2] {'Paused' if paused else 'Resumed'}")
            elif key == ord("s"):
                pathlib.Path("frame_dump").mkdir(exist_ok=True)
                fname = f"frame_dump/frame_{frame_idx:06d}.jpg"
                cv2.imwrite(fname, vis)
                print(f"[Layer2] Saved frame → {fname}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[Layer2] Done — {frame_idx} frames processed.")
        print(f"[Layer2] Cumulative persons tracked : {len(seen_person_ids)}")
        print(f"[Layer2] Cumulative objects tracked : {len(seen_object_ids)}")
        print(f"[Layer2] Total trash events         : {tracker.total_trash_events}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 1+2 — Detection + ByteTrack (standalone test)"
    )
    parser.add_argument(
        "--source", required=True,
        help="Video file path, RTSP URL, or camera index (e.g. 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output to layer2_output.mp4"
    )
    args = parser.parse_args()
    run(args.source, args.save)