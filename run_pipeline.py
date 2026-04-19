"""
Full Pipeline Runner — Layer 1 + Layer 2 + Layer 3

Layer 1 : RT-DETR detection + trash detection
Layer 2 : ByteTrack identity tracking + bin tracking
Layer 3 : Bin–Trash interaction feature extraction (NEW)

Usage:
    python run_pipeline.py --source test2.mp4
    python run_pipeline.py --source 0
    python run_pipeline.py --source test2.mp4 --save
    python run_pipeline.py --source test2.mp4 --debug   ← shows Layer 3 overlays
    python run_pipeline.py --source test2.mp4 --save --debug
"""

import argparse
import time
import cv2

# ── Layer 1 ───────────────────────────────────────────────────────────────────
from Layer1.detector       import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer1.bin_detector   import BinDetector
from Layer1.visualizer     import draw_detections, draw_trash
from Layer1.config         import TRASH_ENABLE

# ── Layer 2 ───────────────────────────────────────────────────────────────────
from Layer2.tracker        import ByteTrackWrapper
from Layer2.bin_tracker    import BinTracker
from Layer2.visualizer     import draw_tracks
from Layer2.bin_visualizer import draw_bins

# ── Layer 3 (NEW) ─────────────────────────────────────────────────────────────
from Layer3.feature_extractor import BinInteractionFeatureExtractor


def run(source: str, save: bool = False, debug: bool = False) -> None:

    # ── Initialise all layers ─────────────────────────────────────────────────
    print("[Pipeline] Initialising Layer 1 (detection)...")
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None
    bin_detector   = BinDetector(device="mps")

    print("[Pipeline] Initialising Layer 2 (tracking)...")
    tracker     = ByteTrackWrapper()
    bin_tracker = BinTracker()

    print("[Pipeline] Initialising Layer 3 (feature extraction)...")
    extractor = BinInteractionFeatureExtractor(debug=debug)

    # ── Open video source ─────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"[Pipeline] Cannot open source: {source}")

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

    print("[Pipeline] Running — Q to quit | P to pause | D to toggle debug\n")

    frame_idx = 0
    paused    = False
    prev_time = time.time()

    # Cumulative unique-ID counters (never decrease)
    seen_person_ids: set = set()
    seen_object_ids: set = set()
    seen_pair_ids:   set = set()   # Layer 3 — cumulative unique trash↔bin pairs

    try:
        while True:

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] End of source.")
                    break

                frame_idx += 1
                ts = time.time()

                # ── Layer 1: Detection ────────────────────────────────────────
                detections = detector.detect(frame)

                trash_detections = []
                if trash_detector:
                    trash_detections = trash_detector.detect(
                        frame.shape, detections
                    )

                bin_detections = bin_detector.detect(frame)

                # ── Layer 2: Tracking ─────────────────────────────────────────
                tracked_objects = tracker.update(
                    detections, trash_detections, frame.shape[:2]
                )
                tracked_bins = bin_tracker.update(bin_detections)

                # Update cumulative unique-ID counters
                for t in tracked_objects:
                    if t.class_name == "person" and not t.is_trash:
                        seen_person_ids.add(t.track_id)
                    elif not t.is_trash:
                        seen_object_ids.add(t.track_id)

                # ── Layer 3: Feature Extraction ───────────────────────────────
                # Returns list of sequence dicts for all active trash↔bin pairs.
                # This does NOT classify — purely builds temporal feature sequences.
                sequences = extractor.update(tracked_objects, tracked_bins, ts)

                # Accumulate all pair_ids ever seen (never decreases)
                for seq in sequences:
                    seen_pair_ids.add(seq["pair_id"])

                # ── Console logging of Layer 3 output (every 30 frames) ───────
                if frame_idx % 30 == 0 and sequences:
                    print(f"\n[Layer3] Frame {frame_idx} — "
                          f"{len(sequences)} active pair(s):")
                    for seq in sequences:
                        last_vec = seq["sequence"][-1] if seq["sequence"] else []
                        print(
                            f"  pair={seq['pair_id']} "
                            f"| seq_len={len(seq['sequence'])} "
                            f"| dist={last_vec[0]:.1f}px "
                            f"| in_zone={last_vec[1]:.0f} "
                            f"| in_region={last_vec[2]:.0f} "
                            f"| vy={last_vec[3]:.2f} "
                            f"| t_in_region={last_vec[5]:.0f} "
                            f"| entry_score={last_vec[7]:.2f}"
                        ) if last_vec else None

                # ── Visualisation ─────────────────────────────────────────────

                # Layer 1 boxes
                vis = draw_detections(frame.copy(), detections)
                vis = draw_trash(vis, trash_detections)

                # Layer 2 — object tracks
                # draw_tracks() draws its own live-frame stats at y=24,46,68.
                # We overwrite those three lines immediately after with our
                # cumulative counters (sets that only ever grow).
                vis = draw_tracks(
                    vis,
                    tracked_objects,
                    total_trash_events=tracker.total_trash_events,
                )

                # ── Overwrite draw_tracks' live counts with cumulative counts ──
                # draw_tracks writes 3 lines starting at y=24 with cyan text.
                # We paint a black rectangle over them and redraw with correct values.
                _stats_lines = [
                    f"Tracked persons : {len(seen_person_ids)}",
                    f"Tracked objects : {len(seen_object_ids)}",
                    f"Trash events    : {tracker.total_trash_events}",
                ]
                # Black out the existing stats block
                cv2.rectangle(vis, (0, 0), (230, 24 + 3 * 22), (0, 0, 0), -1)
                for _i, _txt in enumerate(_stats_lines):
                    cv2.putText(
                        vis, _txt,
                        (10, 24 + _i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1, cv2.LINE_AA,
                    )

                # Layer 2 — bin tracks
                vis = draw_bins(
                    vis,
                    tracked_bins,
                    total_flagged=bin_tracker.total_bins_flagged,
                )

                # Layer 3 — debug overlays (bin_region, bin_zone, trail, score)
                if debug:
                    vis = extractor.draw_debug(vis, tracked_objects, tracked_bins)

                # ── HUD: Layer 3 sequence count (top-right corner) ────────────
                layer3_txt = f"L3 pairs: {len(seen_pair_ids)}"   # cumulative
                cv2.putText(
                    vis, layer3_txt,
                    (W - 160, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (180, 255, 100), 1, cv2.LINE_AA,
                )

                # ── FPS counter ───────────────────────────────────────────────
                now      = time.time()
                fps_live = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    vis, f"FPS:{fps_live:.1f}  F:{frame_idx}",
                    (W - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA,
                )

                cv2.imshow("Pipeline — L1 + L2 + L3", vis)
                if writer:
                    writer.write(vis)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Pipeline] Quit.")
                break
            elif key == ord("p"):
                paused = not paused
                print(f"[Pipeline] {'Paused' if paused else 'Resumed'}")
            elif key == ord("d"):
                debug = not debug
                extractor._debug = debug
                print(f"[Pipeline] Layer 3 debug overlay: {'ON' if debug else 'OFF'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n[Pipeline] Done — {frame_idx} frames processed.")
        print(f"[Pipeline] Cumulative persons tracked : {len(seen_person_ids)}")
        print(f"[Pipeline] Cumulative objects tracked : {len(seen_object_ids)}")
        print(f"[Pipeline] Total trash events         : {tracker.total_trash_events}")
        print(f"[Pipeline] Total bins flagged         : {bin_tracker.total_bins_flagged}")

        # ── Final Layer 3 summary ─────────────────────────────────────────────
        final_seqs = extractor.get_all_sequences()
        print(f"[Pipeline] Layer 3 unique pairs seen  : {len(seen_pair_ids)}")  # cumulative
        for seq in final_seqs:
            print(
                f"  [{seq['pair_id']}]  "
                f"sequence_length={len(seq['sequence'])}  "
                f"timestamps={len(seq['timestamps'])}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Illegal Dumping Detection — Layer 1 + 2 + 3 Pipeline"
    )
    parser.add_argument(
        "--source", required=True,
        help="Video file path, RTSP URL, or camera index (e.g. 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save annotated output to pipeline_output.mp4"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show Layer 3 debug overlays (bin regions, zones, trails, scores)"
    )
    args = parser.parse_args()
    run(args.source, args.save, args.debug)