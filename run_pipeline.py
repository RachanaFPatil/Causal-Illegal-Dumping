"""
Full Pipeline Runner — Layer 1 + 2 + 3 + 4

Usage:
    python run_pipeline.py --source test6.mov
    python run_pipeline.py --source test6.mov --save
    python run_pipeline.py --source test6.mov --collect   ← label sequences for training
"""

import argparse
import time
import cv2

from Layer1.detector       import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer2.tracker        import ByteTrackWrapper
from Layer2.visualizer     import draw_tracks
from Layer3.memory         import MemoryEngine
from Layer3.visualizer     import draw_memory as draw_pairs
from Layer4.inference      import DumpingInference
from Layer4.visualizer     import draw_predictions
from Layer4.dataset        import SequenceCollector
from Layer4.config         import INFER_EVERY_N


def run(source: str, save: bool = False, collect: bool = False):

    # ── Init all layers ───────────────────────────────────
    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    tracker        = ByteTrackWrapper()
    memory         = MemoryEngine()
    model          = DumpingInference(device="mps")
    model.load()   # loads weights if available, otherwise runs untrained

    collector = SequenceCollector() if collect else None

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if save:
        writer = cv2.VideoWriter(
            "layer4_output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )
        print("[Pipeline] Saving → layer4_output.mp4")

    if collect:
        print("[Pipeline] COLLECT MODE — press D=dump, N=normal, S=skip, Q=quit")
    else:
        print("[Pipeline] Running Layer 1+2+3+4 — press Q to quit")

    frame_idx   = 0
    predictions = {}    # {pair_key: float}  — most recent model output per pair
    prev        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Layer 1 — Detection ───────────────────────────
        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)

        # ── Layer 2 — Tracking ────────────────────────────
        tracked = tracker.update(detections, trash_detections, (H, W))

        # ── Layer 3 — Memory ──────────────────────────────
        pairs       = memory.update(tracked)
        ready_pairs = [p for p in pairs if p.ready()]

        # ── Layer 4 — Inference (every N frames) ─────────
        if frame_idx % INFER_EVERY_N == 0:
            for pair in ready_pairs:
                is_viol, prob = model.is_violation(pair)
                predictions[pair.pair_key] = prob

                if is_viol:
                    print(f"[Layer4] 🚨 VIOLATION  person={pair.person_id} "
                          f"obj={pair.object_id}  prob={prob:.2%}  "
                          f"frame={frame_idx}")

        # ── Remove predictions for dead pairs ─────────────
        active_keys = {p.pair_key for p in pairs}
        predictions = {k: v for k, v in predictions.items() if k in active_keys}

        # ── Collect mode: label current ready pairs ───────
        if collect and ready_pairs:
            # Show current pairs in terminal
            for p in ready_pairs:
                print(f"  Pair {p.pair_key} | held={p.held_frames} "
                      f"frames_seen={p.frames_seen} | "
                      f"prob={predictions.get(p.pair_key, -1):.2%}")

        # ── Visualise ─────────────────────────────────────
        vis = draw_tracks(frame.copy(), tracked, tracker.total_trash_events)
        vis = draw_pairs(vis, pairs, tracked)
        vis = draw_predictions(vis, ready_pairs, predictions)

        now = time.time()
        cv2.putText(vis, f"FPS: {1/(now-prev+1e-9):.1f}",
                    (W - 110, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"Pairs: {len(ready_pairs)}/{len(pairs)}",
                    (W - 110, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        prev = now

        cv2.imshow("Illegal Dumping Detection — Layer 1-4", vis)

        if writer:
            writer.write(vis)

        # ── Key handling ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif collect and ready_pairs:
            # Label the FIRST ready pair (most active)
            target_pair = ready_pairs[0]
            if key == ord("d"):
                collector.collect(target_pair, label=1)
            elif key == ord("n"):
                collector.collect(target_pair, label=0)
            elif key == ord("s"):
                print(f"[Collector] Skipped pair {target_pair.pair_key}")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if collect and collector:
        print(f"\n[Pipeline] Collection done. {collector.saved_count} sequences saved "
              f"→ {collector.data_dir}")
    print("[Pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="0")
    parser.add_argument("--save",    action="store_true")
    parser.add_argument("--collect", action="store_true",
                        help="Enable sequence labelling mode (D/N/S keys)")
    args = parser.parse_args()
    run(args.source, args.save, args.collect)