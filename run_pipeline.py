"""
run_pipeline.py — Full Hybrid Detection Pipeline

Integrates all 4 layers:
    Layer 1 → RT-DETR detection + trash detector
    Layer 2 → ByteTrack multi-object tracking
    Layer 3 → Memory engine (pairing + feature extraction + sequence buffering)
    Layer 4 → Hybrid decision (transformer + ML model + rules) + logging

Usage:
    # Run on video file
    python run_pipeline.py --source path/to/video.mp4

    # Run on webcam
    python run_pipeline.py --source 0

    # Save output video
    python run_pipeline.py --source video.mp4 --save

    # Disable ML model (rule + transformer only)
    python run_pipeline.py --source video.mp4 --no_ml

    # Collect training data (label sequences interactively)
    python run_pipeline.py --source video.mp4 --collect

    # Disable logging
    python run_pipeline.py --source video.mp4 --no_log

Controls (while running):
    Q — quit
    P — pause / unpause
    S — save current frame to frame_dump/
"""

import argparse
import time
import cv2
import torch

# ── Layer 1 ───────────────────────────────────────────────
from Layer1.detector      import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer1.visualizer    import draw_detections, draw_trash
from Layer1.config        import TRASH_ENABLE

# ── Layer 2 ───────────────────────────────────────────────
from Layer2.tracker       import ByteTrackWrapper
from Layer2.visualizer    import draw_tracks

# ── Layer 3 ───────────────────────────────────────────────
from Layer3.memory        import MemoryEngine
from Layer3.visualizer    import draw_memory

# ── Layer 4 ───────────────────────────────────────────────
from Layer4.inference     import DumpingInference
from Layer4.ml_model      import MLScoringModel
from Layer4.visualizer    import draw_predictions
from Layer4.logger        import RunLogger, NullLogger
from Layer4.dataset       import SequenceCollector
from Layer4.config        import (
    INFER_EVERY_N, ENABLE_LOGGING, ENABLE_ML_MODEL,
    ML_MODEL_PATH, LOG_DIR, LOG_FLUSH_EVERY,
)


# ══════════════════════════════════════════════════════════
# Device selection
# ══════════════════════════════════════════════════════════

def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ══════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════

def run(
    source:     str,
    save:       bool = False,
    collect:    bool = False,
    enable_ml:  bool = True,
    enable_log: bool = True,
):
    device = _select_device()
    print(f"\n[Pipeline] Device: {device}")

    # ── Initialise all components ──────────────────────────
    print("[Pipeline] Initialising Layer 1 (detection)...")
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None

    print("[Pipeline] Initialising Layer 2 (tracking)...")
    tracker = ByteTrackWrapper()

    print("[Pipeline] Initialising Layer 3 (memory)...")
    memory = MemoryEngine()

    print("[Pipeline] Initialising Layer 4 (inference)...")
    engine = DumpingInference(device=device)
    engine.load()

    # ML model
    if enable_ml and ENABLE_ML_MODEL:
        ml_model = MLScoringModel()
        ml_model.load(ML_MODEL_PATH)
        engine.load_ml_model(ml_model)
        print("[Pipeline] ML scoring model attached.")
    else:
        print("[Pipeline] ML model disabled.")

    # Logger
    if enable_log and ENABLE_LOGGING:
        logger = RunLogger(log_dir=LOG_DIR, flush_every=LOG_FLUSH_EVERY)
    else:
        logger = NullLogger()

    # Data collector (optional)
    collector = SequenceCollector() if collect else None

    # ── Video source ───────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Pipeline] Source: {source}  ({W}×{H} @ {fps:.1f} fps)")

    # Video writer
    writer = None
    if save:
        out_path = "pipeline_output.mp4"
        writer   = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        print(f"[Pipeline] Saving to '{out_path}'")

    print("[Pipeline] Running — Q to quit, P to pause, S to save frame\n")

    frame_idx  = 0
    paused     = False
    prev_time  = time.time()
    predictions: dict = {}

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] End of source.")
                    break

                timestamp = time.time()
                frame_idx += 1

                # ══════════════════════════════════════════
                # LAYER 1 — Detection
                # ══════════════════════════════════════════
                detections = detector.detect(frame)

                trash_detections = []
                if trash_detector:
                    trash_detections = trash_detector.detect(
                        frame.shape, detections
                    )

                # ══════════════════════════════════════════
                # LAYER 2 — Tracking
                # ══════════════════════════════════════════
                tracks = tracker.update(
                    detections, trash_detections, frame.shape[:2]
                )

                # ══════════════════════════════════════════
                # LAYER 3 — Memory + Feature Extraction
                # ══════════════════════════════════════════
                pairs = memory.update(tracks)

                # ══════════════════════════════════════════
                # LAYER 4 — Hybrid Decision
                # ══════════════════════════════════════════
                predictions = engine.run_all_pairs(
                    pairs          = pairs,
                    tracks         = tracks,
                    infer_every_n  = INFER_EVERY_N,
                    frame_idx      = frame_idx,
                )

                # ── Logging ───────────────────────────────
                # Aggregate scores across all pairs for this frame
                ml_scores    = [d.get("ml_score",          0.0) for d in predictions.values()]
                rule_scores  = [d.get("rule_score",         0.0) for d in predictions.values()]
                trans_scores = [d.get("transformer_score",  0.0) for d in predictions.values()]
                top_decision = (
                    max(predictions.values(), key=lambda d: d.get("final_score", 0.0))
                    if predictions else {}
                )
                all_events = []
                all_features = {}
                for d in predictions.values():
                    all_events.extend(d.get("reasons", []))
                    if d.get("feature_dict"):
                        all_features = d["feature_dict"]  # log last pair's features

                logger.log(
                    frame_idx         = frame_idx,
                    timestamp         = timestamp,
                    detections        = detections,
                    tracks            = tracks,
                    pairs             = pairs,
                    features          = all_features,
                    events            = list(set(all_events)),
                    ml_score          = max(ml_scores,   default=0.0),
                    rule_score        = max(rule_scores,  default=0.0),
                    transformer_score = max(trans_scores, default=0.0),
                    final_decision    = top_decision,
                )

                # ── Console alerts ────────────────────────
                for pair_key, decision in predictions.items():
                    if decision.get("alert"):
                        print(
                            f"[Pipeline] 🚨 ALERT frame={frame_idx} "
                            f"pair={pair_key} "
                            f"type={decision['type']} "
                            f"score={decision['final_score']:.3f} "
                            f"reasons={decision['reasons']}"
                        )

                # ══════════════════════════════════════════
                # Visualisation (all layers)
                # ══════════════════════════════════════════
                vis = draw_detections(frame, detections)
                vis = draw_trash(vis, trash_detections)
                vis = draw_tracks(vis, tracks, tracker.total_trash_events)
                vis = draw_memory(vis, pairs, tracks)
                vis = draw_predictions(vis, pairs, predictions)

                # FPS overlay
                now      = time.time()
                fps_live = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(vis, f"FPS:{fps_live:.1f}  F:{frame_idx}",
                            (W - 160, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

                # ── Data collection mode ──────────────────
                if collector and pairs:
                    ready_pairs = [p for p in pairs if p.ready()]
                    if ready_pairs:
                        _show_collect_hint(vis, H)

                cv2.imshow("Illegal Dumping Detection — Hybrid Pipeline", vis)

                if writer:
                    writer.write(vis)

            # ── Key handling ──────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Pipeline] Quit requested.")
                break

            elif key == ord("p"):
                paused = not paused
                print(f"[Pipeline] {'Paused' if paused else 'Resumed'}")

            elif key == ord("s"):
                import pathlib
                pathlib.Path("frame_dump").mkdir(exist_ok=True)
                fname = f"frame_dump/frame_{frame_idx:06d}.jpg"
                cv2.imwrite(fname, vis)
                print(f"[Pipeline] Saved frame → {fname}")

            # ── Collection: D=dump, N=normal ──────────────
            elif collect and pairs:
                ready_pairs = [p for p in pairs if p.ready()]
                if key == ord("d") and ready_pairs:
                    for p in ready_pairs:
                        collector.collect(p, label=1)
                elif key == ord("n") and ready_pairs:
                    for p in ready_pairs:
                        collector.collect(p, label=0)

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.close()

        if collector:
            print(f"[Pipeline] 📦 Collected {collector.saved_count} sequences")

        print(f"[Pipeline] Done — {frame_idx} frames processed.")


def _show_collect_hint(vis, H: int):
    cv2.putText(vis, "COLLECT: D=dump  N=normal",
                (10, H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Illegal Dumping Detection — Hybrid Pipeline")
    parser.add_argument("--source",  required=True, help="Video path or camera index")
    parser.add_argument("--save",    action="store_true", help="Save output video")
    parser.add_argument("--collect", action="store_true", help="Enable training data collection")
    parser.add_argument("--no_ml",   action="store_true", help="Disable ML scoring model")
    parser.add_argument("--no_log",  action="store_true", help="Disable JSON logging")
    args = parser.parse_args()

    run(
        source     = args.source,
        save       = args.save,
        collect    = args.collect,
        enable_ml  = not args.no_ml,
        enable_log = not args.no_log,
    )