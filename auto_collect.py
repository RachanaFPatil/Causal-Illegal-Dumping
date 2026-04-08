"""
Auto-collect sequences from labeled videos.
"""

import cv2
from Layer1.detector       import RTDETRDetector
from Layer1.trash_detector import TrashDetector
from Layer2.tracker        import ByteTrackWrapper
from Layer3.memory         import MemoryEngine
from Layer4.dataset        import SequenceCollector


def collect_from_video(video_path: str, label: int, collector: SequenceCollector, save_gap: int = 15):
    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    tracker        = ByteTrackWrapper()
    memory         = MemoryEngine()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    last_saved = {}
    frame_idx  = 0
    tag = "DUMP" if label == 1 else "NORMAL"
    print(f"\n[AutoCollect] Starting {tag} — {video_path}  (save_gap={save_gap})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)
        tracked          = tracker.update(detections, trash_detections, (H, W))
        pairs            = memory.update(tracked)

        for pair in pairs:
            if not pair.ready():
                continue
            key = pair.pair_key
            if frame_idx - last_saved.get(key, -999) >= save_gap:
                collector.collect(pair, label=label)
                last_saved[key] = frame_idx

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx} | saved so far: {collector.saved_count}")

    cap.release()
    print(f"[AutoCollect] Done — {video_path} | sequences so far: {collector.saved_count}")


if __name__ == "__main__":
    collector = SequenceCollector()

    # Normal videos — save_gap=15
    normal_videos = [
        ("test6.mov", 0),
        ("test7.mov", 0),
        ("test8.mov", 0),
        ("test3.mov", 0),   # ← new normal
        ("test4.mov", 0),   # ← new normal
    ]

    # Dumping videos — save_gap=8 (aggressive)
    dump_videos = [
        ("test2.mp4", 1),
        ("test5.mov", 1),
        ("test1.mov", 1),   # ← new dump
    ]

    for path, label in normal_videos:
        collect_from_video(path, label, collector, save_gap=15)

    for path, label in dump_videos:
        collect_from_video(path, label, collector, save_gap=8)

    # ── Summary ───────────────────────────────────────────
    from pathlib import Path
    import numpy as np
    files = list(Path("data/sequences").glob("*.npz"))
    dump  = sum(1 for f in files if np.load(f)["label"].item() == 1)
    norm  = len(files) - dump
    print(f"\n✅ Total: {len(files)} | Dump: {dump} | Normal: {norm}")
    if dump > 0:
        print(f"   Suggested POS_WEIGHT = {round(norm / dump, 1)}")
    else:
        print("   ⚠️  No dump sequences collected — check your dumping videos")
    print("Now run: python3 -m Layer4.trainer")