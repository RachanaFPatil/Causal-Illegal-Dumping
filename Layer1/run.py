"""
Layer 1 — Entry Point with Robust CCTV AI Engine/run.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes from baseline:
  • Shows a small HUD overlay with brightness / contrast / CLAHE status
    (purely cosmetic — does NOT affect the detection pipeline or Layer 2)
  • All other logic (source opening, VideoWriter, trash detector) is identical.
"""

import argparse
import time
import cv2
import numpy as np

from .detector import RTDETRDetector
from .trash_detector import TrashDetector
from .visualizer import draw_detections, draw_trash
from .config import TRASH_ENABLE


# ── Optional HUD: lighting status overlay ────────────────────────────────────

def _draw_lighting_hud(
    frame:      np.ndarray,
    brightness: float,
    contrast:   float,
    clahe_on:   bool,
) -> np.ndarray:
    """
    Draws a small diagnostic HUD in the bottom-left corner showing
    live lighting metrics and whether CLAHE is active this frame.
    This is purely visual — it does not affect detections or Layer 2.
    """
    H, W = frame.shape[:2]
    lines = [
        f"Brightness : {brightness:5.1f}",
        f"Contrast   : {contrast:5.1f}",
        f"CLAHE      : {'ON ' if clahe_on else 'OFF'}",
    ]
    color_clahe = (0, 200, 255) if clahe_on else (180, 180, 180)
    y_start = H - 20 - (len(lines) - 1) * 20

    for i, txt in enumerate(lines):
        color = color_clahe if i == 2 else (200, 200, 200)
        cv2.putText(
            frame, txt,
            (10, y_start + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            color, 1, cv2.LINE_AA,
        )
    return frame


# ── Main run loop ─────────────────────────────────────────────────────────────

def run(source, save: bool = False):
    detector       = RTDETRDetector()
    trash_detector = TrashDetector() if TRASH_ENABLE else None

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        out_path = "layer1_output.mp4"
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        print(f"[Layer1] Saving output to {out_path}")

    print("[Layer1] Running — press Q to quit")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Lighting report for HUD (reuses the analyze_lighting already
        #    called internally by detector.detect — this call is lightweight
        #    because grayscale computation is O(H*W) and runs in numpy) ──────
        report = detector.get_lighting_report(frame)

        # ── RT-DETR detections (persons + known objects) ─────────────────────
        # Dual-stream + fusion happens transparently inside detector.detect()
        detections = detector.detect(frame)

        # ── Trash detection (unchanged) ───────────────────────────────────────
        trash_detections = []
        if trash_detector:
            trash_detections = trash_detector.detect(frame.shape, detections)

        # ── Visualise ─────────────────────────────────────────────────────────
        vis = draw_detections(frame, detections)
        vis = draw_trash(vis, trash_detections)

        # Lighting HUD (bottom-left, cosmetic only)
        vis = _draw_lighting_hud(
            vis,
            brightness = report.brightness,
            contrast   = report.contrast,
            clahe_on   = report.needs_clahe,
        )

        # FPS counter (top-right, same as baseline)
        now      = time.time()
        fps_live = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            vis, f"FPS: {fps_live:.1f}",
            (w - 110, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.imshow("Layer 1 — Robust CCTV AI Engine", vis)
        if writer:
            writer.write(vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 1 — Robust CCTV AI Engine"
    )
    parser.add_argument("--source", default="0",  help="Video file / RTSP URL / camera index")
    parser.add_argument("--save",   action="store_true", help="Write output to layer1_output.mp4")
    args = parser.parse_args()
    run(args.source, args.save)