"""
VidTrace — Full Pipeline Runner (Layers 1-5)

Controls
--------
Q  : quit
P  : pause / resume
D  : toggle Layer 3 debug overlay
R  : toggle Layer 5 reasoning overlay
"""

from __future__ import annotations

import argparse
import time
from collections import deque
import cv2
import numpy as np
from enhancer import Enhancer, EnhancementResult, cap_frame_generator
from Layer1.detector          import RTDETRDetector
from Layer1.trash_detector    import TrashDetector
from Layer1.bin_detector      import BinDetector
from Layer2.tracker           import ByteTrackWrapper
from Layer2.bin_tracker       import BinTracker
from Layer2.visualizer        import draw_tracks
from Layer2.bin_visualizer    import draw_bins
from Layer3.feature_extractor import BinInteractionFeatureExtractor
from Layer4.dumping_inference import DumpingInference
from Layer5.agent             import DumpingAgent
from Layer5.visualizer        import (
    draw_l5_verdicts,
    draw_l5_reasoning,
    draw_l5_evidence_bars,
    draw_l5_summary_box,
)

_COL_BG = (20, 20, 20)


def _bbox_area(bbox) -> float:
    """
    Return w*h for a bbox [x1, y1, x2, y2].
    Returns 0.0 for ghost/off-screen tracks (negative coords or inverted box).
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    if x2 < 0 or y2 < 0 or x1 > 10000 or y1 > 10000:
        return 0.0
    return float(w * h)


def _bbox_centre(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _draw_l4_banners(frame, events, W, H):
    visible = [e for e in events if e.event != "pending"]
    if not visible:
        return frame
    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 4
    for i, ev in enumerate(visible):
        col   = (0, 140, 255) if ev.event == "illegal_dumping" else (100, 200, 100)
        label = f"[L4] {ev.pair_id} | {ev.event} | conf={ev.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        x = W - tw - 15
        y = H - 20 - i * 22
        cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad), _COL_BG, -1)
        cv2.putText(frame, label, (x, y), font, scale, col, thick, cv2.LINE_AA)
    return frame


def run(source: str, save: bool = False, debug: bool = False) -> None:
    print("[Pipeline] Booting VidTrace...")

    detector       = RTDETRDetector()
    trash_detector = TrashDetector()
    bin_detector   = BinDetector()
    tracker        = ByteTrackWrapper()
    bin_tracker    = BinTracker()
    extractor      = BinInteractionFeatureExtractor(debug=debug)
    inference      = DumpingInference()
    agent          = DumpingAgent()
    enhancer       = Enhancer()
    frame_buffer   = deque(maxlen=100)

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if save:
        out_name = "vidtrace_output.mp4"
        writer   = cv2.VideoWriter(
            out_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        print(f"[Pipeline] Saving -> {out_name}")

    print("[Pipeline] Running — Q:quit P:pause D:L3-debug R:L5-reasoning")

    prev_time   = time.time()
    frame_idx   = 0
    paused      = False
    show_reason = True
    last_l5     = []
    vis         = np.zeros((H, W, 3), dtype=np.uint8)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Layers 1-4
                frame_buffer.append(frame.copy())
                dets         = detector.detect(frame)
                trash_dets   = trash_detector.detect(frame.shape, dets)
                bin_dets     = bin_detector.detect(frame)
                tracked_objs = tracker.update(dets, trash_dets, (H, W))
                tracked_bins = bin_tracker.update(bin_dets)
                ts           = time.time()
                extractor.update(tracked_objs, tracked_bins, ts)
                l4_events    = inference.update(tracked_objs, tracked_bins)

                # Layer 5
                l5_new = agent.update(frame_idx, tracked_objs, tracked_bins, l4_events)
                if l5_new:
                    last_l5 = l5_new
                    for verdict in l5_new:
                        if verdict["event"] == "illegal_dumping":

                            person_obj = next(
                                (o for o in tracked_objs
                                 if o.track_id == verdict["person_id"]),
                                None
                            )

                            # Override: pick the person CLOSEST to the trash
                            # object — more reliable than largest area, which
                            # tends to pick background bystanders.
                            # Ghost tracks are rejected via _bbox_area > 500.
                            trash_obj = next(
                                (o for o in tracked_objs
                                 if o.track_id == verdict["object_id"]),
                                None
                            )
                            all_persons = [
                                o for o in tracked_objs
                                if o.class_name == "person"
                                and _bbox_area(o.bbox) > 500
                            ]
                            if all_persons and trash_obj is not None:
                                tx, ty = _bbox_centre(trash_obj.bbox)
                                closest = min(
                                    all_persons,
                                    key=lambda o: (
                                        (_bbox_centre(o.bbox)[0] - tx) ** 2
                                        + (_bbox_centre(o.bbox)[1] - ty) ** 2
                                    )
                                )
                                if (
                                    person_obj is None
                                    or closest.track_id != person_obj.track_id
                                ):
                                    print(
                                        f"[DEBUG] overriding to "
                                        f"person_id={closest.track_id} "
                                        f"(closest to trash "
                                        f"{verdict['object_id']})"
                                    )
                                    person_obj = closest

                            # Debug: print all tracks and selected person
                            if person_obj is not None:
                                for o in tracked_objs:
                                    print(
                                        f"[DEBUG] track_id={o.track_id} "
                                        f"class={o.class_name} bbox={o.bbox}"
                                    )
                                print(
                                    f"[DEBUG] using person_id="
                                    f"{person_obj.track_id} "
                                    f"bbox={person_obj.bbox}"
                                )
                            else:
                                print(
                                    f"[DEBUG] no valid person found for "
                                    f"verdict P{verdict['person_id']} "
                                    f"T{verdict['object_id']} — "
                                    f"running enhancer on full frame"
                                )

                            # Run enhancer regardless of whether a valid
                            # person_obj was found — the plate may still be
                            # visible in the lookahead frames even if the
                            # person/trash tracks have gone stale.
                            combined = list(cap_frame_generator(cap, 150))
                            result = enhancer.process_event(
                                frame=frame,
                                person_bbox=(
                                    person_obj.bbox
                                    if person_obj is not None
                                    else None
                                ),
                                person_id=verdict["person_id"],
                                pair_id=verdict["pair_id"],
                                save_dir="evidence",
                                frame_iter=(
                                    iter(combined) if combined else None
                                ),
                            )
                            if result.plate_text:
                                print(
                                    f"[Enhancer] 🚗 PLATE: "
                                    f"{result.plate_text} "
                                    f"(conf={result.plate_conf:.2f})"
                                )
                            else:
                                print(
                                    f"[Enhancer] ⚠️  No plate read | "
                                    f"blur={result.blur_score:.1f} | "
                                    f"scanned={result.frames_scanned}f"
                                )

                # Visualise
                vis = frame.copy()
                vis = draw_tracks(vis, tracked_objs, tracker.total_trash_events)
                vis = draw_bins(vis, tracked_bins, bin_tracker.total_bins_flagged)
                if debug:
                    vis = extractor.draw_debug(vis, tracked_objs, tracked_bins)
                vis = _draw_l4_banners(vis, l4_events, W, H)
                vis = draw_l5_verdicts(vis, last_l5)
                vis = draw_l5_evidence_bars(vis, agent.get_all_results())
                vis = draw_l5_summary_box(vis, agent.get_all_results())
                if show_reason:
                    vis = draw_l5_reasoning(vis, agent)

                # HUD
                persons = sum(1 for t in tracked_objs if t.class_name == "person")
                objects = sum(1 for t in tracked_objs if t.is_trash)
                cv2.rectangle(vis, (0, 0), (235, 95), (0, 0, 0), -1)
                for row, txt in enumerate([
                    f"Persons : {persons}",
                    f"Objects : {objects}",
                    f"L5 events: {len(agent.get_all_results())}",
                ]):
                    cv2.putText(vis, txt, (10, 25 + row * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1)

                now       = time.time()
                fps_live  = 1.0 / (now - prev_time + 1e-9)
                prev_time = now
                cv2.putText(vis, f"FPS:{fps_live:.1f}  F:{frame_idx}",
                            (W-165, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                cv2.putText(vis,
                            f"Bins:{len(tracked_bins)} visible "
                            f"{bin_tracker.total_bins_flagged} flagged",
                            (W-260, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,255), 1)

                if writer:
                    writer.write(vis)

            cv2.imshow("VidTrace — Illegal Dumping Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("p"):
                paused = not paused
                print("[Pipeline]", "Paused" if paused else "Resumed")
            elif key == ord("d"):
                debug = not debug
                extractor._debug = debug
                print(f"[Debug] L3 overlay: {'ON' if debug else 'OFF'}")
            elif key == ord("r"):
                show_reason = not show_reason
                print(f"[Debug] L5 reasoning: {'ON' if show_reason else 'OFF'}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n[Pipeline] Done. Frames: {frame_idx}")

        all_results = agent.get_all_results()
        if all_results:
            print(f"\n[Layer5] ══ Final Confirmed Events ({len(all_results)}) ══")
            for r in all_results:
                tag = "🚨 VIOLATION" if r["violation"] else "✅ LEGAL"
                print(
                    f"  {tag}  {r['event']:<20} conf={r['confidence']:.2f}  "
                    f"P{r['person_id']} T{r['object_id']}  "
                    f"coupling={r.get('coupling_frames','?')}f  "
                    f"cos={r.get('peak_coupling',0):.2f}  "
                    f"intent={r.get('intent_score',0):.2f}  "
                    f"frames={r['frames']}"
                )
                print(f"       reason: {r['reason']}")
                print(f"       L5 log: {' -> '.join(r.get('reasoning_log', []))}")
        else:
            print("[Layer5] No confirmed events.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--save",   action="store_true")
    p.add_argument("--debug",  action="store_true")
    args = p.parse_args()
    run(args.source, args.save, args.debug)