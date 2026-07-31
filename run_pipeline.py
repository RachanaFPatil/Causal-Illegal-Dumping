"""
VidTrace — Full Pipeline Runner (Layers 1–5 + ALPR + FaceID + Challan)
=======================================================================
KEY FIXES in this version
--------------------------
1. ONE challan per actual dumping act — deduped by person_id AND trash_id.
   Multiple pair_ids from re-detection of same object collapse to one challan.

2. GPS extracted from video file via ffprobe TAG:location automatically.
   --lat/--lon args still work as override.

3. Hotspot counts per unique camera_id/video per GPS location.
   ONE hotspot event per video run, not per violation.

4. DeliveryAgent + FaceID fully wired.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import re
import subprocess
import sys
import cv2
import numpy as np
import torch


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

DEVICE = _best_device()

def _force_patch_l1_device(device: str) -> None:
    try:
        if "Layer1.config" in sys.modules:
            sys.modules["Layer1.config"].DEVICE = device
        else:
            mod = importlib.import_module("Layer1.config")
            mod.DEVICE = device
        print(f"[Pipeline] Device -> {device}")
    except Exception as exc:
        print(f"[Pipeline] WARNING: {exc}")

_force_patch_l1_device(DEVICE)

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
    draw_l5_verdicts, draw_l5_reasoning,
    draw_l5_evidence_bars, draw_l5_summary_box,
)

try:
    from enhancer import Enhancer, cap_frame_generator
    _enhancer = Enhancer(); ALPR_AVAILABLE = True
    print("[ALPR] enhancer.py loaded")
except Exception as _e:
    ALPR_AVAILABLE = False; _enhancer = None
    print(f"[ALPR] not available: {_e}")

try:
    from penalty_manager_patch import apply_patch; apply_patch()
except Exception:
    pass

try:
    from penalty_manager import PenaltyManager
    _penalty_manager = PenaltyManager(); CHALLAN_AVAILABLE = True
    print("[Challan] penalty_manager loaded.")
except Exception as _e:
    CHALLAN_AVAILABLE = False; _penalty_manager = None
    print(f"[Challan] not available: {_e}")

try:
    from delivery_agent import DeliveryAgent
    _delivery_agent = DeliveryAgent(); _delivery_agent.start()
    DELIVERY_AVAILABLE = True
    print("[Delivery] DeliveryAgent started.")
except Exception as _e:
    DELIVERY_AVAILABLE = False; _delivery_agent = None
    print(f"[Delivery] not available: {_e}")

try:
    from FaceID.face_id_module import FaceIDModule
    _faceid_module = FaceIDModule(); FACEID_AVAILABLE = True
    print("[FaceID] Module loaded.")
except Exception as _e:
    FACEID_AVAILABLE = False; _faceid_module = None
    print(f"[FaceID] not available: {_e}")

try:
    from hotspot.hotspot_manager import HotspotManager
    _hotspot_mgr = HotspotManager(); HOTSPOT_AVAILABLE = True
    print("[Hotspot] HotspotManager loaded.")
except Exception as _e:
    HOTSPOT_AVAILABLE = False; _hotspot_mgr = None
    print(f"[Hotspot] not available: {_e}")


def _extract_gps_from_video(video_path: str) -> tuple:
    """Extract GPS from video TAG:location via ffprobe. Returns (lat, lon)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_format", video_path],
            capture_output=True, text=True, timeout=10
        )
        out = result.stdout + result.stderr
        m = re.search(r'TAG:location(?:-eng)?=([+-]\d+\.\d+)([+-]\d+\.\d+)', out)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            print(f"[GPS] From video: {lat:.4f},{lon:.4f}")
            return lat, lon
    except FileNotFoundError:
        print("[GPS] ffprobe not found. Install ffmpeg to auto-extract GPS.")
    except Exception as e:
        print(f"[GPS] error: {e}")
    return 0.0, 0.0


def _draw_plate(frame: np.ndarray, plate_text: str, conf: float) -> np.ndarray:
    if not plate_text: return frame
    H, W = frame.shape[:2]
    label = f"PLATE: {plate_text}  ({conf:.2f})"
    font = cv2.FONT_HERSHEY_DUPLEX; scale, thick = 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    x, y = W - tw - 18, H - 18
    cv2.rectangle(frame, (x-6, y-th-8), (x+tw+6, y+6), (15,15,15), -1)
    cv2.putText(frame, label, (x,y), font, scale, (0,220,255), thick, cv2.LINE_AA)
    return frame


def _sharpness(f: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


class _FrameBuffer:
    def __init__(self, maxlen=150):
        self._buf = []; self._maxlen = maxlen

    def push(self, idx, frame):
        self._buf.append((idx, frame.copy()))
        if len(self._buf) > self._maxlen: self._buf.pop(0)

    def best_near(self, target, window=40):
        cands = [(i,f) for i,f in self._buf if abs(i-target) <= window]
        if not cands: return 0, None
        best_i, best_f = max(cands, key=lambda x: _sharpness(x[1]))
        print(f"[BEST FRAME] offset={best_i-target:+d} blur={_sharpness(best_f):.1f}")
        return best_i, best_f


def run(source: str, save: bool = False, location: str = "",
        enable_challan: bool = True, enable_faceid: bool = True,
        latitude: float = 0.0, longitude: float = 0.0) -> None:

    print("[Pipeline] Booting VidTrace...")

    detector          = RTDETRDetector()
    trash_detector    = TrashDetector()
    bin_detector      = BinDetector(device="cpu")
    tracker           = ByteTrackWrapper()
    bin_tracker       = BinTracker()
    feature_extractor = BinInteractionFeatureExtractor(debug=False)
    dumping_inference = DumpingInference()
    agent             = DumpingAgent()

    src       = int(source) if source.isdigit() else source
    camera_id = os.path.splitext(os.path.basename(str(source)))[0]

    # GPS: try video metadata, fall back to args
    if latitude == 0.0 and longitude == 0.0:
        v_path = str(source)
        if any(v_path.lower().endswith(ext) for ext in ('.mp4','.avi','.mov','.mkv')):
            latitude, longitude = _extract_gps_from_video(v_path)

    if not location:
        location = camera_id

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path = "vidtrace_output.mp4"

    writer = None
    if save:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
        print(f"[Pipeline] Saving -> {out_path}")

    print(f"[Pipeline] Location: {location}  GPS: {latitude:.4f},{longitude:.4f}")
    print("[Pipeline] Running -- Q:quit P:pause D:L3-debug R:L5-reasoning")

    frame_idx = 0; paused = False
    show_l3_debug = False; show_l5_reason = False
    frame_buf         = _FrameBuffer(maxlen=150)
    all_verdicts      = []
    last_plate        = ("", 0.0)
    alpr_done_events  = set()
    challan_issued    = {}

    # DEDUP: one challan per person per run, one challan per trash object per run
    challan_issued_persons: dict = {}  # person_id -> challan_id
    challan_issued_trash:   dict = {}  # trash_id -> challan_id
    dumping_frames:         dict = {}  # event_key -> frame
    any_violation_confirmed = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            frame_buf.push(frame_idx, frame)

        detections       = detector.detect(frame)
        trash_detections = trash_detector.detect(frame.shape, detections)
        bin_detections   = bin_detector.detect(frame)
        tracked_objects  = tracker.update(detections, trash_detections, frame, (H,W))
        tracked_bins     = bin_tracker.update(bin_detections)
        feature_extractor.update(tracked_objects, tracked_bins, timestamp=frame_idx/fps)
        l4_events        = dumping_inference.update(tracked_objects, tracked_bins)
        new_verdicts     = agent.update(frame_idx, tracked_objects, tracked_bins, l4_events)
        all_verdicts.extend(new_verdicts)

        for verdict in new_verdicts:
            if not verdict.get("violation"):
                continue

            event_key         = verdict.get("pair_id", str(frame_idx))
            verdict_person_id = verdict.get("person_id", -1)
            verdict_trash_id  = verdict.get("object_id", -1)
            confidence        = verdict.get("confidence", 0.0)

            if event_key not in dumping_frames:
                dumping_frames[event_key] = frame.copy()

            # DEDUP CHECK
            if verdict_person_id in challan_issued_persons:
                print(f"[Challan] SKIP P{verdict_person_id} already issued "
                      f"{challan_issued_persons[verdict_person_id]}")
                continue
            if verdict_trash_id in challan_issued_trash:
                print(f"[Challan] SKIP T{verdict_trash_id} already claimed")
                continue

            # ALPR
            alpr_result = None
            if event_key not in alpr_done_events:
                alpr_done_events.add(event_key)
                if ALPR_AVAILABLE and _enhancer:
                    _, bf = frame_buf.best_near(frame_idx, window=40)
                    if bf is None: bf = frame
                    pbbox = None
                    for obj in tracked_objects:
                        if obj.track_id == verdict_person_id:
                            pbbox = tuple(obj.bbox.tolist()); break
                    saved_pos  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_iter = cap_frame_generator(cap, n_frames=150)
                    alpr_result = _enhancer.process_event(
                        frame=bf, person_bbox=pbbox,
                        person_id=str(verdict_person_id),
                        pair_id=event_key, save_dir="evidence",
                        frame_iter=frame_iter,
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
                    if alpr_result and alpr_result.plate_text:
                        last_plate = (alpr_result.plate_text, alpr_result.plate_conf)

            plate_for_challan = (alpr_result.plate_text
                                 if alpr_result and alpr_result.plate_text else None)
            evidence_plate = (alpr_result.saved_paths[0]
                              if alpr_result and alpr_result.saved_paths else None)

            # VEHICLE PATH
            if plate_for_challan and enable_challan and CHALLAN_AVAILABLE:
                try:
                    challan_id = _penalty_manager.create_violation(
                        plate_number              = plate_for_challan,
                        evidence_plate_image_path = evidence_plate,
                        location                  = location,
                        confidence                = confidence,
                    )
                    pdf_path = _penalty_manager.generate_challan(challan_id)
                    challan_issued[event_key]                 = challan_id
                    challan_issued_persons[verdict_person_id] = challan_id
                    challan_issued_trash[verdict_trash_id]    = challan_id
                    any_violation_confirmed = True
                    last_plate = ("", 0.0)
                    print(f"[Challan] Vehicle | {challan_id} | plate={plate_for_challan}")
                    if pdf_path and DELIVERY_AVAILABLE and _delivery_agent:
                        _delivery_agent.notify_new_challan(challan_id)
                except Exception as ce:
                    print(f"[Challan] Vehicle failed: {ce}")

            # FACEID / PEDESTRIAN PATH
            elif enable_challan and enable_faceid and FACEID_AVAILABLE:
                try:
                    dump_f = dumping_frames.get(event_key, frame)
                    _, best_f = frame_buf.best_near(frame_idx, window=40)
                    if best_f is None: best_f = frame
                    result = _faceid_module.process(
                        dumping_frame = dump_f,
                        best_frame    = best_f,
                        location      = location,
                        confidence    = confidence,
                        pair_id       = event_key,
                    )
                    if result and result.get("challan_id"):
                        challan_id = result["challan_id"]
                        challan_issued[event_key]                 = challan_id
                        challan_issued_persons[verdict_person_id] = challan_id
                        challan_issued_trash[verdict_trash_id]    = challan_id
                        any_violation_confirmed = True
                        print(f"[FaceID] Pedestrian | {challan_id} | "
                              f"name={result.get('name','?')}")
                    else:
                        # Unknown face — count for hotspot but no challan
                        any_violation_confirmed = True
                        challan_issued_persons[verdict_person_id] = "UNKNOWN"
                        challan_issued_trash[verdict_trash_id]    = "UNKNOWN"
                except Exception as fe:
                    print(f"[FaceID] Failed: {fe}")

            # ANONYMOUS PEDESTRIAN (no FaceID available)
            elif enable_challan and CHALLAN_AVAILABLE:
                try:
                    challan_id = _penalty_manager.create_violation(
                        plate_number=None, evidence_plate_image_path=evidence_plate,
                        location=location, confidence=confidence,
                    )
                    pdf_path = _penalty_manager.generate_challan(challan_id)
                    challan_issued[event_key]                 = challan_id
                    challan_issued_persons[verdict_person_id] = challan_id
                    challan_issued_trash[verdict_trash_id]    = challan_id
                    any_violation_confirmed = True
                    print(f"[Challan] Anonymous | {challan_id}")
                    if pdf_path and DELIVERY_AVAILABLE and _delivery_agent:
                        _delivery_agent.notify_new_challan(challan_id)
                except Exception as ce:
                    print(f"[Challan] Anonymous failed: {ce}")

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
        cv2.putText(vis, f"F:{frame_idx}", (W-90,H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)
        cv2.imshow("VidTrace -- Illegal Dumping Detection", vis)
        if writer: writer.write(vis)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord("q"): break
        elif key == ord("p"):
            paused = not paused
            print(f"[Pipeline] {'Paused' if paused else 'Resumed'}")
        elif key == ord("d"): show_l3_debug = not show_l3_debug
        elif key == ord("r"): show_l5_reason = not show_l5_reason

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    if DELIVERY_AVAILABLE and _delivery_agent:
        _delivery_agent.stop()

    print(f"\n[Pipeline] Done. Frames: {frame_idx}")

    final_results = agent.get_all_results()
    if final_results:
        print(f"\n[Layer5] Final Events ({len(final_results)})")
        for r in final_results:
            tag = "VIOLATION" if r["violation"] else "LEGAL"
            print(f"  {tag}  P{r['person_id']} T{r['object_id']}  "
                  f"conf={r['confidence']:.2f}  frames={r['frames']}")

    unique_challans = set(v for v in challan_issued.values()
                          if v and v != "UNKNOWN")
    if unique_challans:
        print(f"\n[Challan] Issued: {len(unique_challans)}")
        for cid in unique_challans:
            print(f"  {cid}")
    else:
        print("\n[Challan] No challans issued this run.")

    # HOTSPOT: one event per video, keyed by camera_id
    if HOTSPOT_AVAILABLE and _hotspot_mgr and any_violation_confirmed:
        try:
            n = len(unique_challans) if unique_challans else 1
            result = _hotspot_mgr.log_event(
                camera_id       = camera_id,
                location        = location,
                latitude        = latitude,
                longitude       = longitude,
                violation_count = n,
            )
            escalated = _hotspot_mgr.run_hotspot_check()
            print(f"[Hotspot] Logged. Escalated={len(escalated) if escalated else 0}")
        except Exception as he:
            print(f"[Hotspot] Error: {he}")

    if CHALLAN_AVAILABLE and _penalty_manager:
        try:
            _penalty_manager.check_and_escalate()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="0")
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--location",   default="")
    parser.add_argument("--lat",        type=float, default=0.0)
    parser.add_argument("--lon",        type=float, default=0.0)
    parser.add_argument("--no-challan", action="store_true")
    parser.add_argument("--no-faceid",  action="store_true")
    args = parser.parse_args()
    run(source=args.source, save=args.save, location=args.location,
        enable_challan=not args.no_challan, enable_faceid=not args.no_faceid,
        latitude=args.lat, longitude=args.lon)