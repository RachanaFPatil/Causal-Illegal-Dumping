"""
VidTrace Diagnostic Script
==========================
Run this BEFORE run_pipeline.py to identify exactly where the pipeline breaks.

Usage:
    python diagnose.py --source test7.mp4

It will print a step-by-step report of what works and what fails,
with the exact error message and fix for each broken component.
"""
import sys
import traceback

PASS = "  [OK]  "
FAIL = "  [FAIL]"
WARN = "  [WARN]"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── 1. Python version ──────────────────────────────────────────────────────
section("1. Python / Platform")
import platform
print(f"{PASS} Python {sys.version}")
print(f"{PASS} Platform: {platform.system()} {platform.machine()}")

# ── 2. Device ──────────────────────────────────────────────────────────────
section("2. Torch / Device")
try:
    import torch
    cuda = torch.cuda.is_available()
    mps  = (platform.system() == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available())
    device = "cuda" if cuda else ("mps" if mps else "cpu")
    print(f"{PASS} torch {torch.__version__}")
    print(f"{PASS} CUDA available : {cuda}")
    print(f"{PASS} MPS  available : {mps}")
    print(f"{PASS} Selected device: {device}")
    if device == "mps" and platform.system() != "Darwin":
        print(f"{FAIL} MPS selected on non-Mac — this will cause RT-DETR to fail silently!")
except Exception as e:
    print(f"{FAIL} torch not importable: {e}")

# ── 3. Layer1 imports ──────────────────────────────────────────────────────
section("3. Layer 1 imports")
try:
    import importlib
    mod = importlib.import_module("Layer1.config")
    print(f"{PASS} Layer1.config — DEVICE = '{mod.DEVICE}'")
    if mod.DEVICE == "mps" and platform.system() != "Darwin":
        print(f"{FAIL} Layer1.config.DEVICE is still 'mps' on Windows — fix Layer1/config.py")
    else:
        print(f"{PASS} Device looks correct for this platform")
except Exception as e:
    print(f"{FAIL} Layer1.config: {e}")

try:
    from Layer1.detector import RTDETRDetector
    print(f"{PASS} Layer1.detector — RTDETRDetector importable")
except Exception as e:
    print(f"{FAIL} Layer1.detector: {e}")
    traceback.print_exc()

try:
    from Layer1.trash_detector import TrashDetector
    print(f"{PASS} Layer1.trash_detector")
except Exception as e:
    print(f"{FAIL} Layer1.trash_detector: {e}")

try:
    from Layer1.bin_detector import BinDetector
    print(f"{PASS} Layer1.bin_detector")
except Exception as e:
    print(f"{FAIL} Layer1.bin_detector: {e}")

# ── 4. Layer2 imports ──────────────────────────────────────────────────────
section("4. Layer 2 imports")
try:
    from Layer2.track_state import TrackedObject
    print(f"{PASS} Layer2.track_state — TrackedObject")
    # Check if KalmanFilter2D exists (required by production tracker)
    try:
        from Layer2.track_state import KalmanFilter2D
        print(f"{PASS} Layer2.track_state — KalmanFilter2D present")
    except ImportError:
        print(f"{FAIL} Layer2.track_state — KalmanFilter2D MISSING")
        print(f"       → Replace Layer2/track_state.py with the fixed version provided")
except Exception as e:
    print(f"{FAIL} Layer2.track_state: {e}")
    traceback.print_exc()

try:
    from Layer2.tracker import ByteTrackWrapper
    print(f"{PASS} Layer2.tracker — ByteTrackWrapper importable")
except Exception as e:
    print(f"{FAIL} Layer2.tracker: {e}")
    traceback.print_exc()

try:
    from Layer2.bin_tracker import BinTracker
    print(f"{PASS} Layer2.bin_tracker")
except Exception as e:
    print(f"{FAIL} Layer2.bin_tracker: {e}")

try:
    from Layer2.reid import ReIDEmbedder
    print(f"{PASS} Layer2.reid — ReIDEmbedder")
except Exception as e:
    print(f"{FAIL} Layer2.reid: {e}")

try:
    from Layer2.roi_recovery import ROIRecoveryModule
    print(f"{PASS} Layer2.roi_recovery")
except Exception as e:
    print(f"{FAIL} Layer2.roi_recovery: {e}")

# ── 5. Layer 3–5 imports ───────────────────────────────────────────────────
section("5. Layer 3–5 imports")
for modpath in [
    "Layer3.feature_extractor.BinInteractionFeatureExtractor",
    "Layer4.dumping_inference.DumpingInference",
    "Layer5.agent.DumpingAgent",
]:
    parts = modpath.rsplit(".", 1)
    try:
        mod = importlib.import_module(parts[0])
        getattr(mod, parts[1])
        print(f"{PASS} {modpath}")
    except Exception as e:
        print(f"{FAIL} {modpath}: {e}")

# ── 6. ALPR deps ───────────────────────────────────────────────────────────
section("6. ALPR dependencies")
for pkg, pip_name in [
    ("fast_alpr",       "fast-alpr"),
    ("fast_plate_ocr",  "fast-plate-ocr"),
    ("easyocr",         "easyocr"),
    ("onnxruntime",     "onnxruntime"),
    ("open_image_models", "open-image-models"),
]:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"{PASS} {pkg} ({ver})")
    except ImportError:
        print(f"{WARN} {pkg} not installed — run: pip install {pip_name}")

# ── 7. Enhancer import ─────────────────────────────────────────────────────
section("7. enhancer.py")
try:
    from enhancer import Enhancer, cap_frame_generator
    print(f"{PASS} enhancer.py importable — Enhancer + cap_frame_generator OK")
except ImportError as e:
    print(f"{FAIL} enhancer.py import error: {e}")
    print("       Install missing dep above, then retry")
except Exception as e:
    print(f"{FAIL} enhancer.py runtime error: {e}")
    traceback.print_exc()

# ── 8. Tracker call test ───────────────────────────────────────────────────
section("8. Tracker smoke test (no video needed)")
try:
    from Layer1.detector import Detection
    from Layer2.tracker import ByteTrackWrapper
    import numpy as np
    tracker = ByteTrackWrapper()
    fake_frame = np.zeros((480, 848, 3), dtype=np.uint8)
    result = tracker.update([], [], fake_frame, (480, 848))
    print(f"{PASS} tracker.update() accepted 4-arg call — returned {len(result)} tracks")
except TypeError as e:
    print(f"{FAIL} tracker.update() signature mismatch: {e}")
    print("       The tracker.py in your Layer2 folder may be the wrong version")
except Exception as e:
    print(f"{FAIL} tracker.update() crashed: {e}")
    traceback.print_exc()

# ── 9. Quick detection test ────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None)
args, _ = parser.parse_known_args()

if args.source:
    section(f"9. Detection test on '{args.source}'")
    try:
        import cv2
        cap = cv2.VideoCapture(args.source)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"{FAIL} Could not read first frame from '{args.source}'")
        else:
            print(f"{PASS} Video readable — frame shape: {frame.shape}")
            try:
                from Layer1.detector import RTDETRDetector
                det = RTDETRDetector()
                dets = det.detect(frame)
                print(f"{PASS} RTDETRDetector.detect() → {len(dets)} detections on frame 1")
                for d in dets[:5]:
                    print(f"        {d.class_name}  conf={d.confidence:.2f}  bbox={d.bbox.tolist()}")
                if len(dets) == 0:
                    print(f"{WARN} Zero detections on frame 1!")
                    print("       → Check weights/rtdetr-l.pt exists")
                    print("       → Check DEVICE is not 'mps' on Windows")
            except Exception as e:
                print(f"{FAIL} RTDETRDetector.detect() failed: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"{FAIL} Video open failed: {e}")
else:
    section("9. Detection test")
    print(f"       Skipped — pass --source test7.mp4 to test detection")

print(f"\n{'='*60}")
print("  Diagnostic complete.")
print(f"{'='*60}\n")
print("Action items:")
print("  1. Fix any [FAIL] lines above FIRST")
print("  2. The most common cause of 'No confirmed events' is:")
print("     a) KalmanFilter2D missing from Layer2/track_state.py")
print("     b) Layer1/config.py still has DEVICE='mps' on Windows")
print("     c) Zero detections from RTDETRDetector (check with --source)")
