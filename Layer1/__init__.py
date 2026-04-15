# Layer 1 — Robust CCTV AI Engine/__init__.py
from .detector import RTDETRDetector, Detection, analyze_lighting, apply_clahe, fuse_detections, LightingReport
from .trash_detector import TrashDetector, TrashDetection