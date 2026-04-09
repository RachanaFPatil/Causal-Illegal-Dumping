# Layer4/logger.py
"""
Global Logging System — Layer 4

Logs EVERYTHING to a JSON file per run:
    logs/run_<timestamp>.json

Each entry corresponds to one frame and contains:
    {
        "frame":          int,
        "timestamp":      float,
        "detections":     [...],
        "tracks":         [...],
        "features":       {...},
        "events":         [...],
        "ml_score":       float,
        "rule_score":     float,
        "transformer_score": float,
        "final_decision": {...}
    }

Usage:
    logger = RunLogger()
    logger.log(frame_idx, timestamp, detections, tracks, features,
               events, ml_score, rule_score, transformer_score, decision)
    logger.close()   # flushes and closes file
"""

import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from Layer1.detector    import Detection
from Layer2.track_state import TrackedObject
from Layer3.pair_state  import PairState


LOG_DIR = "logs"


# ══════════════════════════════════════════════════════════
# JSON serialiser — handles numpy types
# ══════════════════════════════════════════════════════════

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════
# Serialisers for domain objects
# ══════════════════════════════════════════════════════════

def _serialise_detection(d: Detection) -> Dict:
    return {
        "bbox":       d.bbox.tolist() if hasattr(d.bbox, "tolist") else list(d.bbox),
        "class_name": d.class_name,
        "confidence": round(float(d.confidence), 4),
        "class_id":   int(d.class_id),
    }


def _serialise_track(t: TrackedObject) -> Dict:
    return {
        "track_id":   int(t.track_id),
        "bbox":       t.bbox.tolist() if hasattr(t.bbox, "tolist") else list(t.bbox),
        "class_name": t.class_name,
        "confidence": round(float(t.confidence), 4),
        "is_trash":   bool(t.is_trash),
        "trash_label":t.trash_label,
        "trash_how":  t.trash_how,
    }


def _serialise_pair(p: PairState) -> Dict:
    return {
        "person_id":      int(p.person_id),
        "object_id":      int(p.object_id),
        "frames_seen":    int(p.frames_seen),
        "held_frames":    int(p.held_frames),
        "released_frames":int(p.released_frames),
        "ever_held":      bool(p.ever_held),
        "seq_len":        len(p.sequence),
    }


# ══════════════════════════════════════════════════════════
# RunLogger
# ══════════════════════════════════════════════════════════

class RunLogger:
    """
    Writes one JSON log file per run.

    The file is a JSON array opened at start and closed at end.
    Entries are flushed periodically so crashes don't lose data.

    Args:
        log_dir:      directory for log files (created if absent)
        flush_every:  write to disk every N frames (default 30)
    """

    def __init__(self, log_dir: str = LOG_DIR, flush_every: int = 30):
        self._flush_every = flush_every
        self._buffer: List[Dict] = []
        self._frame_count = 0
        self._run_start   = time.time()
        self._alert_count = 0

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{ts}.json"
        self._path = str(Path(log_dir) / filename)

        # Start the JSON array
        with open(self._path, "w") as f:
            f.write("[\n")

        self._first_entry = True
        print(f"[Logger] 📝 Logging to '{self._path}'")

    # ── Main log call ──────────────────────────────────────
    def log(
        self,
        frame_idx:         int,
        timestamp:         float,
        detections:        List[Detection],
        tracks:            List[TrackedObject],
        pairs:             List[PairState],
        features:          Dict,                  # feature_dict from agent
        events:            List[str],             # reason strings
        ml_score:          float,
        rule_score:        float,
        transformer_score: float,
        final_decision:    Dict,
    ):
        """
        Log one frame.  Call this every frame in the inference loop,
        even if no alert — normal frames are essential training data.
        """
        self._frame_count += 1
        if final_decision.get("alert"):
            self._alert_count += 1

        entry = {
            "frame":             frame_idx,
            "timestamp":         round(timestamp, 4),
            "run_elapsed_s":     round(timestamp - self._run_start, 2),
            "detections":        [_serialise_detection(d) for d in detections],
            "tracks":            [_serialise_track(t)     for t in tracks],
            "pairs":             [_serialise_pair(p)      for p in pairs],
            "features":          features,
            "events":            events,
            "ml_score":          round(float(ml_score),          4),
            "rule_score":        round(float(rule_score),         4),
            "transformer_score": round(float(transformer_score),  4),
            "final_decision":    final_decision,
        }

        self._buffer.append(entry)

        if self._frame_count % self._flush_every == 0:
            self._flush()

    # ── Flush buffer to file ───────────────────────────────
    def _flush(self):
        if not self._buffer:
            return

        with open(self._path, "a") as f:
            for entry in self._buffer:
                if not self._first_entry:
                    f.write(",\n")
                json.dump(entry, f, cls=_NumpyEncoder, indent=2)
                self._first_entry = False

        self._buffer = []

    # ── Close ──────────────────────────────────────────────
    def close(self):
        """
        Must be called at end of run.  Flushes remaining entries,
        closes the JSON array, and prints a summary.
        """
        self._flush()

        with open(self._path, "a") as f:
            f.write("\n]\n")

        print(f"[Logger] ✅ Log closed: {self._path}")
        print(f"[Logger]    Frames logged : {self._frame_count}")
        print(f"[Logger]    Alerts fired  : {self._alert_count}")
        print(f"[Logger]    Log size      : "
              f"{os.path.getsize(self._path) / 1024:.1f} KB")

    # ── Convenience properties ─────────────────────────────
    @property
    def path(self) -> str:
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def alert_count(self) -> int:
        return self._alert_count


# ══════════════════════════════════════════════════════════
# Null logger — use when you want to disable logging
# ══════════════════════════════════════════════════════════

class NullLogger:
    """Drop-in replacement for RunLogger that does nothing."""

    def log(self, *args, **kwargs):
        pass

    def close(self):
        pass

    @property
    def path(self) -> str:
        return ""

    @property
    def frame_count(self) -> int:
        return 0

    @property
    def alert_count(self) -> int:
        return 0