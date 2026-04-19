"""
Layer 3 — Memory Manager
========================
Maintains a rolling time-window of all tracked objects and bins.
Provides the feature extractor with a consistent historical view.

This is intentionally lightweight — it stores references, not copies,
to stay memory efficient in real-time pipelines.
"""

from collections import deque
from typing import List, Dict, Deque, Tuple
from Layer2.track_state import TrackedObject
from Layer2.bin_tracker import TrackedBin

from .config import WINDOW_SIZE


class MemoryManager:
    """
    Sliding window memory for all tracked objects and bins.

    Usage:
        memory = MemoryManager()
        memory.update(tracked_objects, tracked_bins, frame_number)
        recent_objects = memory.get_recent_objects(n=10)
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._window_size = window_size
        # Each entry: (frame_number, List[TrackedObject])
        self._object_history: Deque[Tuple[int, List[TrackedObject]]] = deque(maxlen=window_size)
        # Each entry: (frame_number, List[TrackedBin])
        self._bin_history:    Deque[Tuple[int, List[TrackedBin]]]    = deque(maxlen=window_size)

        # Fast lookup: track_id → last seen TrackedObject
        self._last_seen_object: Dict[int, TrackedObject] = {}
        # Fast lookup: bin_id → last seen TrackedBin
        self._last_seen_bin:    Dict[int, TrackedBin]    = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        tracked_objects: List[TrackedObject],
        tracked_bins:    List[TrackedBin],
        frame_number:    int,
    ) -> None:
        """Store this frame's tracking output in the sliding window."""
        self._object_history.append((frame_number, tracked_objects))
        self._bin_history.append((frame_number, tracked_bins))

        # Update fast-lookup dicts
        for obj in tracked_objects:
            self._last_seen_object[obj.track_id] = obj
        for tb in tracked_bins:
            self._last_seen_bin[tb.bin_id] = tb

    def get_recent_objects(self, n: int = WINDOW_SIZE) -> List[TrackedObject]:
        """Return all unique tracked objects seen in the last N frames (deduped by track_id)."""
        seen_ids = set()
        result   = []
        for _, objs in reversed(list(self._object_history)[-n:]):
            for obj in objs:
                if obj.track_id not in seen_ids:
                    seen_ids.add(obj.track_id)
                    result.append(obj)
        return result

    def get_recent_bins(self, n: int = WINDOW_SIZE) -> List[TrackedBin]:
        """Return all unique bins seen in the last N frames (deduped by bin_id)."""
        seen_ids = set()
        result   = []
        for _, bins in reversed(list(self._bin_history)[-n:]):
            for tb in bins:
                if tb.bin_id not in seen_ids:
                    seen_ids.add(tb.bin_id)
                    result.append(tb)
        return result

    def last_seen_object(self, track_id: int) -> TrackedObject | None:
        return self._last_seen_object.get(track_id)

    def last_seen_bin(self, bin_id: int) -> TrackedBin | None:
        return self._last_seen_bin.get(bin_id)

    @property
    def frame_count(self) -> int:
        return len(self._object_history)