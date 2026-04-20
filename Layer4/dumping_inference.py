"""
Layer 4 — Dumping Inference (Context-Aware Temporal Intent Evaluator)
======================================================================
Converts Layer 3 output → {"event": "legal_disposal"|"illegal_dumping", "confidence": float}

FIXES in this version:
  FIX 1: Bin distance uses BOTTOM-CENTER of bin bbox (not geometric center).
  FIX 2: BIN_NEAR_PX raised 150 → 200px.
  FIX 3: "Placed in bin" legal path — person standing next to bin counts as release.
  FIX 4: Fast-path legal disposal fires at release moment if trash is already near bin.
  FIX 5: MAX_WAIT_FRAMES raised 45 → 90.
  FIX 6 (NEW — root cause of silent failure):
         DumpingInference now tracks ALL non-person objects near a person, not just
         is_trash==True. TrashDetector's Pipeline A requires MIN_HELD_FRAMES(5) +
         RELEASE_CONFIRM(8) = 13 frames of separation before is_trash fires — which
         NEVER happens when someone walks straight to a bin and places an object
         (it stays near the person the whole time, separation never accumulates).
         This fix builds "candidate pairs" from any non-person object that has been
         near a person for CANDIDATE_HELD_MIN frames. is_trash==True objects are
         still preferred; candidates are the fallback that catches the silent case.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from Layer2.bin_tracker import TrackedBin
from Layer2.track_state import TrackedObject

# ──────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

MIN_POST_RELEASE     = 3     # frames of post-release motion required (relaxed for placement)
BIN_NEAR_PX          = 200   # px — object within this of bin bottom-center → legal
REST_VEL_THRESHOLD   = 3.0   # px/frame
REST_FRAMES          = 4     # consecutive rest frames to call final position
MAX_WAIT_FRAMES      = 90    # give up waiting for rest after this many post-release frames
MAX_PAIR_AGE         = 150   # frames — purge pair state if not updated
HELD_FRAMES_MIN      = 3     # object must have been near person this long before release counts

BIN_RELEASE_FAST_PX  = 220   # if trash is within this of a bin at release moment → legal
NEAR_PERSON_PX       = 150   # px — object within this of person centroid = "held"


# ──────────────────────────────────────────────────────────────────────────────
#  Output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DumpingEvent:
    event:      str    # "legal_disposal" | "illegal_dumping" | "pending"
    confidence: float
    pair_id:    str
    reason:     str = ""

    def to_dict(self) -> Dict:
        return {"event": self.event, "confidence": round(self.confidence, 3)}

    def __repr__(self):
        return f"DumpingEvent({self.event}, conf={self.confidence:.2f}, reason={self.reason})"


# ──────────────────────────────────────────────────────────────────────────────
#  Per-pair state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _PairState:
    pair_id:             str
    trash_track_id:      int
    person_track_id:     int

    release_confirmed:   bool  = False
    post_release_count:  int   = 0
    held_frames:         int   = 0
    release_pos: Optional[Tuple[float, float]] = None

    post_trail: deque = field(default_factory=lambda: deque(maxlen=60))
    consecutive_rest:    int   = 0

    event_triggered:     bool  = False
    locked_result: Optional[DumpingEvent] = None
    frames_since_update: int   = 0


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: np.ndarray) -> Tuple[float, float]:
    return (float((bbox[0] + bbox[2]) / 2.0), float((bbox[1] + bbox[3]) / 2.0))

def _bottom_center(bbox: np.ndarray) -> Tuple[float, float]:
    """FIX 1: bottom-center as bin reference (matches BinTracker's own logging)."""
    return (float((bbox[0] + bbox[2]) / 2.0), float(bbox[3]))

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest_bin(
    pt: Tuple[float, float],
    bins: List[TrackedBin],
) -> Tuple[float, Optional[TrackedBin]]:
    if not bins:
        return float("inf"), None
    best_d, best_b = float("inf"), None
    for tb in bins:
        d = _dist(pt, _bottom_center(tb.bbox))
        if d < best_d:
            best_d, best_b = d, tb
    return best_d, best_b

def _trail_velocity(trail: deque) -> float:
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    tail = pts[-4:] if len(pts) >= 4 else pts
    speeds = [
        math.hypot(tail[i][0] - tail[i-1][0], tail[i][1] - tail[i-1][1])
        for i in range(1, len(tail))
    ]
    return sum(speeds) / len(speeds)


# ──────────────────────────────────────────────────────────────────────────────
#  Main evaluator
# ──────────────────────────────────────────────────────────────────────────────

class DumpingInference:
    """
    Layer 4 — call update() once per frame.

    Usage:
        inference = DumpingInference()
        events = inference.update(tracked_objects, tracked_bins)
        for ev in events:
            if ev.event != "pending":
                print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f}")
    """

    def __init__(self):
        self._pairs: Dict[str, _PairState] = {}

    def update(
        self,
        tracked_objects: List[TrackedObject],
        tracked_bins:    List[TrackedBin],
    ) -> List[DumpingEvent]:

        for ps in self._pairs.values():
            ps.frames_since_update += 1

        persons = [o for o in tracked_objects if o.class_name == "person"]

        # ── FIX 6: Build candidate object list ───────────────────────────────
        # Priority 1: objects already confirmed as trash by Layer 1/2 (is_trash=True)
        # Priority 2: any non-person object near a person long enough — catches the
        #             "walk to bin and place" case where TrashDetector never fires
        #             because the object never separates from the person.
        confirmed_trash = [o for o in tracked_objects if o.is_trash]
        confirmed_ids   = {o.track_id for o in confirmed_trash}

        candidate_trash: List[TrackedObject] = []
        for obj in tracked_objects:
            if obj.class_name == "person":
                continue
            if obj.track_id in confirmed_ids:
                continue
            obj_c = _centroid(obj.bbox)
            for p in persons:
                if _dist(obj_c, _centroid(p.bbox)) <= NEAR_PERSON_PX:
                    candidate_trash.append(obj)
                    break

        all_trash = confirmed_trash + candidate_trash

        results: List[DumpingEvent] = []

        for tr in all_trash:
            person  = self._nearest_person(tr, persons)
            pid     = person.track_id if person else -1
            pair_id = f"person_{pid}_trash_{tr.track_id}"

            ps = self._get_or_create(pair_id, tr.track_id, pid)
            ps.frames_since_update = 0

            # Return locked result — never re-evaluate
            if ps.event_triggered and ps.locked_result is not None:
                results.append(ps.locked_result)
                continue

            # Update hold/release state
            self._update_release_state(ps, tr, person, tracked_bins)

            # Gate: release must be confirmed
            if not ps.release_confirmed:
                if tr.is_trash:
                    results.append(DumpingEvent("pending", 0.0, pair_id, "no_release_yet"))
                continue

            # ── FIX 4: Fast-path legal — already near a bin at release moment ─
            if ps.release_pos is not None and tracked_bins:
                fast_d, fast_bin = _nearest_bin(ps.release_pos, tracked_bins)
                if fast_d <= BIN_RELEASE_FAST_PX:
                    ev = DumpingEvent(
                        "legal_disposal", 0.90, pair_id,
                        f"released_near_bin dist={fast_d:.0f}px "
                        f"bin#{fast_bin.bin_id if fast_bin else '?'}"
                    )
                    ps.event_triggered = True
                    ps.locked_result   = ev
                    results.append(ev)
                    print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f} | {ev.reason}")
                    continue

            if ps.post_release_count < MIN_POST_RELEASE:
                results.append(DumpingEvent(
                    "pending", 0.0, pair_id,
                    f"post_release={ps.post_release_count}<{MIN_POST_RELEASE}"
                ))
                continue

            # Accumulate post-release trail
            ps.post_trail.append(_centroid(tr.bbox))

            # Rest detection
            vel = _trail_velocity(ps.post_trail)
            if vel < REST_VEL_THRESHOLD:
                ps.consecutive_rest += 1
            else:
                ps.consecutive_rest = 0

            object_at_rest = ps.consecutive_rest >= REST_FRAMES
            max_wait_hit   = ps.post_release_count >= MAX_WAIT_FRAMES

            if not (object_at_rest or max_wait_hit):
                results.append(DumpingEvent(
                    "pending", 0.0, pair_id, f"waiting_for_rest vel={vel:.1f}"
                ))
                continue

            # Final decision
            ev = self._decide(ps, tr, tracked_bins)
            ps.event_triggered = True
            ps.locked_result   = ev
            results.append(ev)
            print(f"[Layer4] Event: {ev.event} | Conf: {ev.confidence:.2f} | {ev.reason}")

        self._purge()
        return results

    # ── Release state updater ─────────────────────────────────────────────────

    def _update_release_state(
        self,
        ps:           _PairState,
        tr:           TrackedObject,
        person:       Optional[TrackedObject],
        tracked_bins: List[TrackedBin],
    ) -> None:
        """
        Three release signals in priority order:
          1. trash_how == "thrown"      — Layer 2 explicit signal
          2. Person standing at a bin   — FIX 3: placement / legal disposal path
          3. Person-trash divergence    — fallback throw detection
        Also handles: person leaves frame while holding object.
        """
        if ps.release_confirmed:
            ps.post_release_count += 1
            return

        trash_c = _centroid(tr.bbox)

        # Signal 1: explicit throw from Layer 2
        if tr.trash_how == "thrown":
            ps.release_confirmed  = True
            ps.post_release_count = 0
            ps.release_pos        = trash_c
            return

        # Signal 2: FIX 3 — person has reached a bin (placement path)
        if tracked_bins and person is not None:
            person_c   = _centroid(person.bbox)
            near_d, _  = _nearest_bin(person_c, tracked_bins)
            if near_d <= BIN_RELEASE_FAST_PX and ps.held_frames >= HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c
                return

        # Signal 3: divergence fallback
        if person is not None:
            d = _dist(trash_c, _centroid(person.bbox))
            if d > 120 and ps.held_frames >= HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c
                return
            if d <= NEAR_PERSON_PX:
                ps.held_frames += 1
        else:
            # Person left frame while object was being held — treat as release
            if ps.held_frames >= HELD_FRAMES_MIN:
                ps.release_confirmed  = True
                ps.post_release_count = 0
                ps.release_pos        = trash_c

    # ── Final decision ────────────────────────────────────────────────────────

    def _decide(
        self,
        ps:           _PairState,
        tr:           TrackedObject,
        tracked_bins: List[TrackedBin],
    ) -> DumpingEvent:
        bin_present = len(tracked_bins) > 0
        final_pos   = _centroid(tr.bbox)

        hold_score = min(ps.held_frames / 10.0, 1.0)
        post_score = min(ps.post_release_count / 20.0, 1.0)
        confidence = max(0.3, min(0.95, 0.5 * hold_score + 0.5 * post_score + 0.3))

        if bin_present:
            nearest_d, nearest_bin = _nearest_bin(final_pos, tracked_bins)
            bin_label = f"#{nearest_bin.bin_id}" if nearest_bin else "?"
            if nearest_d <= BIN_NEAR_PX:
                return DumpingEvent(
                    "legal_disposal", round(confidence, 2), ps.pair_id,
                    f"bin{bin_label} bottom_dist={nearest_d:.0f}px <= {BIN_NEAR_PX}px"
                )
            else:
                return DumpingEvent(
                    "illegal_dumping", round(confidence, 2), ps.pair_id,
                    f"bin{bin_label} present but bottom_dist={nearest_d:.0f}px > {BIN_NEAR_PX}px"
                )
        else:
            return DumpingEvent(
                "illegal_dumping", round(confidence, 2), ps.pair_id,
                f"no_bin held={ps.held_frames}f post={ps.post_release_count}f"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _nearest_person(
        self,
        tr:      TrackedObject,
        persons: List[TrackedObject],
    ) -> Optional[TrackedObject]:
        if not persons:
            return None
        tc = _centroid(tr.bbox)
        return min(persons, key=lambda p: _dist(tc, _centroid(p.bbox)))

    def _get_or_create(self, pair_id, trash_id, person_id) -> _PairState:
        if pair_id not in self._pairs:
            self._pairs[pair_id] = _PairState(
                pair_id         = pair_id,
                trash_track_id  = trash_id,
                person_track_id = person_id,
            )
        return self._pairs[pair_id]

    def _purge(self) -> None:
        stale = [
            pid for pid, ps in self._pairs.items()
            if ps.frames_since_update > MAX_PAIR_AGE and not ps.event_triggered
        ]
        for pid in stale:
            del self._pairs[pid]

    def get_active_pairs(self) -> List[_PairState]:
        return list(self._pairs.values())