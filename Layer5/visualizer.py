"""
Layer 5 — Visualizer
Shows agent state, motion coupling, and evidence breakdown on frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from Layer5.agent import DumpingAgent

_COL_VIOLATION = (0,   50, 255)
_COL_LEGAL     = (50, 220,  50)
_COL_PENDING   = (0,  200, 255)
_COL_COUPLED   = (255, 200,   0)   # yellow = motion coupled
_COL_DIVERGING = (0,  140, 255)    # orange = release detected
_COL_BG        = (15,  15,  15)
_FONT          = cv2.FONT_HERSHEY_SIMPLEX

# State colour mapping for reasoning panel
_STATE_COLS = {
    "WATCHING":     (150, 150, 150),
    "POSSESSED":    (255, 200,   0),
    "DIVERGING":    (0,  140, 255),
    "RELEASED":     (200, 100, 255),
    "RESTING":      (100, 255, 200),
    "LOCKED":       (50,  220,  50),
}


def draw_l5_verdicts(frame: np.ndarray, results: List[dict]) -> np.ndarray:
    """Top-right: finalised verdict banners with evidence breakdown."""
    if not results:
        return frame
    H, W = frame.shape[:2]
    scale, thick, pad = 0.46, 1, 5

    for i, r in enumerate(results):
        col = _COL_VIOLATION if r["violation"] else _COL_LEGAL
        label = (
            f"[L5] P{r['person_id']} T{r['object_id']} | "
            f"{r['event'].upper()} | conf={r['confidence']:.2f} | "
            f"couple={r.get('coupling_frames','?')}f "
            f"cos={r.get('peak_coupling',0):.2f} | "
            f"intent={r.get('intent_score',0):.2f} | "
            f"f={r['frames'][0]}-{r['frames'][1]}"
        )
        (tw, th), _ = cv2.getTextSize(label, _FONT, scale, thick)
        x = W - tw - 15
        y = 40 + i * 28
        cv2.rectangle(frame, (x-pad, y-th-pad), (x+tw+pad, y+pad), _COL_BG, -1)
        cv2.putText(frame, label, (x, y), _FONT, scale, col, thick, cv2.LINE_AA)

    return frame


def draw_l5_reasoning(frame: np.ndarray, agent: "DumpingAgent") -> np.ndarray:
    """
    Centre-bottom panel: live agent state per active case.
    Shows state machine step + motion coupling evidence.
    """
    signals = agent.frame_signals
    if not signals:
        return frame

    H, W   = frame.shape[:2]
    scale  = 0.40
    thick  = 1
    pad    = 5
    line_h = 18

    lines = ["── L5 AGENT REASONING ──"] + [
        f"  {pid.split('_')[1]}→{pid.split('_')[3]}: {msg}"
        for pid, msg in signals.items()
    ]

    if not lines:
        return frame

    box_w = max(
        cv2.getTextSize(l, _FONT, scale, thick)[0][0] for l in lines
    ) + pad * 2
    box_h  = len(lines) * line_h + pad * 2
    x0     = (W - box_w) // 2
    y0     = H - box_h - 10

    cv2.rectangle(frame, (x0, y0), (x0+box_w, y0+box_h), _COL_BG, -1)
    cv2.rectangle(frame, (x0, y0), (x0+box_w, y0+box_h), (60, 60, 60), 1)

    for j, line in enumerate(lines):
        # Colour by state keyword
        col = _COL_PENDING
        if j > 0:
            for state_name, state_col in _STATE_COLS.items():
                if state_name in line:
                    col = state_col
                    break
            else:
                col = (180, 180, 180)

        cv2.putText(
            frame, line,
            (x0+pad, y0+pad+(j+1)*line_h),
            _FONT, scale, col, thick, cv2.LINE_AA,
        )

    return frame


def draw_l5_evidence_bars(frame: np.ndarray, results: List[dict]) -> np.ndarray:
    """
    For each locked result, draw a small evidence bar chart
    showing coupling / release / rest / intent scores.
    Only shown for the most recent result.
    """
    if not results:
        return frame

    r     = results[-1]   # most recent
    H, W  = frame.shape[:2]
    x0    = W - 200
    y0    = 70 + len(results) * 28

    bars = [
        ("Coupling",  r.get("peak_coupling",  0.0), _COL_COUPLED),
        ("Release",   r.get("release_clarity",0.0), _COL_DIVERGING),
        ("Rest",      min(r.get("rest_frames",0) / 5.0, 1.0), (100,255,200)),
        ("Intent",    r.get("intent_score",   0.0), (200,200, 50)),
    ]

    bar_w = 80
    bar_h = 8
    pad   = 3

    cv2.rectangle(frame,
                  (x0 - 10, y0 - 5),
                  (x0 + 170, y0 + len(bars) * (bar_h + pad + 12) + 5),
                  _COL_BG, -1)

    for i, (label, val, col) in enumerate(bars):
        y = y0 + i * (bar_h + pad + 12)
        cv2.putText(frame, label, (x0, y + bar_h),
                    _FONT, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
        # Background bar
        cv2.rectangle(frame, (x0 + 60, y), (x0 + 60 + bar_w, y + bar_h), (50,50,50), -1)
        # Filled bar
        fill = int(bar_w * max(0.0, min(val, 1.0)))
        if fill > 0:
            cv2.rectangle(frame, (x0 + 60, y), (x0 + 60 + fill, y + bar_h), col, -1)
        cv2.putText(frame, f"{val:.2f}", (x0 + 60 + bar_w + 4, y + bar_h),
                    _FONT, 0.32, (180,180,180), 1, cv2.LINE_AA)

    return frame


def draw_l5_summary_box(frame: np.ndarray, all_results: List[dict]) -> np.ndarray:
    """Bottom-left: event tally."""
    violations = sum(1 for r in all_results if r["violation"])
    legal      = len(all_results) - violations
    lines = [
        f"[L5] Events: {len(all_results)}",
        f"  Violations: {violations}",
        f"  Legal:      {legal}",
    ]
    H, W  = frame.shape[:2]
    scale, thick, pad = 0.46, 1, 5
    box_h = len(lines) * 20 + pad * 2
    x0, y0 = 10, H - box_h - 10
    cv2.rectangle(frame, (x0, y0), (x0+220, y0+box_h), _COL_BG, -1)
    for j, line in enumerate(lines):
        col = _COL_PENDING if j == 0 else (
            _COL_VIOLATION if ("Violations" in line and violations > 0)
            else (180, 180, 180)
        )
        cv2.putText(frame, line, (x0+pad, y0+pad+(j+1)*18),
                    _FONT, scale, col, thick, cv2.LINE_AA)
    return frame