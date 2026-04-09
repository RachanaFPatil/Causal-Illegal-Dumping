# Layer4/visualizer.py
"""
Layer 4 — Visualizer  (v2 — Hybrid Decision Overlay)

Draws model predictions on top of the Layer 3 visualisation.

New in v2:
    - Shows all three score components (transformer / ML / rule)
    - Colour-coded by dump type
    - Reasons panel for the top-scoring pair
    - Hybrid alert banner includes type and score breakdown
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from Layer3.pair_state import PairState
from .config import HYBRID_THRESHOLD, DUMP_THRESHOLD

# Colours
SAFE_COLOR    = (0, 200,   0)
WARN_COLOR    = (0, 165, 255)
VIOL_COLOR    = (0,   0, 220)
BAR_BG_COLOR  = (40,  40,  40)
PANEL_BG      = (20,  20,  20)
WHITE         = (255, 255, 255)
CYAN          = (0,  255, 255)

# Alert threshold (use hybrid threshold)
THRESHOLD = HYBRID_THRESHOLD


def _prob_color(score: float) -> Tuple[int, int, int]:
    if score >= THRESHOLD:
        return VIOL_COLOR
    if score >= THRESHOLD * 0.75:
        return WARN_COLOR
    return SAFE_COLOR


def _clamp_bar(value: float, bar_w: int) -> int:
    return int(np.clip(value, 0.0, 1.0) * bar_w)


# ══════════════════════════════════════════════════════════
# Main draw function
# ══════════════════════════════════════════════════════════

def draw_predictions(
    frame:       np.ndarray,
    pairs:       List[PairState],
    predictions: Dict[tuple, Dict],
) -> np.ndarray:
    """
    Draw hybrid decision overlay.

    Args:
        frame:       BGR frame (already annotated by Layers 1–3)
        pairs:       active pairs from MemoryEngine
        predictions: {pair_key: decision_dict}  from DumpingInference.run_all_pairs()
    """
    H, W = frame.shape[:2]
    keys = list(predictions.keys())

    top_alert:  Optional[Dict] = None   # highest-confidence alert for banner

    for i, pair in enumerate(pairs):
        key      = pair.pair_key
        decision = predictions.get(key)
        if decision is None:
            continue

        final_score       = decision.get("final_score",       decision.get("confidence", 0.0))
        transformer_score = decision.get("transformer_score", 0.0)
        ml_score          = decision.get("ml_score",          0.0)
        rule_score        = decision.get("rule_score",        0.0)
        is_alert          = decision.get("alert",             False)
        dump_type         = decision.get("type",              "none")

        color = _prob_color(final_score)

        # ── Score bar panel (left side) ─────────────────────
        bx   = 10
        by   = 55 + i * 68
        bw   = 180
        rh   = 10    # row height for sub-bars

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (bx - 3, by - 3),
                      (bx + bw + 80, by + 60),
                      PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Header label
        header = f"P{pair.person_id}→O{pair.object_id}  {final_score:.0%}"
        cv2.putText(frame, header,
                    (bx, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Main bar (final score)
        cv2.rectangle(frame, (bx, by + 16), (bx + bw, by + 16 + rh), BAR_BG_COLOR, -1)
        cv2.rectangle(frame, (bx, by + 16),
                      (bx + _clamp_bar(final_score, bw), by + 16 + rh), color, -1)

        # Sub-bars: transformer | ML | rule
        sub_labels = [
            ("T", transformer_score, (100, 200, 255)),
            ("M", ml_score,          (200, 255, 100)),
            ("R", rule_score,        (255, 180,  50)),
        ]
        for si, (lbl, val, sc) in enumerate(sub_labels):
            sx = bx + si * 62
            sy = by + 32
            cv2.rectangle(frame, (sx, sy), (sx + 55, sy + 7), BAR_BG_COLOR, -1)
            cv2.rectangle(frame, (sx, sy),
                          (sx + _clamp_bar(val, 55), sy + 7), sc, -1)
            cv2.putText(frame, f"{lbl}:{val:.2f}",
                        (sx, sy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, sc, 1, cv2.LINE_AA)

        # Track top alert
        if is_alert:
            if top_alert is None or final_score > top_alert.get("final_score", 0):
                top_alert = {**decision, "pair_key": key}

    # ── Alert banner ───────────────────────────────────────
    if top_alert:
        _draw_alert_banner(frame, H, W, top_alert)

    # ── Reasons panel for top alert ────────────────────────
    if top_alert:
        _draw_reasons_panel(frame, W, top_alert)

    return frame


def _draw_alert_banner(frame: np.ndarray, H: int, W: int, decision: Dict):
    """Red border + bottom banner for active alert."""
    # Red border
    cv2.rectangle(frame, (0, 0), (W, H), VIOL_COLOR, 4)

    dump_type   = decision.get("type", "dumping")
    pair_key    = decision.get("pair_key", ("?", "?"))
    final_score = decision.get("final_score", 0.0)

    # Banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - 38), (W, H), (0, 0, 100), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text = (f"🚨 {dump_type.upper().replace('_',' ')} — "
            f"Person {pair_key[0]} / Object {pair_key[1]} — "
            f"Score {final_score:.0%}")
    cv2.putText(frame, text,
                (10, H - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, WHITE, 2, cv2.LINE_AA)


def _draw_reasons_panel(frame: np.ndarray, W: int, decision: Dict):
    """Small reasons panel in top-right corner."""
    reasons = decision.get("reasons", [])
    if not reasons:
        return

    px    = W - 230
    py    = 50
    lh    = 16
    panel_h = len(reasons) * lh + 28

    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 5, py - 18), (W - 5, py + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "ALERT REASONS",
                (px, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1, cv2.LINE_AA)

    for i, reason in enumerate(reasons[:8]):   # cap at 8 lines
        cv2.putText(frame, f"• {reason}",
                    (px, py + i * lh + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 180), 1, cv2.LINE_AA)