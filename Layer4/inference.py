# Layer4/inference.py
"""
Layer 4 — Inference Engine  (v3 — Hybrid Pipeline)

Wraps three scoring sources:
    1. DumpingClassifier (transformer)  — existing model, unchanged
    2. MLScoringModel   (MLP/LR)        — new: trained from log data
    3. HybridAgent                      — new: blends all three + rules

Public API (backward compatible):
    engine = DumpingInference(device="cpu")
    engine.load()

    # Original API — still works
    prob           = engine.predict(pair)
    is_viol, prob  = engine.is_violation(pair)

    # New hybrid API
    decision = engine.hybrid_decide(pair, tracks, ml_model)
    # → full decision dict from HybridAgent

⚠️  Model outputs RAW LOGITS — torch.sigmoid() applied here.
"""

import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Layer3.pair_state  import PairState
from Layer2.track_state import TrackedObject

from .model    import DumpingClassifier
from .agent    import HybridAgent, build_feature_dict, evaluate_rules
from .ml_model import MLScoringModel
from .config   import (
    MODEL_SAVE_PATH, DUMP_THRESHOLD,
    HYBRID_ALPHA, HYBRID_BETA, HYBRID_GAMMA, HYBRID_THRESHOLD,
    ENABLE_HYBRID, ENABLE_ML_MODEL,
)


class DumpingInference:
    """
    Main inference engine for Layer 4.

    Initialise once at startup, then call per-frame.
    """

    def __init__(self, device: str = "cpu"):
        self.device  = torch.device(device)
        self.model   = DumpingClassifier().to(self.device)
        self.model.eval()
        self._loaded = False

        # Hybrid components
        self._agent    = HybridAgent(
            alpha     = HYBRID_ALPHA,
            beta      = HYBRID_BETA,
            gamma     = HYBRID_GAMMA,
            threshold = HYBRID_THRESHOLD,
        )
        self._ml_model: Optional[MLScoringModel] = None

    # ── Weight loading ────────────────────────────────────
    def load(self, path: str = MODEL_SAVE_PATH) -> bool:
        """Load transformer weights. Returns True if successful."""
        p = Path(path)
        if not p.exists():
            print(f"[Layer4] ⚠️  No transformer weights at '{path}' — running UNTRAINED")
            return False
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        print(f"[Layer4] ✅ Transformer loaded from '{path}'")
        return True

    def load_ml_model(self, ml_model: MLScoringModel):
        """Attach a pre-loaded MLScoringModel."""
        self._ml_model = ml_model

    # ── Original API (backward compatible) ───────────────
    def predict(self, pair: PairState) -> float:
        """
        Transformer-only probability [0,1], or -1.0 if pair not ready.
        Preserved for backward compatibility.
        """
        if not pair.ready():
            return -1.0

        seq = pair.get_sequence()
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()

        return float(prob)

    def is_violation(self, pair: PairState) -> Tuple[bool, float]:
        """Returns (is_violation, probability). Original API."""
        prob = self.predict(pair)
        if prob < 0:
            return False, 0.0
        return prob >= DUMP_THRESHOLD, prob

    # ── New hybrid API ────────────────────────────────────
    def hybrid_decide(
        self,
        pair:   PairState,
        tracks: List[TrackedObject],
    ) -> Dict:
        """
        Full hybrid decision for one pair.

        Steps:
            1. Transformer probability (existing model)
            2. ML model prediction      (new)
            3. Rule-based score        (new)
            4. Blend → final decision  (new)

        Returns: decision dict from HybridAgent.decide()
        """
        if not ENABLE_HYBRID:
            # Legacy mode: just wrap transformer output
            prob = self.predict(pair)
            alert = prob >= DUMP_THRESHOLD
            return {
                "alert":             alert,
                "confidence":        max(prob, 0.0),
                "type":              "pedestrian_dumping" if alert else "none",
                "transformer_score": max(prob, 0.0),
                "ml_score":          0.5,
                "rule_score":        0.0,
                "final_score":       max(prob, 0.0),
                "reasons":           [],
                "pair_key":          pair.pair_key,
                "feature_dict":      {},
            }

        # 1. Transformer score
        transformer_prob = self.predict(pair)

        # 2. ML model score
        if ENABLE_ML_MODEL and self._ml_model is not None:
            feature_dict = build_feature_dict(pair, tracks)
            ml_output    = self._ml_model.predict(feature_dict)
        else:
            feature_dict = build_feature_dict(pair, tracks)
            ml_output    = {"probability": 0.5, "confidence": 0.0}

        # 3 + 4. Agent blends everything
        decision = self._agent.decide(
            pair             = pair,
            tracks           = tracks,
            transformer_prob = transformer_prob,
            ml_output        = ml_output,
        )

        return decision

    def run_all_pairs(
        self,
        pairs:  List[PairState],
        tracks: List[TrackedObject],
        infer_every_n: int = 1,
        frame_idx:     int = 0,
    ) -> Dict[tuple, Dict]:
        """
        Run hybrid decision on all ready pairs.

        Args:
            pairs:         all active pairs from MemoryEngine
            tracks:        all TrackedObject from Layer 2
            infer_every_n: only run transformer every N frames (performance)
            frame_idx:     current frame number (for throttling)

        Returns:
            {pair_key: decision_dict}
        """
        results = {}
        run_transformer = (frame_idx % max(infer_every_n, 1) == 0)

        for pair in pairs:
            if not pair.ready():
                continue

            if run_transformer:
                decision = self.hybrid_decide(pair, tracks)
            else:
                # Skip expensive transformer — use cached or neutral
                feature_dict = build_feature_dict(pair, tracks)
                rule_score, reasons = evaluate_rules(pair, tracks, feature_dict)
                ml_output = (
                    self._ml_model.predict(feature_dict)
                    if (ENABLE_ML_MODEL and self._ml_model is not None)
                    else {"probability": 0.5, "confidence": 0.0}
                )
                decision = self._agent.combine(
                    ml_output   = ml_output,
                    rule_output = (rule_score, reasons),
                    transformer_prob = -1.0,   # not computed this frame
                )
                decision["pair_key"]      = pair.pair_key
                decision["feature_dict"]  = feature_dict

            results[pair.pair_key] = decision

        return results