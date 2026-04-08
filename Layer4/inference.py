# Layer4/inference.py
"""
Layer 4 — Inference Engine  (v2)

⚠️  Model now outputs RAW LOGITS (not probabilities).
    torch.sigmoid() is applied HERE before thresholding.

Usage:
    engine = DumpingInference(device="mps")
    engine.load()
    prob           = engine.predict(pair)
    is_viol, prob  = engine.is_violation(pair)
"""

import torch
from pathlib import Path
from typing import Tuple

from Layer3.pair_state import PairState
from .model  import DumpingClassifier
from .config import MODEL_SAVE_PATH, DUMP_THRESHOLD


class DumpingInference:

    def __init__(self, device: str = "cpu"):
        self.device  = torch.device(device)
        self.model   = DumpingClassifier().to(self.device)
        self.model.eval()
        self._loaded = False

    def load(self, path: str = MODEL_SAVE_PATH) -> bool:
        """Load trained weights. Returns True if loaded, False if file missing."""
        p = Path(path)
        if not p.exists():
            print(f"[Layer4] ⚠️  No weights at '{path}' — running UNTRAINED")
            return False
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        print(f"[Layer4] ✅ Loaded weights from '{path}'")
        return True

    def predict(self, pair: PairState) -> float:
        """
        Returns dumping probability [0, 1], or -1.0 if pair not ready.
        Applies sigmoid here since model outputs raw logits.
        """
        if not pair.ready():
            return -1.0

        seq = pair.get_sequence()
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()   # ← sigmoid applied HERE

        return float(prob)

    def is_violation(self, pair: PairState) -> Tuple[bool, float]:
        """Returns (is_violation: bool, probability: float)."""
        prob = self.predict(pair)
        if prob < 0:
            return False, 0.0
        return prob >= DUMP_THRESHOLD, prob