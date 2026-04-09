# Layer4/ml_model.py
"""
ML Scoring Model — lightweight MLP classifier.

Predicts dumping probability from a structured feature dict built from
Layer 3 PairState data.  Runs in PARALLEL with the existing transformer
(DumpingClassifier) — does NOT replace it.

Input features (8 scalars):
    distance          — current centroid distance (pixels, raw)
    iou               — overlap between person & object bbox
    velocity          — object centroid speed (pixels/frame)
    interaction_time  — total frames pair has been observed
    stationary_time   — frames object has been stationary
    object_area       — object bbox area (pixels²)
    vehicle_present   — 1 if a vehicle-class track is nearby, else 0
    event_count       — held_frames + released_frames from PairState

Output:
    {"probability": float [0,1], "confidence": float [0,1]}

Training:
    Offline, using stored JSON logs (written by Layer4/logger.py).
    Run: python -m Layer4.ml_model  (trains and saves to Layer4/weights/ml_model.pkl)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- optional sklearn (graceful fallback) ----------
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing  import StandardScaler
    from sklearn.pipeline       import Pipeline
    from sklearn.metrics        import classification_report
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ---------- fallback pure-numpy logistic regression -------
# If sklearn is unavailable, we use a minimal numpy LR so the
# system still runs end-to-end without extra deps.

ML_MODEL_PATH = "Layer4/weights/ml_model.pkl"
FEATURE_KEYS  = [
    "distance",
    "iou",
    "velocity",
    "interaction_time",
    "stationary_time",
    "object_area",
    "vehicle_present",
    "event_count",
]


# ══════════════════════════════════════════════════════════
# Geometry helpers (mirror Layer3 where needed)
# ══════════════════════════════════════════════════════════

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def build_feature_vector(feature_dict: Dict) -> np.ndarray:
    """Convert structured dict → ordered numpy feature vector."""
    return np.array([feature_dict.get(k, 0.0) for k in FEATURE_KEYS],
                    dtype=np.float32)


# ══════════════════════════════════════════════════════════
# Pure-numpy fallback logistic regression
# ══════════════════════════════════════════════════════════

class _NumpyLogisticRegression:
    """Minimal LR trained with gradient descent. No sklearn needed."""

    def __init__(self, lr: float = 0.01, epochs: int = 500):
        self.lr     = lr
        self.epochs = epochs
        self.w      = None
        self.b      = 0.0
        self.mu     = None
        self.sigma  = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Normalise
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xn = (X - self.mu) / self.sigma

        n, d   = Xn.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            z    = Xn @ self.w + self.b
            p    = self._sigmoid(z)
            err  = p - y
            self.w -= self.lr * (Xn.T @ err) / n
            self.b -= self.lr * err.mean()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mu) / self.sigma
        return self._sigmoid(Xn @ self.w + self.b)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "_NumpyLogisticRegression":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════

class MLScoringModel:
    """
    Wraps either sklearn MLP pipeline or numpy LR (fallback).

    Usage:
        model = MLScoringModel()
        model.load()                   # loads weights if available
        result = model.predict(feat_dict)
        # → {"probability": 0.82, "confidence": 0.91}

    Training (offline):
        model.train_from_logs("logs/")
        model.save()
    """

    def __init__(self):
        self._model  = None
        self._loaded = False

    # ── Prediction ────────────────────────────────────────
    def predict(self, feature_dict: Dict) -> Dict[str, float]:
        """
        Returns {"probability": float, "confidence": float}.
        If no model loaded, returns neutral 0.5.
        """
        if self._model is None:
            return {"probability": 0.5, "confidence": 0.0}

        x    = build_feature_vector(feature_dict).reshape(1, -1)
        prob = float(self._predict_proba(x)[0])

        # Confidence = distance from decision boundary (0.5)
        confidence = float(abs(prob - 0.5) * 2.0)

        return {"probability": prob, "confidence": confidence}

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if _SKLEARN_OK and isinstance(self._model, Pipeline):
            return self._model.predict_proba(X)[:, 1]
        elif isinstance(self._model, _NumpyLogisticRegression):
            return self._model.predict_proba(X)
        return np.array([0.5])

    # ── Load / Save ───────────────────────────────────────
    def load(self, path: str = ML_MODEL_PATH) -> bool:
        p = Path(path)
        if not p.exists():
            print(f"[MLModel] No weights at '{path}' — running uninitialised (prob=0.5)")
            return False
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._loaded = True
        print(f"[MLModel] ✅ Loaded from '{path}'")
        return True

    def save(self, path: str = ML_MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        print(f"[MLModel] 💾 Saved to '{path}'")

    # ── Training ──────────────────────────────────────────
    def train_from_logs(self, log_dir: str = "logs/") -> Dict:
        """
        Loads all run_*.json log files, extracts features + labels,
        trains the model.

        Label:  1 → final_decision["alert"] == True
                0 → no alert

        Returns: classification report dict.
        """
        X, y = self._load_training_data(log_dir)

        if len(X) == 0:
            print("[MLModel] ⚠️  No training data found in logs.")
            return {}

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"[MLModel] Training on {len(X)} samples  "
              f"({n_pos} dump / {n_neg} normal)")

        if _SKLEARN_OK:
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes = (32, 16),
                    activation         = "relu",
                    max_iter           = 500,
                    random_state       = 42,
                    early_stopping     = True,
                    validation_fraction= 0.15,
                    n_iter_no_change   = 20,
                )),
            ])
            self._model.fit(X, y)
            preds  = self._model.predict(X)
            report = classification_report(y, preds, output_dict=True)
        else:
            # Fallback: numpy LR
            self._model = _NumpyLogisticRegression()
            self._model.fit(X, y)
            probs  = self._model.predict_proba(X)
            preds  = (probs >= 0.5).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            prec   = tp / (tp + fp + 1e-8)
            rec    = tp / (tp + fn + 1e-8)
            f1     = 2 * prec * rec / (prec + rec + 1e-8)
            report = {"precision": prec, "recall": rec, "f1": f1}

        print(f"[MLModel] Train report: {report}")
        return report

    def _load_training_data(self, log_dir: str
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse JSON log files → (X, y) arrays."""
        log_path = Path(log_dir)
        files    = sorted(log_path.glob("run_*.json"))

        X_rows, y_rows = [], []

        for fpath in files:
            try:
                with open(fpath) as f:
                    entries = json.load(f)
            except Exception:
                continue

            if not isinstance(entries, list):
                entries = [entries]

            for entry in entries:
                feat = entry.get("features", {})
                dec  = entry.get("final_decision", {})
                if not feat or not dec:
                    continue

                x = build_feature_vector(feat)
                label = 1 if dec.get("alert", False) else 0
                X_rows.append(x)
                y_rows.append(label)

        if not X_rows:
            return np.array([]), np.array([])

        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


# ══════════════════════════════════════════════════════════
# CLI — python -m Layer4.ml_model
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ML scoring model from logs")
    parser.add_argument("--log_dir", default="logs/", help="Directory with run_*.json logs")
    parser.add_argument("--save",    default=ML_MODEL_PATH, help="Where to save model")
    args = parser.parse_args()

    model = MLScoringModel()
    report = model.train_from_logs(args.log_dir)
    if report:
        model.save(args.save)
        print("\n✅ ML model trained and saved.")
    else:
        print("\n❌ Training failed — collect more log data first.")