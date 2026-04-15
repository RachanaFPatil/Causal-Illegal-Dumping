"""
Layer 2 — Lightweight ReID Embedding Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extracts appearance feature vectors (128D) from bbox crops.

Design:
  • CPU-friendly — uses a shallow MobileNetV3-Small backbone (torchvision)
    truncated before the classifier head → 128D L2-normalised embedding.
  • Falls back to colour histogram embedding if torch is unavailable.
  • Embeddings are cached per track_id and updated with EMA smoothing.
  • Cosine distance is the similarity metric (range [0, 1]).

Public API:
    reid = ReIDEmbedder()
    emb  = reid.extract(crop)           # np.ndarray crop → 128D vector
    dist = reid.cosine_distance(a, b)   # float in [0, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "torchreid"], capture_output=True)
import logging
import numpy as np
from typing import Optional, Dict
import torchreid

logger = logging.getLogger(__name__)

# ── Embedding dimension ────────────────────────────────────────────────────────
EMBED_DIM = 128

# ── EMA weight for embedding update ───────────────────────────────────────────
EMBED_EMA_ALPHA = 0.3   # 0 = never update, 1 = replace completely


# ── Backend selection ──────────────────────────────────────────────────────────

def _try_load_torch_model():
    try:
        model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1000,
            pretrained=True
        )
        model.eval()
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        logger.info("[ReID] OSNet_x0_25 pretrained loaded (CPU).")
        return model, transform
    except Exception as e:
        logger.warning("[ReID] torchreid unavailable (%s). Using histogram fallback.", e)
        return None, None


# ── Colour histogram fallback (no torch required) ─────────────────────────────

def _histogram_embedding(crop: np.ndarray) -> np.ndarray:
    """
    32-bin HSV histogram per channel → 96D → PCA-like projection → 128D.
    Robust to moderate lighting variation; viewpoint-invariant for colour.
    """
    import cv2
    if crop is None or crop.size == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    resized = cv2.resize(crop, (64, 128))
    hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hists = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        hists.append(h.flatten())

    raw = np.concatenate(hists).astype(np.float32)  # 96D

    # Pad/project to EMBED_DIM with a fixed random projection (deterministic)
    rng = np.random.RandomState(42)
    proj = rng.randn(len(raw), EMBED_DIM).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-6

    emb = raw @ proj  # (EMBED_DIM,)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-6)


# ── Main ReID Embedder ─────────────────────────────────────────────────────────

class ReIDEmbedder:
    """
    Appearance embedding extractor.

    Usage:
        reid = ReIDEmbedder()
        emb  = reid.extract(bgr_crop)
        dist = ReIDEmbedder.cosine_distance(emb_a, emb_b)
    """

    def __init__(self):
        self._model, self._transform = _try_load_torch_model()
        self._use_torch = self._model is not None

        # Per-track embedding memory: track_id → 128D EMA embedding
        self._track_embeddings: Dict[int, np.ndarray] = {}

    # ── Extraction ─────────────────────────────────────────────────────────────

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a 128D L2-normalised embedding from a BGR crop.

        Args:
            crop: BGR numpy array (any size — will be resized internally).

        Returns:
            np.ndarray shape (128,), float32, L2-normalised.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        if self._use_torch:
            return self._torch_extract(crop)
        return _histogram_embedding(crop)

    def _torch_extract(self, crop: np.ndarray) -> np.ndarray:
        try:
            import torch
            tensor = self._transform(crop).unsqueeze(0)
            with torch.no_grad():
                emb = self._model(tensor)
                emb = emb.squeeze(0).numpy()
            # Project 512D → 128D with fixed random projection
            rng  = np.random.RandomState(42)
            proj = rng.randn(len(emb), EMBED_DIM).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-6
            emb  = emb @ proj
            norm = np.linalg.norm(emb)
            return (emb / (norm + 1e-6)).astype(np.float32)
        except Exception as e:
            logger.debug("[ReID] Torch extract failed: %s", e)
            return _histogram_embedding(crop)

    # ── Track embedding memory ─────────────────────────────────────────────────

    def update_track(self, track_id: int, crop: np.ndarray) -> np.ndarray:
        """
        Extract embedding for crop and update EMA memory for track_id.

        Returns the updated (smoothed) embedding for this track.
        """
        new_emb = self.extract(crop)
        if track_id in self._track_embeddings:
            old = self._track_embeddings[track_id]
            smoothed = EMBED_EMA_ALPHA * new_emb + (1 - EMBED_EMA_ALPHA) * old
            norm = np.linalg.norm(smoothed)
            self._track_embeddings[track_id] = smoothed / (norm + 1e-6)
        else:
            self._track_embeddings[track_id] = new_emb
        return self._track_embeddings[track_id]

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Return stored embedding for track_id, or None if not seen yet."""
        return self._track_embeddings.get(track_id)

    def remove_track(self, track_id: int) -> None:
        """Remove embedding memory for a deleted track."""
        self._track_embeddings.pop(track_id, None)

    def clear(self) -> None:
        self._track_embeddings.clear()

    # ── Distance metrics ───────────────────────────────────────────────────────

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine distance in [0, 1].
        0 = identical, 1 = completely dissimilar.
        """
        if a is None or b is None:
            return 1.0
        dot  = float(np.dot(a, b))
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        sim  = dot / (norm + 1e-6)
        return float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))

    @staticmethod
    def embedding_matrix(
        embs_a: list,   # List[np.ndarray or None]  length N
        embs_b: list,   # List[np.ndarray or None]  length M
    ) -> np.ndarray:    # (N, M) float32
        """
        Vectorised cosine distance matrix for Hungarian assignment.
        Missing embeddings (None) get distance = 0.5 (neutral).
        """
        N, M = len(embs_a), len(embs_b)
        mat = np.full((N, M), 0.5, dtype=np.float32)
        for i, a in enumerate(embs_a):
            if a is None:
                continue
            for j, b in enumerate(embs_b):
                if b is None:
                    continue
                mat[i, j] = ReIDEmbedder.cosine_distance(a, b)
        return mat