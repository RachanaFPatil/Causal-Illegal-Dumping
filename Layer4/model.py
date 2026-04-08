# Layer4/model.py
"""
Transformer-based binary classifier for illegal dumping detection.

Input:  (batch, T, FEATURE_DIM)
Output: (batch,) — RAW LOGITS (not probabilities)

⚠️  IMPORTANT: This model outputs logits, NOT sigmoid probabilities.
    - Trainer uses BCEWithLogitsLoss (numerically stable, accepts logits)
    - Inference applies torch.sigmoid() before thresholding
    - This is standard PyTorch practice for binary classification

Architecture:
  1. Linear projection: FEATURE_DIM → D_MODEL
  2. Learnable positional encoding
  3. TransformerEncoder with DropPath (stochastic depth) per block
  4. Last time-step → classification head → logit (no sigmoid here)

Regularization:
  - Dropout      : inside attention + FFN + classifier head
  - DropPath     : stochastic depth — drops entire residual per sample per layer
  - Label smooth : applied in trainer via LabelSmoothingBCEWithLogitsLoss
"""

import torch
import torch.nn as nn
from .config import (
    SEQUENCE_LENGTH, FEATURE_DIM,
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD,
    DROPOUT, DROPPATH_RATE, LABEL_SMOOTHING,
)


# ══════════════════════════════════════════════════════════
# 1. DROPPATH
# ══════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """
    Stochastic Depth: drops entire residual path per sample during training.
    At inference time, acts as identity.
    Much stronger than Dropout for transformers on small datasets.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob     = 1 - self.drop_prob
        random_tensor = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


# ══════════════════════════════════════════════════════════
# 2. TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer encoder block with DropPath on both residuals.
    Pre-norm is more stable than post-norm, especially with small data.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, drop_path_rate: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = nhead,
            dropout     = dropout,
            batch_first = True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════
# 3. LABEL SMOOTHING LOSS (works with logits directly)
# ══════════════════════════════════════════════════════════

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss + label smoothing + pos_weight.

    Smoothing targets:
        positive (dump):   1.0 → (1 - smoothing)   e.g. 0.95
        negative (normal): 0.0 → smoothing          e.g. 0.05

    pos_weight: scales loss for positive class — KEY FIX for 21 dump / 118 normal imbalance.
    BCEWithLogitsLoss is numerically stable (uses log-sum-exp internally).
    """

    def __init__(self, pos_weight: torch.Tensor, smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.bce       = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return self.bce(logits, targets_smooth)


# ══════════════════════════════════════════════════════════
# 4. MAIN MODEL
# ══════════════════════════════════════════════════════════

class DumpingClassifier(nn.Module):
    """
    Full model. Outputs RAW LOGITS — apply torch.sigmoid() externally at inference.

    DropPath rates linearly scaled across layers:
        layer 0 → 0.0       (no drop)
        layer N → DROPPATH_RATE   (max drop)
    Standard stochastic depth schedule (Huang et al. 2016).
    """

    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(FEATURE_DIM, D_MODEL)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, SEQUENCE_LENGTH, D_MODEL)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.input_dropout = nn.Dropout(DROPOUT)

        drop_path_rates = [
            DROPPATH_RATE * (i / max(NUM_LAYERS - 1, 1))
            for i in range(NUM_LAYERS)
        ]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model         = D_MODEL,
                nhead           = NHEAD,
                dim_feedforward = DIM_FEEDFORWARD,
                dropout         = DROPOUT,
                drop_path_rate  = drop_path_rates[i],
            )
            for i in range(NUM_LAYERS)
        ])

        # ← NO Sigmoid() here — BCEWithLogitsLoss handles it in training
        # ← inference.py applies torch.sigmoid() before thresholding
        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, FEATURE_DIM)
        Returns:
            logits: (B,) — raw logits, NOT probabilities
        """
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = x[:, -1, :]                        # last time step = most recent context
        return self.classifier(x).squeeze(-1)  # (B,) logits