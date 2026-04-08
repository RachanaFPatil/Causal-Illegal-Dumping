# Layer4/dataset.py
"""
Layer 4 — Dataset  (v2)

Key improvements over v1:
  - stratified_split() — ensures same dump/normal ratio in train and val
  - Stronger augmentation: temporal crop, time-reverse, per-feature scale jitter
  - augment=True for TRAIN only — val always raw (augment=False)
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from Layer3.pair_state import PairState
from .config import DATA_DIR, TRAIN_SPLIT


# ══════════════════════════════════════════════════════════
# 1. COLLECTION
# ══════════════════════════════════════════════════════════

class SequenceCollector:
    """
    Saves labeled sequences to DATA_DIR as .npz files.
        label=1 → illegal dumping
        label=0 → normal behaviour
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._saved = 0

    def collect(self, pair: PairState, label: int) -> str:
        assert label in (0, 1), "label must be 0 or 1"
        if not pair.ready():
            print(f"[Collector] ⚠️  Pair {pair.pair_key} not ready — skipped")
            return ""

        seq  = pair.get_sequence()
        tag  = "dump" if label == 1 else "normal"
        ts   = int(time.time() * 1000)
        name = f"{tag}_p{pair.person_id}_o{pair.object_id}_{ts}.npz"
        path = self.data_dir / name

        np.savez_compressed(
            path,
            sequence = seq,
            label    = np.array(label, dtype=np.int64),
        )
        self._saved += 1
        print(f"[Collector] Saved {tag.upper()} → {path}  (total: {self._saved})")
        return str(path)

    @property
    def saved_count(self) -> int:
        return self._saved


# ══════════════════════════════════════════════════════════
# 2. STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════

def stratified_split(data_dir: str = DATA_DIR,
                     train_split: float = TRAIN_SPLIT
                     ) -> Tuple[List[int], List[int]]:
    """
    Returns (train_indices, val_indices) with the SAME dump/normal ratio
    in both splits.

    Why this matters:
        With only 21 dump samples, a random split can put 0–3 dumps in val
        by chance. That makes validation F1 meaningless (0.0 with zero positives).
        Stratified split guarantees proportional representation.

    Fixed seed=42 for reproducibility — same split every run.
    """
    files   = sorted(Path(data_dir).glob("*.npz"))
    pos_idx = [i for i, f in enumerate(files) if np.load(f)["label"].item() == 1]
    neg_idx = [i for i, f in enumerate(files) if np.load(f)["label"].item() == 0]

    rng = np.random.default_rng(42)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_train = max(1, int(len(pos_idx) * train_split))
    n_neg_train = max(1, int(len(neg_idx) * train_split))

    train_idx = pos_idx[:n_pos_train] + neg_idx[:n_neg_train]
    val_idx   = pos_idx[n_pos_train:] + neg_idx[n_neg_train:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    n_pos_val = len(pos_idx) - n_pos_train
    n_neg_val = len(neg_idx) - n_neg_train
    print(f"[Split] Train: {len(train_idx)} ({n_pos_train} dump / {n_neg_train} normal) | "
          f"Val: {len(val_idx)} ({n_pos_val} dump / {n_neg_val} normal)")

    return train_idx, val_idx


# ══════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════

class DumpingDataset(Dataset):
    """
    Loads .npz sequences from DATA_DIR.

    augment=True  → training augmentations applied (TRAIN split only)
    augment=False → raw sequences (VAL/test only)
    indices       → list of file indices to use (from stratified_split)

    Augmentations (augment=True):
      1. Gaussian noise (σ=0.015) — simulates sensor jitter
      2. Per-feature scale jitter ±8% — simulates distance/camera variation
      3. Temporal crop (shift 0–4 frames) — different clip start points
      4. Time-reverse (50% chance) — teaches model both directions of interaction
    """

    def __init__(self, data_dir: str = DATA_DIR, augment: bool = False,
                 indices: List[int] = None):
        self.data_dir = Path(data_dir)
        self.augment  = augment
        all_files     = sorted(self.data_dir.glob("*.npz"))

        if not all_files:
            raise FileNotFoundError(
                f"No .npz files in '{data_dir}'.\n"
                "Run: python run_pipeline.py --source video.mp4 --collect"
            )

        self.files = [all_files[i] for i in indices] if indices is not None else all_files

        labels = [np.load(f)["label"].item() for f in self.files]
        n_pos  = int(sum(labels))
        n_neg  = len(labels) - n_pos
        print(f"[Dataset] {len(self.files)} seqs — {n_pos} dump / {n_neg} normal  "
              f"({'TRAIN+AUG' if augment else 'VAL/raw'})")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        seq  = torch.tensor(data["sequence"], dtype=torch.float32)   # (T, 9)
        lbl  = torch.tensor(float(data["label"]), dtype=torch.float32)

        if self.augment:
            seq = self._augment(seq)

        return seq, lbl

    def _augment(self, seq: torch.Tensor) -> torch.Tensor:
        # 1. Gaussian noise
        seq = seq + torch.randn_like(seq) * 0.015

        # 2. Per-feature scale jitter ±8%  — shape (1, 9)
        scale = 0.92 + torch.rand(1, seq.shape[1]) * 0.16
        seq   = seq * scale

        # 3. Temporal crop — drop 0–4 frames from start, pad end by repeating last
        shift = torch.randint(0, 5, (1,)).item()
        if shift > 0:
            seq = torch.cat([seq[shift:], seq[-1:].expand(shift, -1)], dim=0)

        # 4. Time reverse (50%)
        if torch.rand(1).item() < 0.5:
            seq = seq.flip(0)

        # Clamp — noise/scale can push features slightly out of range
        seq = torch.clamp(seq, -1.5, 1.5)

        return seq

    @staticmethod
    def class_counts(data_dir: str = DATA_DIR) -> Tuple[int, int]:
        """Returns (n_normal, n_dump) for computing pos_weight in trainer."""
        files = sorted(Path(data_dir).glob("*.npz"))
        n_pos = int(sum(np.load(f)["label"].item() for f in files))
        return len(files) - n_pos, n_pos