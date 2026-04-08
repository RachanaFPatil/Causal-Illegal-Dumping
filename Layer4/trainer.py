# Layer4/trainer.py
"""
Layer 4 — Trainer  (v2)

Key improvements over v1:
  - Stratified train/val split  (critical with only 21 dump samples)
  - BCEWithLogitsLoss + pos_weight computed FROM DATA automatically
  - Proper metrics: Precision, Recall, F1 (not just accuracy)
  - 80 epochs with early stopping on F1 (not val loss)
  - Model outputs logits — sigmoid applied only at inference

Usage:
    python -m Layer4.trainer
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from .model   import DumpingClassifier, LabelSmoothingBCEWithLogitsLoss
from .dataset import DumpingDataset, stratified_split
from .config  import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS, DATA_DIR,
    MODEL_SAVE_PATH, DUMP_THRESHOLD,
)

PATIENCE = 12   # epochs without F1 improvement before stopping


# ─────────────────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    threshold: float = DUMP_THRESHOLD) -> dict:
    """
    Compute precision, recall, F1, accuracy from raw logits + true labels.

    Why not just accuracy:
        A model that always predicts "normal" gets 84% accuracy on this dataset
        but catches zero dumps (recall=0). F1 exposes this failure mode.

    threshold: DUMP_THRESHOLD from config (default 0.4 — lower to favour recall).
    """
    probs  = torch.sigmoid(logits)
    preds  = (probs >= threshold).long()
    labels = labels.long()

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return dict(precision=precision, recall=recall, f1=f1,
                accuracy=accuracy, tp=tp, fp=fp, fn=fn, tn=tn)


# ─────────────────────────────────────────────────────────
def train():
    # ── Device ────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Trainer] Device: {device}\n")

    # ── Class counts → pos_weight ─────────────────────────
    # Computed from actual data — not hardcoded in config.
    # pos_weight = n_normal / n_dump
    # Tells the loss: missing a dump costs (pos_weight)x more than a false alarm.
    n_normal, n_dump = DumpingDataset.class_counts(DATA_DIR)
    if n_dump == 0:
        raise ValueError("No dump sequences in data dir — collect some first.")
    if n_normal == 0:
        raise ValueError("No normal sequences in data dir — collect some first.")

    pw = n_normal / n_dump
    print(f"[Trainer] Class counts: {n_normal} normal / {n_dump} dump")
    print(f"[Trainer] pos_weight   = {pw:.2f}  (auto-computed)\n")

    # ── Stratified split ──────────────────────────────────
    train_indices, val_indices = stratified_split(DATA_DIR)

    train_ds = DumpingDataset(DATA_DIR, augment=True,  indices=train_indices)
    val_ds   = DumpingDataset(DATA_DIR, augment=False, indices=val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    n_train, n_val = len(train_ds), len(val_ds)

    # ── Model ─────────────────────────────────────────────
    model = DumpingClassifier().to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Trainer] Parameters: {total:,}")

    # ── Loss ──────────────────────────────────────────────
    pos_weight = torch.tensor([pw], dtype=torch.float32).to(device)
    criterion  = LabelSmoothingBCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # ── Optimizer + Scheduler ─────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────
    best_val_f1      = 0.0
    best_val_loss    = float("inf")
    patience_counter = 0

    header = (f"{'Ep':>4} | {'TrLoss':>7} {'TrF1':>6} {'TrAcc':>6} | "
              f"{'VaLoss':>7} {'VaF1':>6} {'Prec':>6} {'Rec':>6} | "
              f"TP FP FN TN")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────
        model.train()
        t_loss = 0.0
        t_logits, t_labels = [], []

        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(seqs)
            t_logits.append(logits.detach().cpu())
            t_labels.append(labels.detach().cpu())

        scheduler.step()
        t_loss /= n_train
        tm = compute_metrics(torch.cat(t_logits), torch.cat(t_labels))

        # ── Validate ───────────────────────────────────────
        model.eval()
        v_loss = 0.0
        v_logits, v_labels = [], []

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss   = criterion(logits, labels)
                v_loss += loss.item() * len(seqs)
                v_logits.append(logits.cpu())
                v_labels.append(labels.cpu())

        v_loss /= n_val
        vm = compute_metrics(torch.cat(v_logits), torch.cat(v_labels))

        print(f"{epoch:4d} | {t_loss:7.4f} {tm['f1']:6.3f} {tm['accuracy']:6.3f} | "
              f"{v_loss:7.4f} {vm['f1']:6.3f} {vm['precision']:6.3f} {vm['recall']:6.3f} | "
              f"{vm['tp']} {vm['fp']} {vm['fn']} {vm['tn']}")

        # ── Save best model — tracked by F1, not val loss ──
        # Reason: with imbalance, val loss can decrease while recall→0.
        # F1 directly measures whether we catch dumps (recall) without too many alarms (precision).
        improved = vm['f1'] > best_val_f1 or (
            abs(vm['f1'] - best_val_f1) < 1e-6 and v_loss < best_val_loss
        )
        if improved:
            best_val_f1      = vm['f1']
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"       ✅ Saved  F1={vm['f1']:.3f}  P={vm['precision']:.3f}  R={vm['recall']:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Trainer] ⏹  Early stop at epoch {epoch} "
                      f"(F1 flat for {PATIENCE} epochs)")
                break

    # ── Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Trainer] Done.")
    print(f"  Best val F1     : {best_val_f1:.3f}")
    print(f"  Weights saved   : {MODEL_SAVE_PATH}")
    print(f"\n  📊 Data summary : {n_normal} normal / {n_dump} dump")
    print(f"  💡 If F1 < 0.5  : collect more dump sequences (target: 40–50+)")
    print(f"  💡 Current ratio: 1 dump per {n_normal/n_dump:.1f} normal — ideal is 1:3 or better")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()