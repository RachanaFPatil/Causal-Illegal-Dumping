# Layer4/evaluate.py
"""
Evaluation Script — Layer 4

Reads stored run_*.json log files and computes:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - False positive rate
    - False negative rate
    - Confusion matrix
    - Per-type breakdown (vehicle_dumping, pedestrian_dumping, anomaly)
    - Score distribution stats

Ground truth:
    Ground truth labels are taken from the "final_decision.alert" field.
    This means evaluation is SELF-SUPERVISED unless you edit the logs
    to add a "ground_truth" field manually or programmatically.

    To add real ground truth:
        In each log entry, add:  "ground_truth": 0 or 1
        This script will use that field when present.

Usage:
    python -m Layer4.evaluate                         # all logs in logs/
    python -m Layer4.evaluate --log_dir logs/ --threshold 0.45
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ══════════════════════════════════════════════════════════
# Loader
# ══════════════════════════════════════════════════════════

def load_log_entries(log_dir: str = "logs/") -> List[Dict]:
    """Load all entries from all run_*.json files in log_dir."""
    entries = []
    files   = sorted(Path(log_dir).glob("run_*.json"))

    if not files:
        print(f"[Evaluate] No log files found in '{log_dir}'")
        return []

    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)
        except Exception as e:
            print(f"[Evaluate] ⚠️  Could not load {fpath}: {e}")

    print(f"[Evaluate] Loaded {len(entries)} entries from {len(files)} log files")
    return entries


# ══════════════════════════════════════════════════════════
# Ground truth extraction
# ══════════════════════════════════════════════════════════

def extract_labels_and_scores(
    entries:   List[Dict],
    threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract:
        y_true  — ground truth (0/1)
        y_pred  — predicted label (0/1) after threshold
        scores  — final_score float
        types   — dump type strings

    Ground truth priority:
        1. entry["ground_truth"]  (manually annotated)
        2. entry["final_decision"]["alert"]  (model self-label — for dev purposes)
    """
    y_true, y_pred, scores, types = [], [], [], []

    for entry in entries:
        decision = entry.get("final_decision", {})
        if not decision:
            continue

        final_score = decision.get("final_score", decision.get("confidence", 0.5))
        dump_type   = decision.get("type", "none")

        # Ground truth
        if "ground_truth" in entry:
            gt = int(entry["ground_truth"])
        else:
            # Fall back to model's own alert — useful for self-eval during dev
            gt = 1 if decision.get("alert", False) else 0

        pred = 1 if float(final_score) >= threshold else 0

        y_true.append(gt)
        y_pred.append(pred)
        scores.append(float(final_score))
        types.append(dump_type)

    return (
        np.array(y_true, dtype=int),
        np.array(y_pred, dtype=int),
        np.array(scores, dtype=float),
        types,
    )


# ══════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    fpr       = fp / (fp + tn + 1e-8)   # false positive rate
    fnr       = fn / (fn + tp + 1e-8)   # false negative rate

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       4),
        "fnr":       round(fnr,       4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def score_distribution(scores: np.ndarray) -> Dict:
    if len(scores) == 0:
        return {}
    return {
        "mean":   round(float(scores.mean()),   4),
        "std":    round(float(scores.std()),    4),
        "min":    round(float(scores.min()),    4),
        "max":    round(float(scores.max()),    4),
        "p25":    round(float(np.percentile(scores, 25)), 4),
        "p50":    round(float(np.percentile(scores, 50)), 4),
        "p75":    round(float(np.percentile(scores, 75)), 4),
    }


def per_type_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    types:  List[str],
) -> Dict:
    unique_types = set(types)
    breakdown    = {}
    for t in sorted(unique_types):
        mask = np.array([tp == t for tp in types])
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        m["count"] = int(mask.sum())
        breakdown[t] = m
    return breakdown


# ══════════════════════════════════════════════════════════
# Confusion matrix printer
# ══════════════════════════════════════════════════════════

def print_confusion_matrix(tp: int, fp: int, fn: int, tn: int):
    print("\n  Confusion Matrix:")
    print(f"  {'':15s}  Pred_DUMP  Pred_NORMAL")
    print(f"  {'True_DUMP':15s}  {tp:9d}  {fn:11d}")
    print(f"  {'True_NORMAL':15s}  {fp:9d}  {tn:11d}")


# ══════════════════════════════════════════════════════════
# Main evaluation report
# ══════════════════════════════════════════════════════════

def evaluate(
    log_dir:   str   = "logs/",
    threshold: float = 0.45,
) -> Dict:
    """
    Full evaluation pipeline.  Prints report and returns metrics dict.
    """
    print(f"\n{'='*60}")
    print("  ILLEGAL DUMPING DETECTION — EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Log dir   : {log_dir}")
    print(f"  Threshold : {threshold}")
    print(f"{'='*60}\n")

    entries = load_log_entries(log_dir)
    if not entries:
        print("[Evaluate] No data to evaluate.")
        return {}

    y_true, y_pred, scores, types = extract_labels_and_scores(entries, threshold)

    if len(y_true) == 0:
        print("[Evaluate] No valid entries with decisions found.")
        return {}

    # ── Overall metrics ────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)

    print("  OVERALL METRICS")
    print(f"  {'Accuracy':12s}: {metrics['accuracy']:.4f}")
    print(f"  {'Precision':12s}: {metrics['precision']:.4f}")
    print(f"  {'Recall':12s}: {metrics['recall']:.4f}")
    print(f"  {'F1':12s}: {metrics['f1']:.4f}")
    print(f"  {'FPR':12s}: {metrics['fpr']:.4f}  (false positive rate)")
    print(f"  {'FNR':12s}: {metrics['fnr']:.4f}  (false negative rate)")

    print_confusion_matrix(
        metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    )

    # ── Score distribution ─────────────────────────────────
    dist = score_distribution(scores)
    print(f"\n  SCORE DISTRIBUTION (final_score)")
    print(f"  {'Mean':8s}: {dist.get('mean', 0):.4f}")
    print(f"  {'Std':8s}: {dist.get('std',  0):.4f}")
    print(f"  {'Min':8s}: {dist.get('min',  0):.4f}")
    print(f"  {'Max':8s}: {dist.get('max',  0):.4f}")
    print(f"  {'p25':8s}: {dist.get('p25',  0):.4f}")
    print(f"  {'p50':8s}: {dist.get('p50',  0):.4f}")
    print(f"  {'p75':8s}: {dist.get('p75',  0):.4f}")

    # ── Per type ──────────────────────────────────────────
    breakdown = per_type_breakdown(y_true, y_pred, types)
    if breakdown:
        print(f"\n  PER-TYPE BREAKDOWN")
        for dtype, dm in breakdown.items():
            print(f"\n  [{dtype}]  n={dm['count']}")
            print(f"    Precision={dm['precision']:.3f}  "
                  f"Recall={dm['recall']:.3f}  "
                  f"F1={dm['f1']:.3f}")

    # ── Class balance ─────────────────────────────────────
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    print(f"\n  CLASS BALANCE")
    print(f"  Dump frames   : {n_pos}")
    print(f"  Normal frames : {n_neg}")
    if n_pos > 0:
        ratio = n_neg / n_pos
        print(f"  Ratio         : 1 dump per {ratio:.1f} normal")

    # ── Recommendations ───────────────────────────────────
    print(f"\n  RECOMMENDATIONS")
    if metrics["recall"] < 0.6:
        print("  ⚠️  Recall < 0.6 — model is MISSING DUMPS. "
              "Lower alert threshold or collect more dump data.")
    if metrics["fpr"] > 0.3:
        print("  ⚠️  FPR > 0.3 — too many FALSE ALARMS. "
              "Raise alert threshold or tighten rule thresholds.")
    if metrics["f1"] >= 0.7:
        print("  ✅ F1 >= 0.7 — model is performing well.")
    if n_pos < 20:
        print("  💡 Only {n_pos} dump frames — collect more for reliable evaluation.")

    print(f"\n{'='*60}\n")

    return {
        "overall":      metrics,
        "distribution": dist,
        "breakdown":    breakdown,
        "n_entries":    len(entries),
        "n_positive":   n_pos,
        "n_negative":   n_neg,
    }


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate illegal dumping detection logs")
    parser.add_argument("--log_dir",   default="logs/",  help="Directory with run_*.json files")
    parser.add_argument("--threshold", default=0.45, type=float, help="Alert threshold")
    args = parser.parse_args()

    evaluate(log_dir=args.log_dir, threshold=args.threshold)