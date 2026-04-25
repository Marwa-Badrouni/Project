"""
utils/metrics.py — Evaluation metrics for attack detection.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve,
)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JSD_THRESHOLD, RESULTS_DIR


def evaluate(y_true: np.ndarray,
             anomaly_scores: np.ndarray,
             threshold: float = None) -> dict:
    """
    Full evaluation of the detector.

    Parameters
    ----------
    y_true         : binary ground-truth (0=benign, 1=attack)
    anomaly_scores : continuous anomaly score per sample
    threshold      : decision threshold on scores; if None, uses JSD_THRESHOLD

    Returns
    -------
    metrics dict
    """
    threshold = threshold or JSD_THRESHOLD
    y_pred = (anomaly_scores >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, anomaly_scores)
    auc_pr  = average_precision_score(y_true, anomaly_scores)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    report = classification_report(y_true, y_pred,
                                   target_names=["Benign", "Attack"],
                                   output_dict=True, zero_division=0)

    # Optimal F1 threshold (for reference)
    precisions, recalls, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx       = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])

    fpr_arr, tpr_arr, _ = roc_curve(y_true, anomaly_scores)

    metrics = {
        "threshold_used":    threshold,
        "auc_roc":           round(auc_roc, 4),
        "auc_pr":            round(auc_pr, 4),
        "f1":                round(f1, 4),
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "fpr":               round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix":  cm.tolist(),
        "classification_report": report,
        "best_f1":           round(best_f1, 4),
        "best_threshold":    round(best_threshold, 4),
        # Arrays for plotting
        "_roc": {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()},
        "_pr":  {"precision": precisions.tolist(), "recall": recalls.tolist()},
    }

    _print_summary(metrics)
    return metrics


def _print_summary(m):
    print("\n" + "="*50)
    print("  DETECTION RESULTS")
    print("="*50)
    print(f"  AUC-ROC   : {m['auc_roc']:.4f}")
    print(f"  AUC-PR    : {m['auc_pr']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}  (threshold={m['threshold_used']:.3f})")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  FPR       : {m['fpr']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"  Best F1   : {m['best_f1']:.4f}  (at threshold={m['best_threshold']:.3f})")
    print("="*50 + "\n")


def save_metrics(metrics: dict, name: str = "metrics.json"):
    # Remove plotting arrays before saving (too large for JSON)
    save_dict = {k: v for k, v in metrics.items() if not k.startswith("_")}
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"[Metrics] Saved → {path}")
