import numpy as np
import polars as pl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def apply_threshold(y_proba, threshold=0.5):
    y_proba = np.asarray(y_proba, dtype=float)
    return (y_proba >= threshold).astype(int)


def evaluate_thresholds(y_true, y_proba, thresholds=None, cost_fp=50.0, cost_fn=500.0):
    """
    Рахує метрики моделі для різних threshold.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    rows = []

    for threshold in thresholds:
        y_pred = apply_threshold(y_proba, threshold)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        total_cost = fp * cost_fp + fn * cost_fn

        rows.append({
            "threshold": round(float(threshold), 4),
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total_cost": round(float(total_cost), 2)
        })

    return pl.DataFrame(rows)