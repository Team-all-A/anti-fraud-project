from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def apply_threshold(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_proba, dtype=float) >= threshold).astype(int)


def evaluate_thresholds(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pl.DataFrame:
    """Таблиця метрик для кожного порогу. Зручно для аналізу після запуску."""
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    rows = []
    for thr in thresholds:
        y_pred = apply_threshold(y_proba, thr)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        rows.append({
            "threshold": round(float(thr), 4),
            "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
    return pl.DataFrame(rows)


def optimize_threshold(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float, dict]:
    """
    Знаходить поріг з максимальним F1 на OOF-ймовірностях.
    Повертає (best_threshold, best_f1, metrics_dict).
    """
    df = evaluate_thresholds(y_true, y_proba)
    best_row = df.filter(pl.col("f1") == df["f1"].max()).row(0, named=True)

    best_f1 = best_row["f1"]

    return (
        best_row["threshold"],
        best_f1,
        {
            "tp": best_row["tp"], "tn": best_row["tn"],
            "fp": best_row["fp"], "fn": best_row["fn"],
            "precision": best_row["precision"],
            "recall":    best_row["recall"],
            "f1":        best_f1,
        },
    )