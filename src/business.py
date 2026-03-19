from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total_cost: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    f1: float


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert probabilities to binary predictions using a chosen threshold.
    """
    y_proba = np.asarray(y_proba, dtype=float)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    return (y_proba >= threshold).astype(np.int8)


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float = 50.0,
    cost_fn: float = 500.0,
) -> ThresholdResult:
    """
    Evaluate a single threshold using both business cost and ML metrics.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba must have the same length, got "
            f"{y_true.shape[0]} and {y_proba.shape[0]}"
        )

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("y_proba must contain probabilities in [0, 1]")

    y_pred = apply_threshold(y_proba, threshold)

    # Fixed labels guarantee a stable 2x2 confusion matrix.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total_cost = (fp * cost_fp) + (fn * cost_fn)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return ThresholdResult(
        threshold=float(threshold),
        total_cost=float(total_cost),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 50.0,
    cost_fn: float = 500.0,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, dict]:
    """
    Find the threshold that minimizes business cost.

    Tie-breakers:
    1. lower total cost
    2. higher F1
    3. lower threshold (slightly more fraud-sensitive when otherwise equal)

    Returns:
        best_threshold
        min_cost
        best_metrics dict
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba must have the same length, got "
            f"{y_true.shape[0]} and {y_proba.shape[0]}"
        )

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_result: ThresholdResult | None = None

    for threshold in thresholds:
        result = evaluate_threshold(
            y_true=y_true,
            y_proba=y_proba,
            threshold=float(threshold),
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if best_result is None:
            best_result = result
            continue

        is_better = (
            (result.total_cost < best_result.total_cost)
            or (
                result.total_cost == best_result.total_cost
                and result.f1 > best_result.f1
            )
            or (
                result.total_cost == best_result.total_cost
                and result.f1 == best_result.f1
                and result.threshold < best_result.threshold
            )
        )

        if is_better:
            best_result = result

    assert best_result is not None

    best_metrics = {
        "tn": best_result.tn,
        "fp": best_result.fp,
        "fn": best_result.fn,
        "tp": best_result.tp,
        "precision": best_result.precision,
        "recall": best_result.recall,
        "f1": best_result.f1,
    }

    return best_result.threshold, best_result.total_cost, best_metrics