from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


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
    best_f1  = best_row["f1"]
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


def export_threshold_report(
    y_true:            np.ndarray,
    oof_proba:         np.ndarray,
    optimal_threshold: float,
    oof_f1:            float,
    oof_auc:           float,
    n_train_total:     int,
    n_test_total:      int,
    predicted_fraud_rate_test: float,
    output_path:       str | Path = "threshold_report.csv",
    n_steps:           int = 200,
) -> pl.DataFrame:
    """
    Генерує CSV-звіт для бізнес-аналітика.

    Містить:
      1. Повний sweep метрик по порогах (для побудови кривих)
      2. Окремий рядок з оптимальним порогом (позначено is_optimal=1)

    Виклик ПІСЛЯ того як модель видала OOF probabilities — незалежно від submission.

    Колонки:
      threshold, f1, precision, recall, accuracy,
      tp, tn, fp, fn,
      fp_rate (FP / total_legit),
      fn_rate (FN / total_fraud),
      predicted_fraud_pct (скільки % від усіх відмічено як fraud),
      is_optimal (1 для вибраного порогу)
    """
    y_true  = np.asarray(y_true)
    oof_proba = np.asarray(oof_proba, dtype=float)

    total_fraud = int(y_true.sum())
    total_legit = int(len(y_true) - total_fraud)

    # Sweep по рівномірній сітці в діапазоні реальних proba
    p_lo = float(np.percentile(oof_proba, 0.5))
    p_hi = float(np.percentile(oof_proba, 99.5))
    thresholds = np.linspace(p_lo, p_hi, n_steps)

    # Додаємо оптимальний поріг в список щоб він точно був у таблиці
    thresholds = np.sort(np.unique(np.append(thresholds, optimal_threshold)))

    rows = []
    for thr in thresholds:
        y_pred = apply_threshold(oof_proba, float(thr))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        n_predicted_fraud = tp + fp

        rows.append({
            "threshold":           round(float(thr), 6),
            "f1":                  round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
            "precision":           round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall":              round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "accuracy":            round(float(accuracy_score(y_true, y_pred)), 6),
            "tp":                  tp,
            "tn":                  tn,
            "fp":                  fp,
            "fn":                  fn,
            "fp_rate":             round(fp / max(total_legit, 1), 6),
            "fn_rate":             round(fn / max(total_fraud, 1), 6),
            "predicted_fraud_pct": round(100.0 * n_predicted_fraud / max(len(y_true), 1), 4),
            "is_optimal":          1 if abs(float(thr) - optimal_threshold) < 1e-9 else 0,
        })

    df = pl.DataFrame(rows)

    # ── Зведений рядок з глобальними метриками ────────────────────────────
    summary_path = Path(str(output_path).replace(".csv", "_summary.csv"))
    summary = pl.DataFrame([{
        "generated_at":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "optimal_threshold":         round(optimal_threshold, 6),
        "oof_f1":                    round(oof_f1, 6),
        "oof_auc":                   round(oof_auc, 6),
        "total_fraud_train":         total_fraud,
        "total_legit_train":         total_legit,
        "fraud_rate_train":          round(total_fraud / max(len(y_true), 1), 6),
        "n_train_total":             n_train_total,
        "n_test_total":              n_test_total,
        "predicted_fraud_rate_test": round(predicted_fraud_rate_test, 6),
        "predicted_fraud_count_test": round(predicted_fraud_rate_test * n_test_total),
    }])

    df.write_csv(output_path)
    summary.write_csv(summary_path)

    print(f"\n[BUSINESS REPORT] Threshold sweep  → {output_path}")
    print(f"[BUSINESS REPORT] Summary metrics  → {summary_path}")
    print(f"  Optimal threshold : {optimal_threshold:.6f}")
    print(f"  OOF F1 / AUC      : {oof_f1:.4f} / {oof_auc:.4f}")
    print(f"  Train fraud rate  : {total_fraud / max(len(y_true),1):.4f}  ({total_fraud:,} / {total_legit:,})")
    print(f"  Test predicted    : {predicted_fraud_rate_test:.4f}  (~{round(predicted_fraud_rate_test * n_test_total):,} users)")

    return df