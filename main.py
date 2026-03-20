from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.business import apply_threshold, optimize_threshold
from src.model import get_model
from src.pipeline import build_pipeline
from src.preprocessing import load_and_merge


TRAIN_TRANSACTIONS_PATH = Path("data/train_transactions.csv")
TRAIN_USERS_PATH = Path("data/train_users.csv")
TEST_TRANSACTIONS_PATH = Path("data/test_transactions.csv")
TEST_USERS_PATH = Path("data/test_users.csv")

TARGET_COL = "is_fraud"
ID_COL = "id_user"

N_SPLITS = 5
RANDOM_STATE = 42

COST_FP = 50.0
COST_FN = 500.0

SUBMISSION_PATH = Path("submission.csv")


print("Loading and merging datasets...")

train_df = load_and_merge(str(TRAIN_TRANSACTIONS_PATH), str(TRAIN_USERS_PATH),)
test_df = load_and_merge(str(TEST_TRANSACTIONS_PATH), str(TEST_USERS_PATH),)
print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

y = train_df.get_column(TARGET_COL).to_numpy()
X = train_df.drop(TARGET_COL)
X_test = test_df

skf = StratifiedKFold( n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE,)

oof_proba = np.zeros(X.height, dtype=float)
test_proba = np.zeros(X_test.height, dtype=float)
fold_f1_scores: list[float] = []

print(f"\nStarting {N_SPLITS}-fold Stratified Cross-Validation...")

for fold_number, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(len(y)), y),
    start=1,
):
    print(f"\n--- Fold {fold_number}/{N_SPLITS} ---")

    X_train_fold = (
        X.with_row_index("__row_nr")
        .filter(pl.col("__row_nr").is_in(train_idx.tolist()))
        .drop("__row_nr")
    )
    X_val_fold = (
        X.with_row_index("__row_nr")
        .filter(pl.col("__row_nr").is_in(val_idx.tolist()))
        .drop("__row_nr")
    )

    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]

    model = get_model(y_train_fold)
    pipeline = build_pipeline(model=model)

    pipeline.fit(X_train_fold, y_train_fold)

    val_proba = pipeline.predict_proba(X_val_fold)[:, 1]
    test_fold_proba = pipeline.predict_proba(X_test)[:, 1]

    oof_proba[val_idx] = val_proba
    test_proba += test_fold_proba / N_SPLITS

    val_pred_default = apply_threshold(val_proba, 0.5)
    fold_f1 = f1_score(y_val_fold, val_pred_default, zero_division=0)
    fold_f1_scores.append(fold_f1)

    pred_fraud_count = int((val_proba >= 0.5).sum())
    true_fraud_count = int(y_val_fold.sum())

    print(f"Fold F1 @ 0.50: {fold_f1:.4f}")
    print(
        f"Validation positives: true={true_fraud_count}, predicted={pred_fraud_count}, "
        f"proba min/mean/max={val_proba.min():.4f}/{val_proba.mean():.4f}/{val_proba.max():.4f}"
    )

print("\nCross-validation summary")
print(f"Mean fold F1 @ 0.50: {np.mean(fold_f1_scores):.4f}")
print(f"Std fold F1  @ 0.50: {np.std(fold_f1_scores):.4f}")

print("\nOptimizing business threshold...")

optimal_threshold, min_cost, metrics = optimize_threshold( y_true=y, y_proba=oof_proba, cost_fp=COST_FP, cost_fn=COST_FN,)

print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Minimum modeled cost: ${min_cost:,.2f}")
print(
    f"Confusion matrix: TN={metrics['tn']}, FP={metrics['fp']}, "
    f"FN={metrics['fn']}, TP={metrics['tp']}"
)

if "precision" in metrics and "recall" in metrics and "f1" in metrics:
    print(
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}"
    )

final_predictions = apply_threshold(test_proba, optimal_threshold)

submission = pl.DataFrame(
    {
        ID_COL: test_df.get_column(ID_COL),
        TARGET_COL: final_predictions,
    }
)

submission.write_csv(SUBMISSION_PATH)
print(f"\nSaved submission to: {SUBMISSION_PATH}")