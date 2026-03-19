from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import classification_report, f1_score

from src.flat_features import align_to_features, build_features, to_model_matrix
from src.model import (
    FraudLGBMClassifier,
    default_params,
    run_optuna,
    suggest_scale_pos_weight,
)


DATA_DIR = Path("data")
TRAIN_USERS = DATA_DIR / "train_users.csv"
TRAIN_TRX = DATA_DIR / "train_transactions.csv"
TEST_USERS = DATA_DIR / "test_users.csv"
TEST_TRX = DATA_DIR / "test_transactions.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
USE_OPTUNA = False
N_TRIALS = 30


PRESET_PARAMS: dict | None = None


def load_raw_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    print("Loading raw datasets...")
    train_users = pl.read_csv(TRAIN_USERS, infer_schema_length=10_000)
    train_trx = pl.read_csv(TRAIN_TRX, infer_schema_length=10_000)
    test_users = pl.read_csv(TEST_USERS, infer_schema_length=10_000)
    test_trx = pl.read_csv(TEST_TRX, infer_schema_length=10_000)
    return train_users, train_trx, test_users, test_trx


def time_aware_split(df_train_full: pl.DataFrame):
    df_sorted = df_train_full.sort("timestamp_reg", nulls_last=True)
    n = len(df_sorted)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_df = df_sorted[:n_train]
    val_df = df_sorted[n_train:n_train + n_val]
    holdout_df = df_sorted[n_train + n_val:]
    return train_df, val_df, holdout_df


def pick_params(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if PRESET_PARAMS is not None:
        return PRESET_PARAMS

    if USE_OPTUNA:
        print(f"Running Optuna ({N_TRIALS} trials)...")
        return run_optuna(X_train, y_train, n_trials=N_TRIALS)

    pos_weight = suggest_scale_pos_weight(y_train)
    return default_params(scale_pos_weight=pos_weight)


def main() -> None:
    train_users, train_trx, test_users, test_trx = load_raw_data()

    print("Building user-level features...")
    df_train_full = build_features(train_users, train_trx)
    df_test_full = build_features(test_users, test_trx)

    print(f"Train flat shape: {df_train_full.shape}")
    print(f"Test flat shape:  {df_test_full.shape}")

    train_df, val_df, holdout_df = time_aware_split(df_train_full)

    X_train, feat_names = to_model_matrix(train_df)
    X_val = align_to_features(val_df, feat_names)
    X_hold = align_to_features(holdout_df, feat_names)

    y_train = train_df["is_fraud"].to_numpy().astype(np.int8)
    y_val = val_df["is_fraud"].to_numpy().astype(np.int8)
    y_hold = holdout_df["is_fraud"].to_numpy().astype(np.int8)

    params = pick_params(X_train, y_train)
    print("Model params:")
    print(params)

    print("Training validation model...")
    model = FraudLGBMClassifier(lgbm_params=params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    val_f1, best_thr = model.tune_threshold(X_val, y_val)
    print(f"Validation threshold={best_thr:.3f} | val F1={val_f1:.4f}")

    hold_preds = model.predict(X_hold)
    hold_f1 = f1_score(y_hold, hold_preds, zero_division=0)
    print("=" * 45)
    print(f"HOLDOUT F1: {hold_f1:.4f}")
    print(f"Threshold : {model.threshold_:.3f}")
    print("=" * 45)
    print(classification_report(y_hold, hold_preds, target_names=["not fraud", "fraud"], zero_division=0))

    print("Retraining on full train set...")
    X_full, full_feat_names = to_model_matrix(df_train_full)
    y_full = df_train_full["is_fraud"].to_numpy().astype(np.int8)
    X_test = align_to_features(df_test_full, full_feat_names)

    final_model = FraudLGBMClassifier(lgbm_params=params, threshold=model.threshold_)
    final_model.fit(X_full, y_full)

    test_preds = final_model.predict(X_test)
    submission = pl.DataFrame(
        {
            "id_user": test_users["id_user"],
            "is_fraud": test_preds.tolist(),
        }
    )
    submission.write_csv("submission.csv")
    print(f"Saved submission.csv | rows={len(submission)} | fraud={int(test_preds.sum())} ({test_preds.mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
