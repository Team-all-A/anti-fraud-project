from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.business import optimize_threshold
from src.model import get_model, run_optuna, find_best_threshold
from src.preprocessing import PolarsImputer, PolarsToModelFrame, PolarsTargetEncoder, load_and_merge
from src.rule_based_filter import (
    apply_rule_based_filter,
    build_flat_user_dataset,
    build_submission,
    prepare_ml_zone,
)


# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_TRANSACTIONS_PATH = Path("data/train_transactions.csv")
TRAIN_USERS_PATH        = Path("data/train_users.csv")
TEST_TRANSACTIONS_PATH  = Path("data/test_transactions.csv")
TEST_USERS_PATH         = Path("data/test_users.csv")

TARGET_COL      = "is_fraud"
ID_COL          = "id_user"
N_SPLITS        = 5
N_TRIALS        = 50
RANDOM_STATE    = 42
COST_FP         = 50.0
COST_FN         = 500.0
SUBMISSION_PATH = Path("submission.csv")

TARGET_ENCODE_COLS = [
    "reg_country",
    "traffic_type",
    "gender",
    "email_domain",
    "dominant_payment_country",
]


# ── 1. Load raw data ──────────────────────────────────────────────────────────

print("Loading data...")
train_raw = load_and_merge(str(TRAIN_TRANSACTIONS_PATH), str(TRAIN_USERS_PATH))
test_raw  = load_and_merge(str(TEST_TRANSACTIONS_PATH),  str(TEST_USERS_PATH))
print(f"Train: {train_raw.shape} | Test: {test_raw.shape}")


# ── 2. Aggregate to user level ────────────────────────────────────────────────
# Safe globally: each user's stats come from their own transactions only.

print("\nBuilding user-level features...")
train_flat = build_flat_user_dataset(train_raw, is_train=True)
test_flat  = build_flat_user_dataset(test_raw,  is_train=False)


# ── 3. Rule-based filter ──────────────────────────────────────────────────────

print("\nApplying rule-based filter...")
train_filtered = apply_rule_based_filter(train_flat, verbose=True)
test_filtered  = apply_rule_based_filter(test_flat,  verbose=True)


# ── 4. Prepare ML zone ────────────────────────────────────────────────────────


train_ml, y = prepare_ml_zone(train_filtered, is_train=True)
test_ml,  _ = prepare_ml_zone(test_filtered,  is_train=False)

X      = train_ml.drop([TARGET_COL, "rule_decision", "rule_triggers"])
X_test = test_ml.drop(["rule_decision", "rule_triggers"])

print(f"\nML zone → {X.height} train users | {X_test.height} test users")
print(f"Fraud rate in ML zone: {y.mean():.4f}")


# ── 5. Fold preprocessor (used inside both Optuna and the OOF loop) ───────────
#
# This function is the single source of truth for what happens inside a fold.
# Both run_optuna and the OOF CV loop call it — guaranteeing identical
# preprocessing in both places.
#
# It fits the imputer and target encoder on X_train, then applies them to
# X_val (and optionally X_test). Stateful steps never see val data.

def preprocess_fold(
    X_train:         pl.DataFrame,
    y_train:         np.ndarray,
    X_val:           pl.DataFrame,
    X_test_polars:   pl.DataFrame | None = None,
) -> tuple:
    """
    Fit all stateful preprocessing on X_train only.

    All inputs must be Polars DataFrames. Outputs are pandas DataFrames
    ready for LightGBM. PolarsToModelFrame.fit() is called here while
    X_train is still Polars — never on the pandas output.

    Returns (X_train_pd, X_val_pd) when X_test_polars is None.
    Returns (X_train_pd, X_val_pd, X_test_pd) when X_test_polars is provided.
    """
    imputer = PolarsImputer()
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_val   = imputer.transform(X_val)

    target_enc = PolarsTargetEncoder(cat_cols=TARGET_ENCODE_COLS)
    X_train    = target_enc.fit_transform(X_train, y_train)
    X_val      = target_enc.transform(X_val)

    # frame.fit() must receive a Polars DataFrame — X_train is still Polars here.
    frame = PolarsToModelFrame(drop_cols=[ID_COL])
    frame.fit(X_train)

    if X_test_polars is not None:
        X_test_transformed = target_enc.transform(imputer.transform(X_test_polars))
        return frame.transform(X_train), frame.transform(X_val), frame.transform(X_test_transformed)

    return frame.transform(X_train), frame.transform(X_val)


# ── 6. Optuna hyperparameter tuning ──────────────────────────────────────────
# Runs N_TRIALS × N_SPLITS model fits.
# preprocess_fold is called inside every CV fold so there is no leakage.

print(f"\nRunning Optuna ({N_TRIALS} trials × {N_SPLITS} folds)...")
best_params = run_optuna(
    X=X,
    y=y,
    preprocess_fold=preprocess_fold,
    n_trials=N_TRIALS,
    n_splits=N_SPLITS,
)


# ── 7. OOF cross-validation with best params ─────────────────────────────────
# Same fold splits and same preprocess_fold as in Optuna.
# Produces out-of-fold probabilities for threshold optimisation.

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_proba  = np.zeros(len(y), dtype=float)
test_proba = np.zeros(X_test.height, dtype=float)
fold_aucs: list[float] = []
fold_f1s:  list[float] = []

print(f"\nOOF CV with best params ({N_SPLITS} folds)...\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    X_train_fold = X[train_idx]
    X_val_fold   = X[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold   = y[val_idx]

    X_train_pd, X_val_pd, X_test_pd = preprocess_fold(
        X_train_fold, y_train_fold, X_val_fold, X_test_polars=X_test
    )

    model = get_model(params=best_params)
    model.fit(X_train_pd, y_train_fold, eval_set=[(X_val_pd, y_val_fold)])

    oof_proba[val_idx] = model.predict_proba(X_val_pd)[:, 1]
    test_proba        += model.predict_proba(X_test_pd)[:, 1] / N_SPLITS

    val_proba = oof_proba[val_idx]
    fold_auc  = roc_auc_score(y_val_fold, val_proba)
    _, fold_f1 = find_best_threshold(y_val_fold, val_proba)
    fold_aucs.append(fold_auc)
    fold_f1s.append(fold_f1)
    print(
        f"  Fold {fold}: AUC={fold_auc:.4f}  F1={fold_f1:.4f} | "
        f"proba min={val_proba.min():.4f} mean={val_proba.mean():.4f} max={val_proba.max():.4f} | "
        f"fraud {y_val_fold.sum()}/{len(y_val_fold)}"
    )

print(f"\nCV mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"CV mean F1 : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

# ── Probability calibration (isotonic) ───────────────────────────────────────
# OOF ймовірності стиснуті навколо prior (~0.03) через низький scale_pos_weight.
# Isotonic regression відображає raw proba → calibrated proba без витоку:
# вона fit на OOF (вже out-of-fold), apply на test через той самий маппінг.
# ── Probability calibration (Platt scaling) ───────────────────────────────────
# Isotonic regression overfits на вузькому діапазоні (0.034–0.054) і будує
# ступінчасту функцію → більшість прогнозів йде до 0 або 1.
# LogisticRegression (Platt) — плавна монотонна функція, менш схильна до overfitting.
# Fit на OOF (out-of-fold) → застосовуємо той самий маппінг до test.
from sklearn.linear_model import LogisticRegression as _LR

print("\nCalibrating probabilities (Platt scaling)...")
platt = _LR(C=1.0, solver="lbfgs", max_iter=1000)
platt.fit(oof_proba.reshape(-1, 1), y)
oof_proba_cal  = platt.predict_proba(oof_proba.reshape(-1, 1))[:, 1]
test_proba_cal = platt.predict_proba(test_proba.reshape(-1, 1))[:, 1]

fraud_mask = (y == 1)
print(f"  Before cal → fraud mean={oof_proba[fraud_mask].mean():.5f}  legit mean={oof_proba[~fraud_mask].mean():.5f}")
print(f"  After  cal → fraud mean={oof_proba_cal[fraud_mask].mean():.5f}  legit mean={oof_proba_cal[~fraud_mask].mean():.5f}")

# ── Аналіз розподілу ймовірностей ─────────────────────────────────────────────
print("\nOOF Calibrated probability distribution:")
print(f"  Min:    {oof_proba_cal.min():.5f}")
print(f"  Median: {np.median(oof_proba_cal):.5f}")
print(f"  Mean:   {oof_proba_cal.mean():.5f}")
print(f"  90th %: {np.percentile(oof_proba_cal, 90):.5f}")
print(f"  95th %: {np.percentile(oof_proba_cal, 95):.5f}")
print(f"  99th %: {np.percentile(oof_proba_cal, 99):.5f}")
print(f"  Max:    {oof_proba_cal.max():.5f}")

print(f"\n  Mean proba for FRAUD (y=1): {oof_proba_cal[fraud_mask].mean():.5f}")
print(f"  Mean proba for LEGIT (y=0): {oof_proba_cal[~fraud_mask].mean():.5f}")


# ── 8. Threshold optimisation on calibrated OOF predictions ──────────────────

print("\nOptimising threshold on calibrated OOF predictions...")
optimal_threshold, best_f1, metrics = optimize_threshold(
    y_true=y,
    y_proba=oof_proba_cal,
)
print(f"Threshold : {optimal_threshold:.4f} | OOF F1: {best_f1:.4f}")
print(f"F1={metrics['f1']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}")
print(f"TP={metrics['tp']} | FP={metrics['fp']} | FN={metrics['fn']} | TN={metrics['tn']}")


# ── 9. Submission ─────────────────────────────────────────────────────────────

submission = build_submission(test_filtered, test_proba_cal, optimal_threshold)
submission.write_csv(SUBMISSION_PATH)
print(f"\nSaved {SUBMISSION_PATH}")