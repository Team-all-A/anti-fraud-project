from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.business import optimize_threshold
from src.features import build_flat_user_dataset
from src.model import get_model, run_optuna, find_best_threshold
from src.preprocessing import PolarsImputer, PolarsToModelFrame, PolarsTargetEncoder, load_and_merge
from src.rule_based_filter import (
    apply_rule_based_filter,
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

    # Encode only columns that actually exist in X_train — safety guard
    actual_encode_cols = [c for c in TARGET_ENCODE_COLS if c in X_train.columns]
    target_enc = PolarsTargetEncoder(cat_cols=actual_encode_cols)
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

# Calibration (isotonic/Platt) не працює коли raw proba стиснуті у вузький
# діапазон (~0.02–0.05) — обидва методи або overfitяться або колапсують до prior.
# Використовуємо raw OOF ймовірності напряму.

fraud_mask = (y == 1)

# ── Аналіз розподілу ймовірностей ─────────────────────────────────────────────
print("\nOOF Probability distribution:")
print(f"  Min:    {oof_proba.min():.5f}")
print(f"  Median: {np.median(oof_proba):.5f}")
print(f"  Mean:   {oof_proba.mean():.5f}")
print(f"  90th %: {np.percentile(oof_proba, 90):.5f}")
print(f"  95th %: {np.percentile(oof_proba, 95):.5f}")
print(f"  99th %: {np.percentile(oof_proba, 99):.5f}")
print(f"  Max:    {oof_proba.max():.5f}")
print(f"\n  Mean proba for FRAUD (y=1): {oof_proba[fraud_mask].mean():.5f}")
print(f"  Mean proba for LEGIT (y=0): {oof_proba[~fraud_mask].mean():.5f}")


# ── 8. Threshold optimisation on raw OOF predictions ─────────────────────────
# optimize_threshold шукає по сітці 0.01–0.99 з кроком 0.01 — занадто грубо
# для вузького діапазону ймовірностей (0.033–0.071).
# find_best_threshold шукає 300 кроків між p1 і p99 реального розподілу —
# той самий метод що використовується в CV фолдах → результат узгоджений.

print("\nOptimising threshold on OOF predictions...")

# Абсолютні ймовірності стиснуті в діапазон 0.001 — threshold на них ненадійний.
# Замість цього класифікуємо топ-K% юзерів за рангом ймовірності.
# K оптимізується на OOF (без leakage).
# Це еквівалентно threshold але robust до probability compression.

n_total  = len(y)
n_fraud  = int(y.sum())

# Шукаємо оптимальний відсоток юзерів для позначення як fraud
# Діапазон: від 0.5× до 5× реального fraud rate
fraud_rate = n_fraud / n_total
k_lo = max(1, int(n_total * fraud_rate * 0.5))
k_hi = int(n_total * fraud_rate * 5.0)

best_f1_rank  = 0.0
best_k        = n_fraud  # default = actual count
ranked_idx    = np.argsort(oof_proba)[::-1]  # від найвищого до найнижчого

print(f"  Searching top-K fraud in range [{k_lo:,} – {k_hi:,}] users...")

for k in np.unique(np.linspace(k_lo, k_hi, 500).astype(int)):
    preds = np.zeros(n_total, dtype=int)
    preds[ranked_idx[:k]] = 1
    f = f1_score(y, preds, zero_division=0)
    if f > best_f1_rank:
        best_f1_rank = float(f)
        best_k       = int(k)

# Конвертуємо best_k назад у threshold для submission
optimal_threshold = float(oof_proba[ranked_idx[best_k - 1]])

from sklearn.metrics import precision_score, recall_score
best_preds = np.zeros(n_total, dtype=int)
best_preds[ranked_idx[:best_k]] = 1
tp = int(((y == 1) & (best_preds == 1)).sum())
fp = int(((y == 0) & (best_preds == 1)).sum())
fn = int(((y == 1) & (best_preds == 0)).sum())
tn = int(((y == 0) & (best_preds == 0)).sum())
prec = tp / max(tp + fp, 1)
rec  = tp / max(tp + fn, 1)

print(f"Best top-K  : {best_k:,} users ({100*best_k/n_total:.2f}% of ML zone)")
print(f"Threshold   : {optimal_threshold:.5f} | OOF F1: {best_f1_rank:.4f}")
print(f"F1={best_f1_rank:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
print(f"TP={tp} | FP={fp} | FN={fn} | TN={tn}")

# Threshold curve для діагностики
print("\nTop-K curve (F1 at key percentages of ML zone):")
for pct in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
    k_tmp = int(n_total * pct / 100)
    if k_tmp < 1: continue
    preds_tmp = np.zeros(n_total, dtype=int)
    preds_tmp[ranked_idx[:k_tmp]] = 1
    f = f1_score(y, preds_tmp, zero_division=0)
    p = precision_score(y, preds_tmp, zero_division=0)
    r = recall_score(y, preds_tmp, zero_division=0)
    marker = " ← optimal" if abs(k_tmp - best_k) < n_total * 0.005 else ""
    print(f"  top {pct:>2}%  k={k_tmp:>6,}  F1={f:.4f}  prec={p:.4f}  rec={r:.4f}{marker}")


# ── 9. Submission ─────────────────────────────────────────────────────────────

submission = build_submission(test_filtered, test_proba, optimal_threshold)
submission.write_csv(SUBMISSION_PATH)
print(f"\nSaved {SUBMISSION_PATH}")