from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import find_best_threshold, get_model, run_optuna
from src.preprocessing import (
    PolarsImputer,
    PolarsTargetEncoder,
    PolarsToModelFrame,
    load_and_merge,
)
from src.rule_based_filter import (
    apply_rule_based_filter,
    build_flat_user_dataset,
    build_submission,
    prepare_ml_zone,
)

# ── Конфіг ───────────────────────────────────────────────────────────────────

TRAIN_TRANSACTIONS_PATH = Path("data/train_transactions.csv")
TRAIN_USERS_PATH        = Path("data/train_users.csv")
TEST_TRANSACTIONS_PATH  = Path("data/test_transactions.csv")
TEST_USERS_PATH         = Path("data/test_users.csv")
SUBMISSION_PATH         = Path("submission.csv")

TARGET_COL   = "is_fraud"
ID_COL       = "id_user"
N_SPLITS     = 5
N_TRIALS     = 50
RANDOM_STATE = 42

# Multi-seed ensemble: кожен seed = окремий CV, фінал = середнє OOF
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

TARGET_ENCODE_COLS = [
    "reg_country",
    "traffic_type",
    "gender",
    "email_domain",
    "dominant_payment_country",
]

# ── Pipeline ──────────────────────────────────────────────────────────────────

print("=" * 65)
print("  FRAUD DETECTION PIPELINE")
print("=" * 65)

print("\n[1] Loading data...")
train_raw = load_and_merge(str(TRAIN_TRANSACTIONS_PATH), str(TRAIN_USERS_PATH))
test_raw  = load_and_merge(str(TEST_TRANSACTIONS_PATH),  str(TEST_USERS_PATH))
print(f"    Train: {train_raw.shape} | Test: {test_raw.shape}")

print("\n[2] Building user-level features...")
train_flat = build_flat_user_dataset(train_raw, is_train=True)
test_flat  = build_flat_user_dataset(test_raw,  is_train=False)

print("\n[3] Applying rule-based filter...")
train_filtered = apply_rule_based_filter(train_flat, verbose=True)
test_filtered  = apply_rule_based_filter(test_flat,  verbose=True)

print("\n[4] Preparing ML zone...")
train_ml, y = prepare_ml_zone(train_filtered, is_train=True)
test_ml,  _  = prepare_ml_zone(test_filtered,  is_train=False)

X      = train_ml.drop([TARGET_COL, "rule_decision", "rule_triggers"])
X_test = test_ml.drop(["rule_decision", "rule_triggers"])

n_fraud = int(y.sum())
n_total = len(y)
print(f"    ML zone → {n_total:,} train | {X_test.height:,} test users")
print(f"    Fraud rate: {y.mean():.4f}  ({n_fraud:,} fraud / {n_total - n_fraud:,} legit)")


# ── Preprocess fold ───────────────────────────────────────────────────────────

def preprocess_fold(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_val:   pl.DataFrame,
    X_test_polars: pl.DataFrame | None = None,
) -> tuple:
    """
    Fold-safe preprocessing: fit тільки на X_train, transform на val і test.
    Якщо передано X_test_polars — повертає (X_train_pd, X_val_pd, X_test_pd).
    Інакше — (X_train_pd, X_val_pd).
    """
    imputer = PolarsImputer()
    imputer.fit(X_train)
    X_train_i = imputer.transform(X_train)
    X_val_i   = imputer.transform(X_val)

    enc = PolarsTargetEncoder(cat_cols=TARGET_ENCODE_COLS)
    X_train_e = enc.fit_transform(X_train_i, y_train)   # LOO на train
    X_val_e   = enc.transform(X_val_i)                   # plain lookup на val

    frame = PolarsToModelFrame(drop_cols=[ID_COL])
    frame.fit(X_train_e)

    if X_test_polars is not None:
        X_test_e = enc.transform(imputer.transform(X_test_polars))
        return (
            frame.transform(X_train_e),
            frame.transform(X_val_e),
            frame.transform(X_test_e),
        )
    return frame.transform(X_train_e), frame.transform(X_val_e)


# ── Optuna ────────────────────────────────────────────────────────────────────

print(f"\n[5] Optuna hyperparameter search ({N_TRIALS} trials × {N_SPLITS} folds)...")
best_params = run_optuna(
    X=X,
    y=y,
    preprocess_fold=preprocess_fold,
    n_trials=N_TRIALS,
    n_splits=N_SPLITS,
)


# ── Multi-seed ensemble ───────────────────────────────────────────────────────

print(f"\n[6] Multi-seed ensemble OOF ({len(ENSEMBLE_SEEDS)} seeds × {N_SPLITS} folds)...")
print(f"    Seeds: {ENSEMBLE_SEEDS}\n")

oof_proba_accum  = np.zeros(len(y),        dtype=float)
test_proba_accum = np.zeros(X_test.height, dtype=float)

for seed_idx, seed in enumerate(ENSEMBLE_SEEDS, 1):
    print(f"  ── Seed {seed} ({seed_idx}/{len(ENSEMBLE_SEEDS)}) ──")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_proba_seed  = np.zeros(len(y),        dtype=float)
    test_proba_seed = np.zeros(X_test.height, dtype=float)
    fold_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        # preprocess_fold з X_test — encoder fit на train fold, transform test одразу
        X_train_pd, X_val_pd, X_test_pd = preprocess_fold(
            X[train_idx], y[train_idx], X[val_idx], X_test_polars=X_test
        )

        model = get_model(params=best_params)
        model.fit(X_train_pd, y[train_idx], eval_set=[(X_val_pd, y[val_idx])])

        oof_proba_seed[val_idx]  = model.predict_proba(X_val_pd)[:, 1]
        test_proba_seed         += model.predict_proba(X_test_pd)[:, 1] / N_SPLITS

        auc = roc_auc_score(y[val_idx], oof_proba_seed[val_idx])
        fold_aucs.append(auc)

        thr_f, f1_f = find_best_threshold(y[val_idx], oof_proba_seed[val_idx])
        print(
            f"    Fold {fold}: AUC={auc:.4f} | F1={f1_f:.4f}@{thr_f:.4f} | "
            f"proba [{oof_proba_seed[val_idx].min():.4f}–{oof_proba_seed[val_idx].max():.4f}]"
        )

    seed_auc      = float(np.mean(fold_aucs))
    seed_thr, seed_f1 = find_best_threshold(y, oof_proba_seed)
    print(f"    Seed {seed} → AUC={seed_auc:.4f}  OOF F1={seed_f1:.4f}@{seed_thr:.4f}\n")

    oof_proba_accum  += oof_proba_seed  / len(ENSEMBLE_SEEDS)
    test_proba_accum += test_proba_seed / len(ENSEMBLE_SEEDS)


# ── Threshold optimisation ────────────────────────────────────────────────────

print("[7] Threshold optimisation on ensemble OOF...")
print("-" * 65)

optimal_threshold, oof_f1 = find_best_threshold(y, oof_proba_accum)
oof_pred = (oof_proba_accum >= optimal_threshold).astype(int)
tp = int(np.sum((oof_pred == 1) & (y == 1)))
fp = int(np.sum((oof_pred == 1) & (y == 0)))
fn = int(np.sum((oof_pred == 0) & (y == 1)))
p  = tp / max(tp + fp, 1)
r  = tp / max(tp + fn, 1)
oof_auc = roc_auc_score(y, oof_proba_accum)

print(f"  thr={optimal_threshold:.4f} | F1={oof_f1:.4f} | P={p:.4f} | R={r:.4f}")
print(f"  TP={tp}  FP={fp}  FN={fn}")
print("-" * 65)
print(f"\n  OOF AUC : {oof_auc:.4f}")
print(f"  OOF F1  : {oof_f1:.4f}")


# ── Submission ────────────────────────────────────────────────────────────────

print("\n[8] Building submission...")
submission = build_submission(test_filtered, test_proba_accum, optimal_threshold)
submission.write_csv(SUBMISSION_PATH)

print(f"\n{'='*65}")
print(f"  Saved → {SUBMISSION_PATH}")
print(f"  OOF F1 = {oof_f1:.4f}  |  AUC = {oof_auc:.4f}")
print(f"{'='*65}\n")