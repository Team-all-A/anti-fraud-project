from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.business import export_threshold_report
from src.model import find_best_threshold, find_best_topk, get_model, run_optuna
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
THRESHOLD_REPORT_PATH   = Path("threshold_report.csv")

TARGET_COL   = "is_fraud"
ID_COL       = "id_user"
N_SPLITS     = 5
N_TRIALS     = 150      # збільшено з 50 — best trial був 43/50, простір не вичерпано
RANDOM_STATE = 42

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
    X_train:       pl.DataFrame,
    y_train:       np.ndarray,
    X_val:         pl.DataFrame,
    X_test_polars: pl.DataFrame | None = None,
) -> tuple:
    """
    Fold-safe preprocessing: fit тільки на X_train, transform на val і test.
    Повертає (X_train_pd, X_val_pd) або (X_train_pd, X_val_pd, X_test_pd).
    """
    imputer = PolarsImputer()
    imputer.fit(X_train)
    X_train_i = imputer.transform(X_train)
    X_val_i   = imputer.transform(X_val)

    enc = PolarsTargetEncoder(cat_cols=TARGET_ENCODE_COLS)
    X_train_e = enc.fit_transform(X_train_i, y_train)  # LOO на train
    X_val_e   = enc.transform(X_val_i)                  # plain lookup на val

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

    seed_auc          = float(np.mean(fold_aucs))
    seed_thr, seed_f1 = find_best_threshold(y, oof_proba_seed)
    print(f"    Seed {seed} → AUC={seed_auc:.4f}  OOF F1={seed_f1:.4f}@{seed_thr:.4f}\n")

    oof_proba_accum  += oof_proba_seed  / len(ENSEMBLE_SEEDS)
    test_proba_accum += test_proba_seed / len(ENSEMBLE_SEEDS)


# ── Threshold optimisation ────────────────────────────────────────────────────
# Порівнюємо два підходи і беремо кращий:
#   1. find_best_threshold — лінійний пошук по абсолютних значеннях proba
#   2. find_best_topk      — rank-based, не залежить від абсолютних значень
#                            переважає коли proba стиснуті в вузький діапазон

print("[7] Threshold optimisation on ensemble OOF...")
print("-" * 65)

thr_abs, f1_abs   = find_best_threshold(y, oof_proba_accum)
k_opt, f1_topk, thr_topk = find_best_topk(y, oof_proba_accum)

pred_abs  = (oof_proba_accum >= thr_abs).astype(int)
tp_a = int(np.sum((pred_abs == 1) & (y == 1)))
fp_a = int(np.sum((pred_abs == 1) & (y == 0)))
fn_a = int(np.sum((pred_abs == 0) & (y == 1)))

pred_topk = (oof_proba_accum >= thr_topk).astype(int)
tp_k = int(np.sum((pred_topk == 1) & (y == 1)))
fp_k = int(np.sum((pred_topk == 1) & (y == 0)))
fn_k = int(np.sum((pred_topk == 0) & (y == 1)))

print(f"  Threshold search : thr={thr_abs:.4f} | F1={f1_abs:.4f} | "
      f"P={tp_a/max(tp_a+fp_a,1):.4f} | R={tp_a/max(tp_a+fn_a,1):.4f} | TP={tp_a} FP={fp_a} FN={fn_a}")
print(f"  Rank Top-K={k_opt:,}  : thr={thr_topk:.4f} | F1={f1_topk:.4f} | "
      f"P={tp_k/max(tp_k+fp_k,1):.4f} | R={tp_k/max(tp_k+fn_k,1):.4f} | TP={tp_k} FP={fp_k} FN={fn_k}")

if f1_topk >= f1_abs:
    optimal_threshold = thr_topk
    oof_f1            = f1_topk
    winner            = f"Rank Top-K={k_opt:,} (F1={f1_topk:.4f})"
else:
    optimal_threshold = thr_abs
    oof_f1            = f1_abs
    winner            = f"Threshold search (F1={f1_abs:.4f})"

oof_auc = roc_auc_score(y, oof_proba_accum)
print(f"\n  → Using {winner}")
print("-" * 65)
print(f"\n  OOF AUC : {oof_auc:.4f}")
print(f"  OOF F1  : {oof_f1:.4f}")


# ── Submission ────────────────────────────────────────────────────────────────

print("\n[8] Building submission...")
submission = build_submission(test_filtered, test_proba_accum, optimal_threshold)
submission.write_csv(SUBMISSION_PATH)

predicted_fraud_rate_test = float(submission["is_fraud"].mean())


# ── Business analyst report (незалежно від submission) ───────────────────────
# Генерується на базі OOF proba які вже є в пам'яті.
# Не впливає на submission — просто записує CSV для аналітика.

print("\n[9] Exporting business analyst report...")
export_threshold_report(
    y_true=y,
    oof_proba=oof_proba_accum,
    optimal_threshold=optimal_threshold,
    oof_f1=oof_f1,
    oof_auc=oof_auc,
    n_train_total=train_raw.height,
    n_test_total=test_raw.height,
    predicted_fraud_rate_test=predicted_fraud_rate_test,
    output_path=THRESHOLD_REPORT_PATH,
)


print(f"\n{'='*65}")
print(f"  Saved → {SUBMISSION_PATH}")
print(f"  OOF F1 = {oof_f1:.4f}  |  AUC = {oof_auc:.4f}")
print(f"{'='*65}\n")