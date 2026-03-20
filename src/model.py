from __future__ import annotations

from typing import Callable

import numpy as np
import lightgbm as lgb
import optuna
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Utilities ─────────────────────────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    probs:  np.ndarray,
    start:  float = 0.05,
    stop:   float = 0.95,
    step:   float = 0.01,
) -> tuple[float, float]:
    """Grid-search the threshold that maximises F1. Returns (best_f1, best_threshold)."""
    best_f1  = 0.0
    best_thr = 0.5

    for thr in np.arange(start, stop, step):
        f = f1_score(y_true, (probs >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1  = float(f)
            best_thr = float(thr)

    return best_f1, best_thr


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Returns neg/pos ratio for LightGBM class imbalance handling."""
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return max(1.0, negatives / positives) if positives > 0 else 1.0


# ── Classifier ────────────────────────────────────────────────────────────────

class FraudLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible LightGBM wrapper.

    Supports:
    - predict_proba() for probability output
    - predict()       with a tunable decision threshold
    - tune_threshold() to find the best F1 threshold on a val set
    - early stopping  when eval_set is provided to fit()
    """

    def __init__(
        self,
        lgbm_params:           dict | None = None,
        threshold:             float = 0.5,
        early_stopping_rounds: int   = 50,
    ):
        self.lgbm_params           = lgbm_params or {}
        self.threshold             = threshold
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, eval_set=None):
        base_params = {
            "objective":    "binary",
            "boosting_type": "gbdt",
            "n_jobs":        -1,
            "verbose":       -1,
            "random_state":  RANDOM_STATE,
        }
        params = {**base_params, **self.lgbm_params}

        self.model_     = LGBMClassifier(**params)
        self.classes_   = np.array([0, 1])
        self.threshold_ = float(self.threshold)

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"]  = eval_set
            fit_kwargs["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def tune_threshold(self, X_val, y_val) -> tuple[float, float]:
        """Find the F1-maximising threshold on a val set. Returns (best_threshold, best_f1)."""
        probs             = self.predict_proba(X_val)[:, 1]
        best_f1, best_thr = find_best_threshold(y_val, probs)
        self.threshold_   = best_thr
        return best_thr, best_f1


# ── Default params (used when Optuna is skipped) ──────────────────────────────

def get_default_lgbm_params(y_train: np.ndarray | None = None) -> dict:
    return {
        "n_estimators":      700,
        "learning_rate":     0.03,
        "num_leaves":        31,
        "max_depth":         -1,
        "min_child_samples": 40,
        "subsample":         0.8,
        "subsample_freq":    1,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.5,
        "reg_lambda":        1.0,
        "scale_pos_weight":  compute_scale_pos_weight(y_train) if y_train is not None else 1.0,
    }


def get_model(
    y_train:   np.ndarray | None = None,
    params:    dict | None       = None,
    threshold: float             = 0.5,
) -> FraudLGBMClassifier:
    """
    Factory function.
    - Pass `params` (from run_optuna) to use tuned hyperparameters.
    - Omit `params` to fall back to sensible defaults.
    """
    lgbm_params = params if params is not None else get_default_lgbm_params(y_train)
    return FraudLGBMClassifier(lgbm_params=lgbm_params, threshold=threshold)


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def run_optuna(
    X:               pl.DataFrame,
    y:               np.ndarray,
    preprocess_fold: Callable[[pl.DataFrame, np.ndarray, pl.DataFrame], tuple],
    n_trials:        int = 50,
    n_splits:        int = 5,
) -> dict:
    """
    Tune LightGBM hyperparameters with Optuna.

    Each trial runs a full stratified K-Fold CV. Preprocessing (imputation,
    target encoding, column selection) is done inside every fold through the
    `preprocess_fold` callable, which is defined and passed in from main.py.
    This keeps model.py decoupled from preprocessing logic.

    Args:
        X:               Polars DataFrame — ML-zone users before preprocessing.
        y:               Target array aligned with X rows.
        preprocess_fold: Callable with signature:
                           (X_train, y_train, X_val) -> (X_train_pd, X_val_pd)
                         Must fit all stateful steps (imputer, target encoder,
                         model frame) on X_train only, then apply to X_val.
        n_trials:        Number of Optuna trials.
        n_splits:        Number of CV folds per trial.

    Returns:
        dict of best hyperparameters, ready to pass to get_model(params=...).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",       200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate",    0.01,  0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves",          20,  200),
            "max_depth":         trial.suggest_int("max_depth",            4,   10),
            "min_child_samples": trial.suggest_int("min_child_samples",   10,  100),
            "subsample":         trial.suggest_float("subsample",         0.6,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.6,  1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-8,  5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-8,  5.0, log=True),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight",  1.0, 30.0),
        }

        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
            X_train_fold = X[train_idx]
            X_val_fold   = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold   = y[val_idx]

            # Preprocessing is fold-safe: fitted on train only, applied to val.
            X_train_pd, X_val_pd = preprocess_fold(X_train_fold, y_train_fold, X_val_fold)

            clf = FraudLGBMClassifier(lgbm_params=params)
            clf.fit(X_train_pd, y_train_fold, eval_set=[(X_val_pd, y_val_fold)])
            
            scores.append(roc_auc_score(y_val_fold, clf.predict_proba(X_val_pd)[:, 1]))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best CV AUC : {study.best_value:.4f}")
    print(f"Best params       : {study.best_params}")

    return study.best_params