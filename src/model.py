from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_steps: int = 300,
) -> tuple[float, float]:
    # Шукаємо від p1 до max (не p99) — при стиснутих ймовірностях
    # p99 може бути всього 0.033 при max=0.100, і оптимум поза діапазоном.
    p_min = float(np.percentile(probs, 1))
    p_max = float(probs.max())

    best_f1, best_thr = 0.0, (p_min + p_max) / 2
    for thr in np.linspace(p_min, p_max, n_steps):
        f = f1_score(y_true, (probs >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_thr = float(thr)
    return best_thr, best_f1


def compute_scale_pos_weight(y: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0
    ratio = negatives / positives
    print(f"  neg/pos ratio: {negatives}/{positives} = {ratio:.1f}")
    return max(1.0, ratio)



# КЛАСИФІКАТОР
# ─────────────────────────────────────────────────────────────────────────────
class FraudLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lgbm_params: dict | None = None,
        threshold: float = 0.5,
        early_stopping_rounds: int = 80,
    ):
        self.lgbm_params           = lgbm_params or {}
        self.threshold             = threshold
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, eval_set=None):
        params = {
            "objective":     "binary",
            "boosting_type": "gbdt",
            "n_jobs":        -1,
            "verbose":       -1,
            "random_state":  RANDOM_STATE,
            "metric":        "auc",
            # is_unbalance=True використовує внутрішній oversampling замість
            # scale_pos_weight. Це дає кращу калібрацію ймовірностей при
            # збереженні AUC, бо не штучно зміщує posterior.
            "is_unbalance":  True,
        }
        # lgbm_params може містити scale_pos_weight з Optuna — видаляємо його,
        # бо використовуємо is_unbalance замість нього.
        lgbm_params_clean = {
            k: v for k, v in self.lgbm_params.items()
            if k != "scale_pos_weight"
        }
        params = {**params, **lgbm_params_clean}
        self.model_     = LGBMClassifier(**params)
        self.classes_   = np.array([0, 1])
        self.threshold_ = float(self.threshold)

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"]   = eval_set
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            ]
        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def tune_threshold(self, X_val, y_val) -> tuple[float, float]:
        probs = self.predict_proba(X_val)[:, 1]
        best_thr, best_f1 = find_best_threshold(y_val, probs)
        self.threshold_ = best_thr
        return best_thr, best_f1



# ДЕФОЛТНІ ПАРАМЕТРИ
# ─────────────────────────────────────────────────────────────────────────────
def get_default_lgbm_params(y_train: np.ndarray | None = None) -> dict:
    spw = compute_scale_pos_weight(y_train) if y_train is not None else 38.0
    return {
        "n_estimators":      1000,
        "learning_rate":     0.02,
        "num_leaves":        63,
        "max_depth":         -1,
        "min_child_samples": 20,
        "subsample":         0.8,
        "subsample_freq":    1,
        "colsample_bytree":  0.7,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "scale_pos_weight":  spw,
    }


def get_model(
    y_train: np.ndarray | None = None,
    params: dict | None = None,
    threshold: float = 0.5,
) -> FraudLGBMClassifier:
    lgbm_params = params if params is not None else get_default_lgbm_params(y_train)
    return FraudLGBMClassifier(lgbm_params=lgbm_params, threshold=threshold)



# OPTUNA
# ─────────────────────────────────────────────────────────────────────────────
def run_optuna(
    X: pl.DataFrame,
    y: np.ndarray,
    preprocess_fold: Callable[[pl.DataFrame, np.ndarray, pl.DataFrame], tuple],
    n_trials: int = 50,
    n_splits: int = 5,
) -> dict:
    # is_unbalance=True вже в FraudLGBMClassifier.fit() — не потрібен SPW.
    print(f"  Using is_unbalance=True (internal oversampling, no scale_pos_weight)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 127),
            "max_depth":         trial.suggest_int("max_depth", 4, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",  1e-8, 2.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 2.0, log=True),
            # scale_pos_weight прибрано — використовується is_unbalance=True
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            X_train_pd, X_val_pd = preprocess_fold(
                X[train_idx], y[train_idx], X[val_idx]
            )
            clf = FraudLGBMClassifier(lgbm_params=params)
            clf.fit(X_train_pd, y[train_idx], eval_set=[(X_val_pd, y[val_idx])])

            probs = clf.predict_proba(X_val_pd)[:, 1]
            # AUC — threshold-незалежна метрика, чесніша для hyperparameter search.
            # find_best_threshold всередині objective завищує оцінку кожного трайлу
            # (optimal threshold на val ≠ threshold який буде в продакшені).
            # Threshold оптимізується окремо на OOF після повного CV.
            fold_score = roc_auc_score(y[val_idx], probs)
            scores.append(fold_score)

            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best CV AUC : {study.best_value:.4f}")
    print(f"Best params        : {study.best_params}")
    return study.best_params