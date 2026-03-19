from __future__ import annotations

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
except Exception:
    optuna = None


RANDOM_STATE = 42


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        score = f1_score(y_true, (probs >= thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_thr = float(score), float(thr)
    return best_f1, best_thr


class FraudLGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lgbm_params: dict | None = None, threshold: float = 0.5):
        self.lgbm_params = lgbm_params or {}
        self.threshold = threshold

    def fit(self, X, y, eval_set=None):
        params = {
            "objective": "binary",
            "n_jobs": -1,
            "verbose": -1,
            "random_state": RANDOM_STATE,
            **self.lgbm_params,
        }

        self.model_ = LGBMClassifier(**params)
        self.classes_ = np.array([0, 1])
        self.threshold_ = self.threshold

        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs = {
                "eval_set": eval_set,
                "callbacks": [lgb.early_stopping(50, verbose=False)],
            }

        self.model_.fit(X, y, **fit_kwargs)

        best_iter = getattr(self.model_, "best_iteration_", None)
        self.best_iteration_ = int(best_iter) if best_iter is not None and best_iter > 0 else None
        return self

    def predict_proba(self, X):
        if getattr(self, "best_iteration_", None) is not None:
            return self.model_.predict_proba(X, num_iteration=self.best_iteration_)
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def tune_threshold(self, X_val, y_val) -> tuple[float, float]:
        probs = self.predict_proba(X_val)[:, 1]
        best_f1, best_thr = find_best_threshold(y_val, probs)
        self.threshold_ = best_thr
        return best_f1, best_thr


def default_params(scale_pos_weight: float | None = None) -> dict:
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return params


def suggest_scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / max(pos, 1))


def run_optuna(
    X_train,
    y_train: np.ndarray,
    n_splits: int = 5,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict:
    if optuna is None:
        raise ImportError("optuna is not installed. Install it or disable tuning.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0),
        }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores: list[float] = []

        for tr_idx, vl_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
            X_vl = X_train.iloc[vl_idx] if hasattr(X_train, "iloc") else X_train[vl_idx]

            clf = FraudLGBMClassifier(lgbm_params=params)
            clf.fit(X_tr, y_train[tr_idx], eval_set=[(X_vl, y_train[vl_idx])])
            clf.tune_threshold(X_vl, y_train[vl_idx])
            preds = clf.predict(X_vl)
            scores.append(f1_score(y_train[vl_idx], preds, zero_division=0))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return dict(study.best_params)