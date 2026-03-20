from __future__ import annotations

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score


RANDOM_STATE = 42


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Search for the threshold that maximizes F1.

    Returns:
        best_f1, best_threshold
    """
    best_f1 = 0.0
    best_thr = 0.5

    for thr in np.arange(start, stop, step):
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)

    return best_f1, best_thr


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute class imbalance weight from the training labels only.
    """
    y = np.asarray(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    if positives == 0:
        return 1.0

    return max(1.0, negatives / positives)


class FraudLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible LightGBM wrapper from the ML engineer's approach.

    Why this fits your current project:
    - works inside sklearn Pipeline
    - supports predict_proba()
    - supports custom threshold via predict()
    - keeps LightGBM config centralized
    """

    def __init__(
        self,
        lgbm_params: dict | None = None,
        threshold: float = 0.5,
        use_early_stopping: bool = False,
        early_stopping_rounds: int = 50,
    ):
        self.lgbm_params = lgbm_params or {}
        self.threshold = threshold
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, eval_set=None):
        """
        Fit underlying LightGBM model.

        Note:
        In your current Pipeline-based architecture, eval_set is usually not used,
        because validation data would need to be transformed by the same pipeline
        first. So this wrapper supports it, but your current main flow does not
        rely on it.
        """
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_jobs": -1,
            "verbose": -1,
            "random_state": RANDOM_STATE,
            **self.lgbm_params,
        }

        self.model_ = LGBMClassifier(**params)
        self.classes_ = np.array([0, 1])
        self.threshold_ = float(self.threshold)

        fit_kwargs = {}

        if self.use_early_stopping and eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            ]

        self.model_.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def tune_threshold(self, X_val, y_val) -> tuple[float, float]:
        """
        Tune decision threshold on a validation set.

        Returns:
            best_threshold, best_f1
        """
        probs = self.predict_proba(X_val)[:, 1]
        best_f1, best_thr = find_best_threshold(y_val, probs)
        self.threshold_ = best_thr
        return best_thr, best_f1


def get_default_lgbm_params(y_train: np.ndarray | None = None) -> dict:
    """
    Default LightGBM params adapted from your earlier project setup,
    but wrapped into the ML engineer's model class.
    """
    scale_pos_weight = (
        compute_scale_pos_weight(y_train)
        if y_train is not None
        else 1.0
    )

    return {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }


def get_model(
    y_train: np.ndarray | None = None,
    threshold: float = 0.5,
) -> FraudLGBMClassifier:
    """
    Factory used by your existing main.py and pipeline.py.

    This keeps your current architecture unchanged.
    """
    params = get_default_lgbm_params(y_train)

    return FraudLGBMClassifier(
        lgbm_params=params,
        threshold=threshold,
        use_early_stopping=False,
        early_stopping_rounds=50,
    )