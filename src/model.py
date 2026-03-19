from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute LightGBM's scale_pos_weight from the training labels only.

    Formula:
        negatives / positives

    Why:
    - better than a hardcoded constant
    - adapts to the actual class imbalance in each train fold
    - avoids using validation/test label distribution
    """
    y = np.asarray(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    if positives == 0:
        return 1.0

    return max(1.0, negatives / positives)


def get_model(
    y_train: np.ndarray | None = None,
    random_state: int = 42,
) -> LGBMClassifier:
    """
    Build a LightGBM classifier for imbalanced fraud detection.

    Design goals:
    - strong baseline for tabular fraud data
    - robust on mixed numeric engineered features
    - no dependence on feature scaling
    - class imbalance handled from training labels when available
    """
    scale_pos_weight = (
        compute_scale_pos_weight(y_train)
        if y_train is not None
        else 1.0
    )

    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )