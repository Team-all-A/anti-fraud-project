from lightgbm import LGBMClassifier

def get_model() -> LGBMClassifier:
    """
    Ініціалізація градієнтного бустингу.
    Гіперпараметри встановлені для базового старту, Аліна налаштовуватиме їх через Optuna/GridSearch.
    """
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        # Штраф за пропуск шахрая (False Negative) у 20 разів вищий, ніж за помилку на звичайному клієнті
        scale_pos_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )