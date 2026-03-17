import numpy as np
from sklearn.metrics import confusion_matrix

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray, cost_fp: float = 50.0, cost_fn: float = 500.0) -> tuple:
    """
    Пошук оптимального порогу відсікання на основі матриці витрат.
    cost_fp: Вартість блокування легітимного клієнта (втрата LTV, навантаження на підтримку).
    cost_fn: Вартість пропущеного шахрая (втрачені кошти, штрафи, чарджбеки).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    min_cost = float('inf')
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Обчислення матриці помилок
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Розрахунок загальних фінансових втрат
        current_cost = (fp * cost_fp) + (fn * cost_fn)
        
        if current_cost < min_cost:
            min_cost = current_cost
            best_threshold = thresh
            best_metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            
    return best_threshold, min_cost, best_metrics

def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Конвертація ймовірностей у бінарні класи за знайденим порогом."""
    return (y_proba >= threshold).astype(int)