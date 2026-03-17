import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.preprocessing import load_and_merge
from src.pipeline import build_pipeline

# --- ЗОНА ВІДПОВІДАЛЬНОСТІ АЛІНИ (Моделер) ---
# Розкоментувати та налаштувати гіперпараметри після тестів
# from lightgbm import LGBMClassifier

print("Downloading data...")
# 1. Завантаження сирих даних (без застосування статистичних трансформацій)
df_train = load_and_merge('data/train_transactions.csv', 'data/train_users.csv')
df_test = load_and_merge('data/test_transactions.csv', 'data/test_users.csv')

# 2. Підготовка базових масивів
y = df_train['is_fraud'].to_numpy()
X = df_train.drop('is_fraud')
X_test = df_test  # Тестовий набір без міток не зазнає розбиття

# 3. Налаштування Stratified K-Fold
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Масиви для збереження результатів
oof_predictions = np.zeros(len(X))  # Out-of-Fold прогнози для аналізу Дмитра
test_predictions = np.zeros(len(X_test)) # Фінальні прогнози для submission

print(f"Starting {N_SPLITS}-Fold Stratified Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"\n--- Fold {fold + 1} ---")

    # 4. Ізольоване розбиття Polars DataFrame за індексами NumPy
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    # --- ЗОНА ВІДПОВІДАЛЬНОСТІ АЛІНИ (Моделер) ---
    # Ініціалізація алгоритму з урахуванням дисбалансу класів (scale_pos_weight)
    # model = LGBMClassifier(n_estimators=300, scale_pos_weight=20, random_state=42, n_jobs=-1)
    # pipeline = build_pipeline(model=model)
    
    pipeline = build_pipeline() # Тимчасова заглушка

    # --- ЗОНА ВІДПОВІДАЛЬНОСТІ МІШІ (Аналітик) ---
    # Фічі та фільтри Міші працюють під капотом виклику fit/transform
    # Їхня логіка зашита в PolarsFeatureEngineer та PolarsTargetEncoder у файлі pipeline.py

    # 5. Навчання пайплайну виключно на K-1 фолдах
    pipeline.fit(X_train_fold, y_train_fold)

    # 6. Валідація на K-тому фолді (без витоку даних)
    val_preds_proba = pipeline.predict_proba(X_val_fold)[:, 1]
    val_preds_binary = pipeline.predict(X_val_fold)
    
    oof_predictions[val_idx] = val_preds_proba
    fold_f1 = f1_score(y_val_fold, val_preds_binary)
    print(f"Fold {fold + 1} F1-Score: {fold_f1:.4f}")

    # 7. Генерація прогнозів для реального тестового файлу
    # Результат усереднюється між усіма фолдами для підвищення стабільності
    test_predictions += pipeline.predict_proba(X_test)[:, 1] / N_SPLITS

# --- ЗОНА ВІДПОВІДАЛЬНОСТІ ДМИТРА (Бізнес-аналітик) ---
# Дмитро має використовувати масив oof_predictions та y для розрахунку матриці витрат (Cost Matrix)
# та пошуку оптимального порогу відсікання (Threshold Tuning), відмінного від стандартних 0.5.

print("\nGenerating final submission...")
# Збереження сирих ймовірностей. Бізнес-логіка Дмитра згодом перетворить їх на 0/1 за потреби.
submission = pl.DataFrame({
    'id_user': df_test['id_user'],
    'is_fraud': test_predictions
})
submission.write_csv('submission.csv')
print("Ready! submission.csv is saved.")