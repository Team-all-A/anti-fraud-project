import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing import PolarsDatetimeParser, PolarsImputer

# 1. Адаптер для конвертації форматів
class PolarsToNumpyConverter(BaseEstimator, TransformerMixin):
    """
    Конвертує Polars DataFrame у матрицю NumPy.
    Відфільтровує ідентифікатори та залишкові текстові колонки, 
    які алгоритм машинного навчання не здатний обробити.
    """
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols if drop_cols else []

    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame) -> np.ndarray:
        df = X.clone()
        
        # Видалення колонок-ідентифікаторів
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(cols_to_drop)
        
        # Автоматичне видалення залишкових текстових колонок
        string_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.Utf8]
        df = df.drop(string_cols)
        
        return df.to_numpy()

# 2. Збірка пайплайну
def build_pipeline(model=None) -> Pipeline:
    """
    Формує послідовність обробки від сирих даних до моделі.
    """
    if model is None:
        model = GradientBoostingClassifier()

    pipeline = Pipeline(steps=[
        # Блок 1: Попередня обробка на рівні Polars
        ('dates', PolarsDatetimeParser(date_cols=['timestamp_tr', 'timestamp_reg'])),
        ('imputer', PolarsImputer()),
        
        # Блок 2: Місце для інтеграції фіч аналітика (Custom Transformers Міші)
        # ('target_encoder', PolarsTargetEncoder(...)),
        # ('feature_engineer', PolarsFeatureEngineer(...)),
        
        # Блок 3: Перехідний шар
        ('to_numpy', PolarsToNumpyConverter(drop_cols=['id_user', 'card_mask_hash', 'card_holder', 'email'])),
        
        # Блок 4: Обробка на рівні NumPy / scikit-learn
        ('scaler', StandardScaler()),
        
        # Блок 5: Модель машинного навчання
        ('model', model)
    ])
    return pipeline