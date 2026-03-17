import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PolarsLogicalFeatures(BaseEstimator, TransformerMixin):
    """Генерація безстанових логічних прапорців та порівнянь."""
    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        
        # 1. Фільтри статусу та безпеки
        if 'error_group' in df.columns:
            df = df.with_columns(
                (pl.col('error_group') == 'antifraud').cast(pl.Int8).alias('blocked_by_antifraud')
            )
        if 'status' in df.columns:
            df = df.with_columns(
                (pl.col('status') == 'fail').cast(pl.Int8).alias('is_failed_tx')
            )
            
        # 2. Географічні аномалії
        if 'card_country' in df.columns and 'payment_country' in df.columns:
            df = df.with_columns(
                (pl.col('card_country') != pl.col('payment_country')).cast(pl.Int8).alias('card_payment_mismatch')
            )
        if 'reg_country' in df.columns and 'card_country' in df.columns:
            df = df.with_columns(
                (pl.col('reg_country') != pl.col('card_country')).cast(pl.Int8).alias('reg_card_mismatch')
            )
            
        # 3. Фінансові тригери
        if 'amount' in df.columns:
            df = df.with_columns(
                (pl.col('amount') > 500).cast(pl.Int8).alias('is_large_tx')
            )
            
        return df

class PolarsUserAggregator(BaseEstimator, TransformerMixin):
    """Обчислення агрегованої історії користувача в межах поточного вікна даних."""
    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        
        if 'id_user' in df.columns:
            # Загальна кількість транзакцій користувача
            tx_count = df.group_by('id_user').agg(pl.len().alias('user_tx_count'))
            df = df.join(tx_count, on='id_user', how='left')
            
            # Кількість неуспішних транзакцій користувача
            if 'status' in df.columns:
                fail_count = (
                    df.filter(pl.col('status') == 'fail')
                      .group_by('id_user')
                      .agg(pl.len().alias('user_fail_count'))
                )
                df = df.join(fail_count, on='id_user', how='left')
                df = df.with_columns(pl.col('user_fail_count').fill_null(0))
                
        return df

class PolarsTargetEncoder(BaseEstimator, TransformerMixin):
    """Цільове кодування категоріальних змінних із збереженням стану (запобігання Data Leakage)."""
    def __init__(self, cat_cols: list):
        self.cat_cols = cat_cols
        self.mappings = {}
        self.global_means = {}

    def fit(self, X: pl.DataFrame, y: np.ndarray):
        df = X.with_columns(pl.Series("target", y))
        
        for col in self.cat_cols:
            if col in df.columns:
                # Збереження глобального середнього для невідомих категорій у тестовій вибірці
                self.global_means[col] = df["target"].mean()
                
                # Обчислення ймовірності шахрайства для кожної категорії
                mapping = (
                    df.group_by(col)
                    .agg(pl.col("target").mean().alias(f"{col}_target_enc"))
                )
                self.mappings[col] = mapping
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        for col in self.cat_cols:
            if col in df.columns and col in self.mappings:
                df = df.join(self.mappings[col], on=col, how="left")
                df = df.with_columns(pl.col(f"{col}_target_enc").fill_null(self.global_means.get(col, 0)))
                df = df.drop(col) # Видалення оригінальної текстової колонки
        return df