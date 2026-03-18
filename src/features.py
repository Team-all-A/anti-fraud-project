import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def build_user_level_features(df: pl.DataFrame) -> pl.DataFrame:
    agg_exprs = [
        pl.len().alias("tx_count"),
    ]

    if "amount" in df.columns:
        agg_exprs += [
            pl.col("amount").sum().alias("amount_sum"),
            pl.col("amount").mean().alias("amount_mean"),
            pl.col("amount").max().alias("amount_max"),
            pl.col("amount").std().alias("amount_std"),
        ]

    if "status" in df.columns:
        agg_exprs += [
            (pl.col("status") == "fail").sum().alias("fail_count"),
            (pl.col("status") == "success").sum().alias("success_count"),
        ]

    user_level = df.group_by("id_user").agg(agg_exprs)

    user_static = df.select([c for c in df.columns if c not in ["timestamp_tr"]]).unique(subset=["id_user"])

    result = user_static.join(user_level, on="id_user", how="left")
    return result

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
    """
    Stateful user-level агрегатор.
    fit()   -> рахує історію користувачів на train fold і зберігає її
    transform() -> приєднує вже обчислені агрегати до будь-якого нового X
    """

    def __init__(self, user_col: str = "id_user"):
        self.user_col = user_col
        self.user_stats_ = None
        self.defaults_ = {}
        self.feature_cols_ = []

    def fit(self, X: pl.DataFrame, y=None):
        df = X.clone()

        if self.user_col not in df.columns:
            self.user_stats_ = None
            self.defaults_ = {}
            self.feature_cols_ = []
            return self

        agg_exprs = []

        # Базові агрегати
        agg_exprs.append(pl.len().alias("user_tx_count"))

        if "status" in df.columns:
            agg_exprs.extend([
                (pl.col("status") == "fail").sum().alias("user_fail_count"),
                (pl.col("status") == "success").sum().alias("user_success_count"),
            ])

        if "amount" in df.columns:
            agg_exprs.extend([
                pl.col("amount").mean().alias("user_amount_mean"),
                pl.col("amount").sum().alias("user_amount_sum"),
                pl.col("amount").max().alias("user_amount_max"),
                pl.col("amount").std().alias("user_amount_std"),
            ])

        if "currency" in df.columns:
            agg_exprs.append(
                pl.col("currency").n_unique().alias("user_currency_nunique")
            )

        if "payment_country" in df.columns:
            agg_exprs.append(
                pl.col("payment_country").n_unique().alias("user_payment_country_nunique")
            )

        if "card_country" in df.columns:
            agg_exprs.append(
                pl.col("card_country").n_unique().alias("user_card_country_nunique")
            )

        if "transaction_type" in df.columns:
            agg_exprs.append(
                pl.col("transaction_type").n_unique().alias("user_tx_type_nunique")
            )

        self.user_stats_ = df.group_by(self.user_col).agg(agg_exprs)

        self.feature_cols_ = [
            c for c in self.user_stats_.columns
            if c != self.user_col
        ]

        for col in self.feature_cols_:
            dtype = self.user_stats_[col].dtype

            if dtype in (pl.Float32, pl.Float64):
                val = self.user_stats_[col].mean()
                self.defaults_[col] = 0.0 if val is None else float(val)

            elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                val = self.user_stats_[col].median()
                self.defaults_[col] = 0 if val is None else int(val)

            else:
                self.defaults_[col] = 0

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        if self.user_stats_ is None or self.user_col not in df.columns:
            return df

        df = df.join(self.user_stats_, on=self.user_col, how="left")

        fill_exprs = []
        for col, default_value in self.defaults_.items():
            if col in df.columns:
                fill_exprs.append(pl.col(col).fill_null(default_value).alias(col))

        if fill_exprs:
            df = df.with_columns(fill_exprs)

        # Похідні ratio-фічі
        if "user_fail_count" in df.columns and "user_tx_count" in df.columns:
            df = df.with_columns(
                (pl.col("user_fail_count") / pl.col("user_tx_count").clip(lower_bound=1))
                .alias("user_fail_rate")
            )

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