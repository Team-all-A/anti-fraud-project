from __future__ import annotations

from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


ID_COLS   = ["id_user", "card_mask_hash", "card_holder", "email"]

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_and_merge(transaction_path: str, users_path: str) -> pl.DataFrame:
    tx_path   = Path(transaction_path)
    usr_path  = Path(users_path)

    if not tx_path.exists():
        raise FileNotFoundError(f"Transactions not found: {tx_path}")
    if not usr_path.exists():
        raise FileNotFoundError(f"Users not found: {usr_path}")

    transactions = pl.read_csv(tx_path,  infer_schema_length=10_000)
    users        = pl.read_csv(usr_path, infer_schema_length=10_000)

    for df, name, col in [(transactions, "transactions", "id_user"),
                          (users,        "users",        "id_user")]:
        if col not in df.columns:
            raise ValueError(f"{name} is missing required column '{col}'")

    # Deduplicate users defensively to prevent accidental row explosion on join.
    if users.get_column("id_user").is_duplicated().any():
        users = users.unique(subset=["id_user"], keep="first")

    return users.join(transactions, on="id_user", how="left")


# ── Imputer (stateful — fit on train fold only) ───────────────────────────────

class PolarsImputer(BaseEstimator, TransformerMixin):
    """
    Fold-safe median/constant imputer.

    fit()      — learns fill values from the training fold only.
    transform() — applies those values to any split (val, test, inference).

    Strategy:
      numeric → median of training fold
      string  → "unknown"
      boolean → False
    """

    def __init__(
        self,
        string_fill:  str  = "unknown",
        bool_fill:    bool = False,
        exclude_cols: list[str] | None = None,
    ):
        self.string_fill  = string_fill
        self.bool_fill    = bool_fill
        self.exclude_cols = set(exclude_cols or ["id_user"])

        self.numeric_fills_: dict[str, float | int] = {}
        self.string_cols_:   list[str] = []
        self.bool_cols_:     list[str] = []

    def fit(self, X: pl.DataFrame, y=None):
        self.numeric_fills_ = {}
        self.string_cols_   = []
        self.bool_cols_     = []

        for col, dtype in zip(X.columns, X.dtypes):
            if col in self.exclude_cols:
                continue

            if dtype in NUMERIC_DTYPES:
                median = X.get_column(col).median()
                self.numeric_fills_[col] = 0 if median is None else median

            elif dtype == pl.Utf8:
                self.string_cols_.append(col)

            elif dtype == pl.Boolean:
                self.bool_cols_.append(col)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        # Strings: strip whitespace, treat empty as null, fill with constant.
        string_exprs = []
        for col in self.string_cols_:
            if col not in df.columns:
                continue
            clean = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
            string_exprs.append(
                pl.when(clean.eq(""))
                .then(None)
                .otherwise(clean)
                .fill_null(self.string_fill)
                .alias(col)
            )
        if string_exprs:
            df = df.with_columns(string_exprs)

        # Booleans.
        bool_exprs = [
            pl.col(col).fill_null(self.bool_fill).alias(col)
            for col in self.bool_cols_
            if col in df.columns
        ]
        if bool_exprs:
            df = df.with_columns(bool_exprs)

        # Numerics: fill with per-column medians learned from train.
        numeric_exprs = [
            pl.col(col).fill_null(fill).alias(col)
            for col, fill in self.numeric_fills_.items()
            if col in df.columns
        ]
        if numeric_exprs:
            df = df.with_columns(numeric_exprs)

        return df


# ── Model frame converter ─────────────────────────

class PolarsToModelFrame(BaseEstimator, TransformerMixin):
    """
    Convert a Polars DataFrame into a stable pandas DataFrame for LightGBM.

    fit()      — records the ordered list of numeric feature columns from X_train.
    transform() — enforces that exact column set and order on any split.
                  Missing columns are filled with 0, extra columns are dropped.
    """

    NUMERIC_DTYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }

    def __init__(self, drop_cols: list[str] | None = None):
        self.drop_cols = drop_cols if drop_cols is not None else list(ID_COLS)
        self.feature_names_: list[str] = []

    def _prepare(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        to_drop = [c for c in self.drop_cols if c in df.columns]
        if to_drop:
            df = df.drop(to_drop)

        bool_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Boolean]
        if bool_cols:
            df = df.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])

        # Keep numeric columns only (string cols handled by target encoder before this).
        df = df.select([c for c, t in zip(df.columns, df.dtypes) if t in self.NUMERIC_DTYPES])

        float_cols = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Float32, pl.Float64)]
        if float_cols:
            df = df.with_columns([
                pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in float_cols
            ])

        if df.width > 0:
            df = df.with_columns([pl.col(c).fill_null(0) for c in df.columns])

        return df

    def fit(self, X: pl.DataFrame, y=None):
        self.feature_names_ = self._prepare(X).columns
        return self

    def transform(self, X: pl.DataFrame) -> pd.DataFrame:
        df = self._prepare(X)

        missing = [c for c in self.feature_names_ if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(0.0).alias(c) for c in missing])

        extra = [c for c in df.columns if c not in self.feature_names_]
        if extra:
            df = df.drop(extra)

        df = df.select(self.feature_names_)
        return pd.DataFrame(df.to_dict(as_series=False), columns=self.feature_names_)
    

# ── Target encoding (stateful) ────────────────────────────────────────

class PolarsTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoder.

    WHEN TO CALL: inside each fold, after imputation.

    fit_transform(X_train, y_train)
        Learns category → fraud-rate mapping from train.
        Uses leave-one-out encoding for train rows so a row does not encode itself.

    transform(X_val)
        Applies the stored mapping (learned from train) to val/test.
        Unseen categories fall back to global_mean_.

    Call order inside fold:
        1. PolarsImputer.fit(X_train)
        2. PolarsImputer.transform(X_train), transform(X_val)
        3. PolarsTargetEncoder.fit_transform(X_train, y_train)   ← LOO on train
        4. PolarsTargetEncoder.transform(X_val)                  ← plain lookup
    """

    def __init__(
        self,
        cat_cols:     list[str],
        smoothing:    float = 20.0,
        drop_original: bool = True,
    ):
        self.cat_cols      = cat_cols
        self.smoothing     = smoothing
        self.drop_original = drop_original

        self.global_mean_: float = 0.0
        self.mappings_: dict[str, pl.DataFrame] = {}

    def fit(self, X: pl.DataFrame, y: np.ndarray):
        self.global_mean_ = float(np.mean(y)) if len(y) > 0 else 0.0
        self.mappings_ = {}

        df = X.with_columns(pl.Series("__target", y))

        for col in self.cat_cols:
            if col not in df.columns:
                continue

            stats = df.group_by(col).agg([
                pl.col("__target").sum().alias("__sum"),
                pl.len().alias("__count"),
            ])

            self.mappings_[col] = stats.with_columns(
                ((pl.col("__sum") + self.smoothing * self.global_mean_) /
                 (pl.col("__count") + self.smoothing))
                .alias(f"{col}_target_enc")
            ).select([col, f"{col}_target_enc"])

        return self

    def fit_transform(self, X: pl.DataFrame, y: np.ndarray) -> pl.DataFrame:
        """LOO encoding for train: each row excludes itself from its category mean."""
        self.fit(X, y)

        df = X.with_columns(pl.Series("__target", y))

        for col in self.cat_cols:
            if col not in df.columns:
                continue

            stats = df.group_by(col).agg([
                pl.col("__target").sum().alias(f"__{col}_sum"),
                pl.len().alias(f"__{col}_count"),
            ])
            df = df.join(stats, on=col, how="left")

            df = df.with_columns(
                pl.when(pl.col(f"__{col}_count") > 1)
                .then(
                    ((pl.col(f"__{col}_sum") - pl.col("__target")) +
                     self.smoothing * self.global_mean_) /
                    ((pl.col(f"__{col}_count") - 1) + self.smoothing)
                )
                .otherwise(self.global_mean_)
                .alias(f"{col}_target_enc")
            )

            drop = [f"__{col}_sum", f"__{col}_count"]
            if self.drop_original:
                drop.append(col)
            df = df.drop(drop)

        return df.drop("__target")

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        for col in self.cat_cols:
            if col not in df.columns or col not in self.mappings_:
                continue

            enc_col = f"{col}_target_enc"
            df = df.join(self.mappings_[col], on=col, how="left")
            df = df.with_columns(pl.col(enc_col).fill_null(self.global_mean_).alias(enc_col))

            if self.drop_original:
                df = df.drop(col)

        return df