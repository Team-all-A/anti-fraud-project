from __future__ import annotations

import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.features import (
    PolarsLogicalFeatures,
    PolarsTargetEncoder,
    PolarsUserAggregator,
)
from src.model import get_model
from src.preprocessing import (
    DATE_COLS,
    ID_COLS,
    PolarsDatetimeParser,
    PolarsImputer,
)


TARGET_ENCODE_COLS = [
    "transaction_type",
    "currency",
    "card_brand",
    "card_type",
    "card_country",
    "payment_country",
    "gender",
    "reg_country",
    "traffic_type",
]


class PolarsToModelFrame(BaseEstimator, TransformerMixin):
    """
    Convert a Polars DataFrame into a stable pandas DataFrame for LightGBM.

    Why pandas here:
    - keeps feature names
    - avoids the LightGBM warning about invalid feature names
    - still lets the rest of the pipeline stay in Polars
    """

    def __init__(self, drop_cols: list[str] | None = None):
        self.drop_cols = drop_cols if drop_cols is not None else list(ID_COLS)
        self.feature_names_: list[str] = []

    @staticmethod
    def _is_numeric_dtype(dtype: pl.DataType) -> bool:
        return dtype in {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

    def _prepare_frame(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        cols_to_drop = [col for col in self.drop_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        bool_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype == pl.Boolean
        ]
        if bool_cols:
            df = df.with_columns(
                [pl.col(col).cast(pl.Int8).alias(col) for col in bool_cols]
            )

        numeric_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if self._is_numeric_dtype(dtype)
        ]
        df = df.select(numeric_cols)

        float_cols = [
            col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Float32, pl.Float64)
        ]
        if float_cols:
            df = df.with_columns([
                pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in float_cols
            ])

        if df.width > 0:
            df = df.with_columns([
                pl.col(col).fill_null(0).alias(col)
                for col in df.columns
            ])

        return df

    def fit(self, X: pl.DataFrame, y=None):
        df = self._prepare_frame(X)
        self.feature_names_ = df.columns
        return self

    def transform(self, X: pl.DataFrame) -> pd.DataFrame:
        df = self._prepare_frame(X)

        missing_cols = [col for col in self.feature_names_ if col not in df.columns]
        if missing_cols:
            df = df.with_columns([
                pl.lit(0.0).alias(col) for col in missing_cols
            ])

        extra_cols = [col for col in df.columns if col not in self.feature_names_]
        if extra_cols:
            df = df.drop(extra_cols)

        df = df.select(self.feature_names_)

        return df.to_pandas()


def build_pipeline(
    model=None,
    target_encode_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> Pipeline:
    if model is None:
        model = get_model()

    target_encode_cols = (
        target_encode_cols if target_encode_cols is not None else TARGET_ENCODE_COLS
    )
    drop_cols = drop_cols if drop_cols is not None else list(ID_COLS)

    return Pipeline(
        steps=[
            ("dates", PolarsDatetimeParser(date_cols=DATE_COLS)),
            ("imputer", PolarsImputer()),
            ("logical_features", PolarsLogicalFeatures()),
            ("user_aggregator", PolarsUserAggregator(user_col="id_user")),
            ("target_encoder", PolarsTargetEncoder(cat_cols=target_encode_cols)),
            ("to_model_frame", PolarsToModelFrame(drop_cols=drop_cols)),
            ("model", model),
        ]
    )