from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

ID_COLS = ["id_user", "card_mask_hash", "card_holder", "email"]
DATE_COLS = ["timestamp_tr", "timestamp_reg"]

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


def _validate_required_columns(df: pl.DataFrame, required: list[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}. "
            f"Available columns: {df.columns}"
        )


def load_and_merge(transaction_path: str, users_path: str) -> pl.DataFrame:
    """
    Load users and transactions, then merge them by user id.

    Important design choice:
    we keep a LEFT join from users to transactions so every user remains in the
    dataset even if they have no transactions. This matches the current project
    structure, where user metadata is the base table and transactions expand it.

    Returns:
        Polars DataFrame containing one row per joined user-transaction record.
    """
    transaction_path = Path(transaction_path)
    users_path = Path(users_path)

    if not transaction_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {transaction_path}")
    if not users_path.exists():
        raise FileNotFoundError(f"Users file not found: {users_path}")

    transactions = pl.read_csv(transaction_path, infer_schema_length=10_000,)
    users = pl.read_csv(users_path, infer_schema_length=10_000,)

    _validate_required_columns(transactions, ["id_user"], "transactions")
    _validate_required_columns(users, ["id_user"], "users")

    # Defensive deduplication of user table.
    # If duplicate users exist, the join can explode unexpectedly.
    if users.get_column("id_user").is_duplicated().any():
        users = users.unique(subset=["id_user"], keep="first")

    df = users.join(transactions, on="id_user", how="left")
    return df


class PolarsDatetimeParser(BaseEstimator, TransformerMixin):
    """
    Parse raw datetime columns and add simple calendar features.

    This transformer is stateless, so it is safe to use before splitting logic
    as long as it only performs row-wise parsing. In our project it still runs
    inside the fold pipeline for consistency.
    """

    def __init__(self, date_cols: list[str]):
        self.date_cols = date_cols

    def fit(self, X: pl.DataFrame, y=None):
        return self

    def _parse_datetime_column(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        if col not in df.columns:
            return df

        dtype = df.schema[col]

        if dtype == pl.Datetime:
            parsed_expr = pl.col(col).dt.replace_time_zone(None).alias(col)
        elif dtype == pl.Date:
            parsed_expr = pl.col(col).cast(pl.Datetime).alias(col)
        else:
            parsed_expr = (
                pl.col(col)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.to_datetime(time_zone="UTC", strict=False)
                .dt.replace_time_zone(None)
                .alias(col)
            )

        df = df.with_columns(parsed_expr)

        # Add a missingness flag because "date absent" can itself be informative.
        # Keep it generic here; downstream feature engineering may use it.
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_missing")
        )

        # Calendar decomposition. These are safe row-level features.
        df = df.with_columns([
            pl.col(col).dt.year().alias(f"{col}_year"),
            pl.col(col).dt.month().alias(f"{col}_month"),
            pl.col(col).dt.day().alias(f"{col}_day"),
            pl.col(col).dt.hour().alias(f"{col}_hour"),
            pl.col(col).dt.weekday().alias(f"{col}_weekday"),
        ])

        return df

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        for col in self.date_cols:
            df = self._parse_datetime_column(df, col)

        return df


class PolarsImputer(BaseEstimator, TransformerMixin):
    """
    Fold-safe imputer.

    fit():
        learns fill values from the training fold only

    transform():
        applies those values to validation / test / inference data

    Strategy:
    - numeric columns -> median
    - string columns  -> "unknown"
    - boolean columns -> False

    Raw datetime columns are intentionally left nullable.
    Their derived numeric components are handled by numeric imputation after
    PolarsDatetimeParser runs.
    """

    def __init__(
        self,
        string_fill_value: str = "unknown",
        bool_fill_value: bool = False,
        exclude_cols: list[str] | None = None,
    ):
        self.string_fill_value = string_fill_value
        self.bool_fill_value = bool_fill_value
        self.exclude_cols = exclude_cols if exclude_cols is not None else ["id_user"]

        self.numeric_fill_values_: dict[str, float | int] = {}
        self.string_cols_: list[str] = []
        self.bool_cols_: list[str] = []

    def fit(self, X: pl.DataFrame, y=None):
        self.numeric_fill_values_ = {}
        self.string_cols_ = []
        self.bool_cols_ = []

        for col, dtype in zip(X.columns, X.dtypes):
            if col in self.exclude_cols:
                continue

            if dtype in NUMERIC_DTYPES:
                median_value = X.get_column(col).median()
                self.numeric_fill_values_[col] = 0 if median_value is None else median_value

            elif dtype == pl.Utf8:
                self.string_cols_.append(col)

            elif dtype == pl.Boolean:
                self.bool_cols_.append(col)

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        # Normalize strings a bit before filling:
        # trim whitespace, convert empty strings to null, then fill with default.
        string_exprs = []
        for col in self.string_cols_:
            if col in df.columns:
                string_exprs.append(
                    pl.when(
                        pl.col(col)
                        .cast(pl.Utf8, strict=False)
                        .str.strip_chars()
                        .eq("")
                    )
                    .then(None)
                    .otherwise(
                        pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
                    )
                    .fill_null(self.string_fill_value)
                    .alias(col)
                )

        if string_exprs:
            df = df.with_columns(string_exprs)

        bool_exprs = []
        for col in self.bool_cols_:
            if col in df.columns:
                bool_exprs.append(
                    pl.col(col).fill_null(self.bool_fill_value).alias(col)
                )

        if bool_exprs:
            df = df.with_columns(bool_exprs)

        numeric_exprs = []
        for col, fill_value in self.numeric_fill_values_.items():
            if col in df.columns:
                numeric_exprs.append(
                    pl.col(col).fill_null(fill_value).alias(col)
                )

        if numeric_exprs:
            df = df.with_columns(numeric_exprs)

        return df