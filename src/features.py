import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
FLOAT_DTYPES = {pl.Float32, pl.Float64}
NUMERIC_DTYPES = INT_DTYPES | FLOAT_DTYPES


def _safe_div_expr(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    """
    Safe division for Polars expressions.
    Returns 0.0 when the denominator is null or <= 0.
    """
    return (
        pl.when(denominator.fill_null(0) > 0)
        .then(numerator / denominator)
        .otherwise(0.0)
        .alias(alias)
    )


def build_user_level_features(
    df: pl.DataFrame,
    user_col: str = "id_user",
    fail_rate_threshold: float = 0.80,
    unique_cards_threshold: int = 3,
    avg_hours_between_tx_threshold: float = 2.0,
    micro_payment_threshold: float = 1.0,
) -> pl.DataFrame:
    """
    Build train-fold user aggregates from transaction-level rows.

    Important:
    this function is safe only when called on the training fold inside fit().
    The resulting statistics can then be joined onto validation / test rows.
    """
    if user_col not in df.columns:
        return pl.DataFrame()

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("user_tx_count"),
    ]

    if "amount" in df.columns:
        agg_exprs.extend([
            pl.col("amount").sum().alias("user_amount_sum"),
            pl.col("amount").mean().alias("user_amount_mean"),
            pl.col("amount").max().alias("user_amount_max"),
            pl.col("amount").min().alias("user_amount_min"),
            pl.col("amount").std().alias("user_amount_std"),
        ])

    if "status" in df.columns:
        agg_exprs.extend([
            (pl.col("status") == "fail").sum().alias("user_fail_count"),
            (pl.col("status") == "success").sum().alias("user_success_count"),
        ])

    if "blocked_by_antifraud" in df.columns:
        agg_exprs.append(pl.col("blocked_by_antifraud").max().alias("user_has_antifraud_error"))
    if "has_fraud_error_group" in df.columns:
        agg_exprs.append(pl.col("has_fraud_error_group").max().alias("user_has_fraud_error"))
    if "insufficient_funds_error" in df.columns:
        agg_exprs.append(pl.col("insufficient_funds_error").sum().alias("user_insufficient_funds_count"))
    if "do_not_honor_error" in df.columns:
        agg_exprs.append(pl.col("do_not_honor_error").sum().alias("user_do_not_honor_count"))
    if "cvv_error" in df.columns:
        agg_exprs.append(pl.col("cvv_error").sum().alias("user_cvv_error_count"))
    if "three_ds_error" in df.columns:
        agg_exprs.append(pl.col("three_ds_error").sum().alias("user_3ds_error_count"))

    if "error_group" in df.columns:
        agg_exprs.append(
            pl.col("error_group").drop_nulls().n_unique().alias("user_error_diversity_count")
        )

    if "currency" in df.columns:
        agg_exprs.append(pl.col("currency").n_unique().alias("user_currency_nunique"))
    if "payment_country" in df.columns:
        agg_exprs.append(pl.col("payment_country").n_unique().alias("user_payment_country_nunique"))
    if "card_country" in df.columns:
        agg_exprs.append(pl.col("card_country").n_unique().alias("user_card_country_nunique"))
    if "transaction_type" in df.columns:
        agg_exprs.append(pl.col("transaction_type").n_unique().alias("user_tx_type_nunique"))
    if "card_mask_hash" in df.columns:
        agg_exprs.append(pl.col("card_mask_hash").n_unique().alias("user_unique_cards_count"))
    if "card_holder" in df.columns:
        agg_exprs.append(
            pl.col("card_holder").drop_nulls().n_unique().alias("user_unique_card_holders_count")
        )

    if "card_payment_mismatch" in df.columns:
        agg_exprs.append(pl.col("card_payment_mismatch").max().alias("user_has_card_payment_mismatch"))
    if "reg_payment_mismatch" in df.columns:
        agg_exprs.append(pl.col("reg_payment_mismatch").max().alias("user_has_reg_payment_mismatch"))
    if "triple_geo_mismatch" in df.columns:
        agg_exprs.append(pl.col("triple_geo_mismatch").max().alias("user_has_triple_geo_mismatch"))

    if "is_credit_card" in df.columns:
        agg_exprs.append(pl.col("is_credit_card").sum().alias("user_credit_card_tx_count"))
    if "is_visa" in df.columns:
        agg_exprs.append(pl.col("is_visa").sum().alias("user_visa_tx_count"))
    if "null_card_holder_non_gpay" in df.columns:
        agg_exprs.append(
            pl.col("null_card_holder_non_gpay").sum().alias("user_null_holder_non_gpay_count")
        )

    if "is_card_init" in df.columns:
        agg_exprs.append(pl.col("is_card_init").sum().alias("user_card_init_count"))
    if "is_card_recurring" in df.columns:
        agg_exprs.append(pl.col("is_card_recurring").sum().alias("user_card_recurring_count"))
    if "is_google_pay" in df.columns:
        agg_exprs.append(pl.col("is_google_pay").sum().alias("user_google_pay_count"))
    if "is_night_tx" in df.columns:
        agg_exprs.append(pl.col("is_night_tx").sum().alias("user_night_tx_count"))
    if "is_eur_tx" in df.columns:
        agg_exprs.append(pl.col("is_eur_tx").sum().alias("user_eur_tx_count"))
    if "is_micro_amount" in df.columns:
        agg_exprs.append(pl.col("is_micro_amount").sum().alias("user_micro_tx_count"))

    if "timestamp_tr" in df.columns:
        agg_exprs.extend([
            pl.col("timestamp_tr").min().alias("user_first_tx_ts"),
            pl.col("timestamp_tr").max().alias("user_last_tx_ts"),
        ])

    if "timestamp_reg" in df.columns:
        agg_exprs.append(pl.col("timestamp_reg").min().alias("user_reg_ts"))

    stats = df.group_by(user_col).agg(agg_exprs)

    numeric_fill_exprs = [
        pl.col(col).fill_null(0).alias(col)
        for col, dtype in zip(stats.columns, stats.dtypes)
        if col != user_col and dtype in NUMERIC_DTYPES
    ]
    if numeric_fill_exprs:
        stats = stats.with_columns(numeric_fill_exprs)

    derived_exprs: list[pl.Expr] = []

    if {"user_fail_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_fail_count"),
                pl.col("user_tx_count"),
                "user_fail_rate",
            )
        )

    if {"user_success_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_success_count"),
                pl.col("user_tx_count"),
                "user_success_rate",
            )
        )

    if {"user_credit_card_tx_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_credit_card_tx_count"),
                pl.col("user_tx_count"),
                "user_credit_card_rate",
            )
        )

    if {"user_visa_tx_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_visa_tx_count"),
                pl.col("user_tx_count"),
                "user_visa_rate",
            )
        )

    if {"user_null_holder_non_gpay_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_null_holder_non_gpay_count"),
                pl.col("user_tx_count"),
                "user_null_holder_rate",
            )
        )

    if {"user_night_tx_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_night_tx_count"),
                pl.col("user_tx_count"),
                "user_night_tx_rate",
            )
        )

    if {"user_eur_tx_count", "user_tx_count"}.issubset(stats.columns):
        derived_exprs.append(
            _safe_div_expr(
                pl.col("user_eur_tx_count"),
                pl.col("user_tx_count"),
                "user_eur_tx_rate",
            )
        )

    if {"user_card_init_count", "user_card_recurring_count"}.issubset(stats.columns):
        derived_exprs.append(
            (
                pl.col("user_card_init_count") /
                (pl.col("user_card_recurring_count") + 1)
            ).alias("user_init_to_recurring_ratio")
        )

    if {"user_last_tx_ts", "user_first_tx_ts"}.issubset(stats.columns):
        derived_exprs.append(
            (
                (pl.col("user_last_tx_ts") - pl.col("user_first_tx_ts")).dt.total_seconds() / 3600
            ).alias("user_tx_timespan_hours")
        )

    if {"user_first_tx_ts", "user_reg_ts"}.issubset(stats.columns):
        derived_exprs.append(
            (
                ((pl.col("user_first_tx_ts") - pl.col("user_reg_ts")).dt.total_seconds() / 3600)
                .clip(lower_bound=0)
            ).alias("user_reg_to_first_tx_hours")
        )
        derived_exprs.append(
            (
                ((pl.col("user_first_tx_ts") - pl.col("user_reg_ts")).dt.total_seconds() / 86400)
                .clip(lower_bound=0)
            ).alias("user_account_age_days")
        )

    if derived_exprs:
        stats = stats.with_columns(derived_exprs)

    post_exprs: list[pl.Expr] = []

    if {"user_tx_timespan_hours", "user_tx_count"}.issubset(stats.columns):
        post_exprs.append(
            pl.when(pl.col("user_tx_count") > 1)
            .then(pl.col("user_tx_timespan_hours") / (pl.col("user_tx_count") - 1))
            .otherwise(0.0)
            .alias("user_avg_hours_between_tx")
        )

    if {"user_unique_cards_count", "user_account_age_days"}.issubset(stats.columns):
        post_exprs.append(
            pl.when(pl.col("user_account_age_days") > 0)
            .then(pl.col("user_unique_cards_count") / pl.col("user_account_age_days"))
            .otherwise(pl.col("user_unique_cards_count").cast(pl.Float64))
            .alias("user_cards_per_day")
        )

    if post_exprs:
        stats = stats.with_columns(post_exprs)

    rule_exprs: list[pl.Expr] = []

    if {"user_amount_min", "user_fail_count"}.issubset(stats.columns):
        rule_exprs.append(
            (
                (pl.col("user_amount_min") < micro_payment_threshold) &
                (pl.col("user_fail_count") >= 2)
            ).cast(pl.Int8).alias("user_micro_payment_flag")
        )

    if {"user_unique_cards_count", "user_fail_rate", "user_avg_hours_between_tx"}.issubset(stats.columns):
        rule_exprs.append(
            (
                (pl.col("user_unique_cards_count") >= unique_cards_threshold) &
                (pl.col("user_fail_rate") > fail_rate_threshold) &
                (pl.col("user_avg_hours_between_tx") < avg_hours_between_tx_threshold)
            ).cast(pl.Int8).alias("user_card_testing_flag")
        )

    if rule_exprs:
        stats = stats.with_columns(rule_exprs)

    score_parts = []
    for col in [
        "user_has_antifraud_error",
        "user_has_fraud_error",
        "user_has_triple_geo_mismatch",
        "user_micro_payment_flag",
        "user_card_testing_flag",
    ]:
        if col in stats.columns:
            score_parts.append(pl.col(col).fill_null(0).cast(pl.Int64))

    if "user_unique_cards_count" in stats.columns:
        score_parts.append((pl.col("user_unique_cards_count") >= unique_cards_threshold).cast(pl.Int64))
    if "user_fail_rate" in stats.columns:
        score_parts.append((pl.col("user_fail_rate") > fail_rate_threshold).cast(pl.Int64))

    if score_parts:
        stats = stats.with_columns(sum(score_parts).cast(pl.Int16).alias("user_rule_score"))

    temp_cols_to_drop = [
        col for col in ["user_first_tx_ts", "user_last_tx_ts", "user_reg_ts"]
        if col in stats.columns
    ]
    if temp_cols_to_drop:
        stats = stats.drop(temp_cols_to_drop)

    return stats


class PolarsLogicalFeatures(BaseEstimator, TransformerMixin):
    """
    Stateless row-level anti-fraud features.

    These features are safe to create before aggregation because each one depends
    only on the current row (or on date differences within the same row).
    """

    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        exprs: list[pl.Expr] = []

        if "email" in df.columns:
            exprs.append(
                pl.col("email")
                .cast(pl.Utf8)
                .str.extract(r"@(.+)$", 1)
                .fill_null("unknown")
                .alias("email_domain")
            )
            exprs.append(pl.col("email").is_null().cast(pl.Int8).alias("email_is_missing"))

        if "card_holder" in df.columns:
            exprs.append(pl.col("card_holder").is_null().cast(pl.Int8).alias("card_holder_is_missing"))

        if "amount" in df.columns:
            exprs.extend([
                (pl.col("amount") > 500).cast(pl.Int8).alias("is_large_amount"),
                (pl.col("amount") < 1.0).cast(pl.Int8).alias("is_micro_amount"),
                pl.col("amount").log1p().alias("amount_log1p"),
            ])

        if "status" in df.columns:
            exprs.extend([
                (pl.col("status") == "fail").cast(pl.Int8).alias("is_failed_tx"),
                (pl.col("status") == "success").cast(pl.Int8).alias("is_success_tx"),
            ])

        if "error_group" in df.columns:
            exprs.extend([
                (pl.col("error_group") == "antifraud").cast(pl.Int8).alias("blocked_by_antifraud"),
                (pl.col("error_group") == "fraud").cast(pl.Int8).alias("has_fraud_error_group"),
                pl.col("error_group").is_not_null().cast(pl.Int8).alias("has_any_error_group"),
            ])

        if "error_desc" in df.columns:
            err = pl.col("error_desc").cast(pl.Utf8).str.to_lowercase().fill_null("")
            exprs.extend([
                err.str.contains("insufficient funds", literal=True).cast(pl.Int8).alias("insufficient_funds_error"),
                err.str.contains("do not honor", literal=True).cast(pl.Int8).alias("do_not_honor_error"),
                err.str.contains("cvv", literal=True).cast(pl.Int8).alias("cvv_error"),
                err.str.contains("3ds", literal=True).cast(pl.Int8).alias("three_ds_error"),
            ])

        if {"card_country", "payment_country"}.issubset(df.columns):
            exprs.append(
                (pl.col("card_country") != pl.col("payment_country")).cast(pl.Int8).alias("card_payment_mismatch")
            )

        if {"reg_country", "payment_country"}.issubset(df.columns):
            exprs.append(
                (pl.col("reg_country") != pl.col("payment_country")).cast(pl.Int8).alias("reg_payment_mismatch")
            )

        if {"reg_country", "card_country", "payment_country"}.issubset(df.columns):
            exprs.append(
                (
                    (pl.col("reg_country") != pl.col("card_country")) &
                    (pl.col("reg_country") != pl.col("payment_country"))
                ).cast(pl.Int8).alias("triple_geo_mismatch")
            )

        if "card_type" in df.columns:
            exprs.append((pl.col("card_type") == "CREDIT").cast(pl.Int8).alias("is_credit_card"))

        if "card_brand" in df.columns:
            exprs.append((pl.col("card_brand") == "VISA").cast(pl.Int8).alias("is_visa"))

        if "transaction_type" in df.columns:
            exprs.extend([
                (pl.col("transaction_type") == "card_init").cast(pl.Int8).alias("is_card_init"),
                (pl.col("transaction_type") == "card_recurring").cast(pl.Int8).alias("is_card_recurring"),
                (pl.col("transaction_type") == "google-pay").cast(pl.Int8).alias("is_google_pay"),
                (pl.col("transaction_type") != "google-pay").cast(pl.Int8).alias("is_non_google_pay"),
            ])

        if {"transaction_type", "card_holder"}.issubset(df.columns):
            exprs.append(
                (
                    (pl.col("transaction_type") != "google-pay") &
                    pl.col("card_holder").is_null()
                ).cast(pl.Int8).alias("null_card_holder_non_gpay")
            )

        if "currency" in df.columns:
            exprs.append((pl.col("currency") == "EUR").cast(pl.Int8).alias("is_eur_tx"))

        if "timestamp_tr" in df.columns:
            exprs.extend([
                pl.col("timestamp_tr").dt.hour().is_between(0, 5, closed="both").cast(pl.Int8).alias("is_night_tx"),
                pl.col("timestamp_tr").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend_tx"),
            ])

        if {"timestamp_tr", "timestamp_reg"}.issubset(df.columns):
            exprs.extend([
                (
                    ((pl.col("timestamp_tr") - pl.col("timestamp_reg")).dt.total_seconds() / 3600)
                    .clip(lower_bound=0)
                ).alias("hours_since_registration"),
                (
                    ((pl.col("timestamp_tr") - pl.col("timestamp_reg")).dt.total_seconds() / 86400)
                    .clip(lower_bound=0)
                ).alias("days_since_registration"),
            ])

        if not exprs:
            return df

        return df.with_columns(exprs)


class PolarsUserAggregator(BaseEstimator, TransformerMixin):
    """
    Stateful user-level aggregator.

    fit() computes user history using only the training fold.
    transform() joins those learned user statistics onto any new frame.
    """

    def __init__(self, user_col: str = "id_user"):
        self.user_col = user_col
        self.user_stats_: pl.DataFrame | None = None
        self.defaults_: dict[str, float | int] = {}
        self.feature_cols_: list[str] = []

    def fit(self, X: pl.DataFrame, y=None):
        df = X.clone()

        if self.user_col not in df.columns:
            self.user_stats_ = None
            self.defaults_ = {}
            self.feature_cols_ = []
            return self

        self.user_stats_ = build_user_level_features(df, user_col=self.user_col)

        self.feature_cols_ = [
            col for col in self.user_stats_.columns
            if col != self.user_col
        ]

        self.defaults_ = {}
        for col in self.feature_cols_:
            series = self.user_stats_.get_column(col)
            dtype = series.dtype

            if dtype in FLOAT_DTYPES:
                value = series.median()
                self.defaults_[col] = 0.0 if value is None else float(value)
            elif dtype in INT_DTYPES:
                value = series.median()
                self.defaults_[col] = 0 if value is None else int(value)
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

        return df


class PolarsTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoding with a leakage-safe training transform.

    - fit(): learns category statistics on the training fold only
    - transform(): applies stored mappings to validation / test / inference data
    - fit_transform(): generates leave-one-out style encodings for the training
      fold so a row does not directly encode itself
    """

    def __init__(
        self,
        cat_cols: list[str],
        smoothing: float = 20.0,
        drop_original: bool = True,
    ):
        self.cat_cols = cat_cols
        self.smoothing = smoothing
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
                pl.col("__target").sum().alias("__target_sum"),
                pl.len().alias("__target_count"),
            ])

            mapping = stats.with_columns(
                (
                    (pl.col("__target_sum") + self.smoothing * self.global_mean_) /
                    (pl.col("__target_count") + self.smoothing)
                ).alias(f"{col}_target_enc")
            ).select([col, f"{col}_target_enc"])

            self.mappings_[col] = mapping

        return self

    def fit_transform(self, X: pl.DataFrame, y: np.ndarray):
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
                    (
                        (pl.col(f"__{col}_sum") - pl.col("__target")) +
                        self.smoothing * self.global_mean_
                    ) /
                    ((pl.col(f"__{col}_count") - 1) + self.smoothing)
                )
                .otherwise(self.global_mean_)
                .alias(f"{col}_target_enc")
            )

            drop_cols = [f"__{col}_sum", f"__{col}_count"]
            if self.drop_original:
                drop_cols.append(col)
            df = df.drop(drop_cols)

        return df.drop("__target")

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()

        for col in self.cat_cols:
            if col not in df.columns or col not in self.mappings_:
                continue

            encoded_col = f"{col}_target_enc"
            df = df.join(self.mappings_[col], on=col, how="left")
            df = df.with_columns(
                pl.col(encoded_col).fill_null(self.global_mean_).alias(encoded_col)
            )

            if self.drop_original:
                df = df.drop(col)

        return df