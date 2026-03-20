from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


INT_DTYPES     = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
FLOAT_DTYPES   = {pl.Float32, pl.Float64}
NUMERIC_DTYPES = INT_DTYPES | FLOAT_DTYPES


def _safe_div(num: pl.Expr, den: pl.Expr, alias: str) -> pl.Expr:
    """Divide num/den, returning 0.0 when den is null or zero."""
    return (
        pl.when(den.fill_null(0) > 0)
        .then(num / den)
        .otherwise(0.0)
        .alias(alias)
    )


# ── Step 1: row-level flags (stateless) ───────────────────────────────────────

class PolarsLogicalFeatures(BaseEstimator, TransformerMixin):
    """
    Stateless row-level feature flags derived from a single transaction row.

    WHEN TO CALL: globally, before the fold split.
    WHY SAFE:     no statistics are computed across rows or users.
                  Every expression depends only on the current row's values.

    OUTPUT: same number of rows as input, with new flag columns appended.
    These columns are then consumed by PolarsUserAggregator (step 2).
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
                (pl.col("card_country") != pl.col("payment_country"))
                .cast(pl.Int8).alias("card_payment_mismatch")
            )

        if {"reg_country", "payment_country"}.issubset(df.columns):
            exprs.append(
                (pl.col("reg_country") != pl.col("payment_country"))
                .cast(pl.Int8).alias("reg_payment_mismatch")
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
                pl.col("timestamp_tr").dt.hour().is_between(0, 5, closed="both")
                .cast(pl.Int8).alias("is_night_tx"),
                pl.col("timestamp_tr").dt.weekday().is_in([6, 7])
                .cast(pl.Int8).alias("is_weekend_tx"),
            ])

        if {"timestamp_tr", "timestamp_reg"}.issubset(df.columns):
            exprs.extend([
                (
                    (pl.col("timestamp_tr") - pl.col("timestamp_reg")).dt.total_seconds() / 3600
                ).clip(lower_bound=0).alias("hours_since_registration"),
                (
                    (pl.col("timestamp_tr") - pl.col("timestamp_reg")).dt.total_seconds() / 86400
                ).clip(lower_bound=0).alias("days_since_registration"),
            ])

        return df.with_columns(exprs) if exprs else df


# ── Step 2: user-level aggregation (stateful) ─────────────────────────────────

def build_user_level_features(
    df: pl.DataFrame,
    user_col: str = "id_user",
    fail_rate_threshold: float = 0.80,
    unique_cards_threshold: int = 3,
    avg_hours_between_tx_threshold: float = 2.0,
    micro_payment_threshold: float = 1.0,
) -> pl.DataFrame:
    """
    Aggregate transaction-level rows into one-row-per-user statistics.

    Called by PolarsUserAggregator.fit() on the training fold only.
    The resulting stats are then joined to val/test rows via transform().

    Requires PolarsLogicalFeatures to have run first (uses its output columns
    such as blocked_by_antifraud, is_visa, is_night_tx, etc.).
    """
    if user_col not in df.columns:
        return pl.DataFrame()

    agg: list[pl.Expr] = [pl.len().alias("user_tx_count")]

    if "amount" in df.columns:
        agg.extend([
            pl.col("amount").sum().alias("user_amount_sum"),
            pl.col("amount").mean().alias("user_amount_mean"),
            pl.col("amount").max().alias("user_amount_max"),
            pl.col("amount").min().alias("user_amount_min"),
            pl.col("amount").std().alias("user_amount_std"),
        ])

    if "status" in df.columns:
        agg.extend([
            (pl.col("status") == "fail").sum().alias("user_fail_count"),
            (pl.col("status") == "success").sum().alias("user_success_count"),
        ])

    for col, out in [
        ("blocked_by_antifraud",   "user_has_antifraud_error"),
        ("has_fraud_error_group",  "user_has_fraud_error"),
        ("card_payment_mismatch",  "user_has_card_payment_mismatch"),
        ("reg_payment_mismatch",   "user_has_reg_payment_mismatch"),
        ("triple_geo_mismatch",    "user_has_triple_geo_mismatch"),
        ("is_credit_card",         "user_credit_card_tx_count"),
        ("is_visa",                "user_visa_tx_count"),
        ("is_card_init",           "user_card_init_count"),
        ("is_card_recurring",      "user_card_recurring_count"),
        ("is_google_pay",          "user_google_pay_count"),
        ("is_night_tx",            "user_night_tx_count"),
        ("is_eur_tx",              "user_eur_tx_count"),
        ("is_micro_amount",        "user_micro_tx_count"),
        ("insufficient_funds_error","user_insufficient_funds_count"),
        ("do_not_honor_error",     "user_do_not_honor_count"),
        ("cvv_error",              "user_cvv_error_count"),
        ("three_ds_error",         "user_3ds_error_count"),
        ("null_card_holder_non_gpay","user_null_holder_non_gpay_count"),
    ]:
        if col in df.columns:
            op = pl.col(col).max() if col in {
                "blocked_by_antifraud", "has_fraud_error_group",
                "card_payment_mismatch", "reg_payment_mismatch", "triple_geo_mismatch",
            } else pl.col(col).sum()
            agg.append(op.alias(out))

    for col, out in [
        ("currency",         "user_currency_nunique"),
        ("payment_country",  "user_payment_country_nunique"),
        ("card_country",     "user_card_country_nunique"),
        ("transaction_type", "user_tx_type_nunique"),
    ]:
        if col in df.columns:
            agg.append(pl.col(col).n_unique().alias(out))

    if "card_mask_hash" in df.columns:
        agg.append(pl.col("card_mask_hash").n_unique().alias("user_unique_cards_count"))
    if "card_holder" in df.columns:
        agg.append(pl.col("card_holder").drop_nulls().n_unique().alias("user_unique_card_holders_count"))
    if "error_group" in df.columns:
        agg.append(pl.col("error_group").drop_nulls().n_unique().alias("user_error_diversity_count"))
    if "timestamp_tr" in df.columns:
        agg.extend([
            pl.col("timestamp_tr").min().alias("user_first_tx_ts"),
            pl.col("timestamp_tr").max().alias("user_last_tx_ts"),
        ])
    if "timestamp_reg" in df.columns:
        agg.append(pl.col("timestamp_reg").min().alias("user_reg_ts"))

    stats = df.group_by(user_col).agg(agg)

    # Fill nulls on all numeric columns.
    stats = stats.with_columns([
        pl.col(col).fill_null(0)
        for col, dtype in zip(stats.columns, stats.dtypes)
        if col != user_col and dtype in NUMERIC_DTYPES
    ])

    # Derived ratios.
    derived: list[pl.Expr] = []

    for num_col, alias in [
        ("user_fail_count",                 "user_fail_rate"),
        ("user_success_count",              "user_success_rate"),
        ("user_credit_card_tx_count",       "user_credit_card_rate"),
        ("user_visa_tx_count",              "user_visa_rate"),
        ("user_null_holder_non_gpay_count", "user_null_holder_rate"),
        ("user_night_tx_count",             "user_night_tx_rate"),
        ("user_eur_tx_count",               "user_eur_tx_rate"),
    ]:
        if {num_col, "user_tx_count"}.issubset(stats.columns):
            derived.append(_safe_div(pl.col(num_col), pl.col("user_tx_count"), alias))

    if {"user_card_init_count", "user_card_recurring_count"}.issubset(stats.columns):
        derived.append(
            (pl.col("user_card_init_count") / (pl.col("user_card_recurring_count") + 1))
            .alias("user_init_to_recurring_ratio")
        )

    if {"user_last_tx_ts", "user_first_tx_ts"}.issubset(stats.columns):
        derived.append(
            ((pl.col("user_last_tx_ts") - pl.col("user_first_tx_ts")).dt.total_seconds() / 3600)
            .alias("user_tx_timespan_hours")
        )

    if {"user_first_tx_ts", "user_reg_ts"}.issubset(stats.columns):
        diff = (pl.col("user_first_tx_ts") - pl.col("user_reg_ts")).dt.total_seconds()
        derived.extend([
            (diff / 3600).clip(lower_bound=0).alias("user_reg_to_first_tx_hours"),
            (diff / 86400).clip(lower_bound=0).alias("user_account_age_days"),
        ])

    if derived:
        stats = stats.with_columns(derived)

    # Post-derived.
    post: list[pl.Expr] = []

    if {"user_tx_timespan_hours", "user_tx_count"}.issubset(stats.columns):
        post.append(
            pl.when(pl.col("user_tx_count") > 1)
            .then(pl.col("user_tx_timespan_hours") / (pl.col("user_tx_count") - 1))
            .otherwise(0.0)
            .alias("user_avg_hours_between_tx")
        )

    if {"user_unique_cards_count", "user_account_age_days"}.issubset(stats.columns):
        post.append(
            pl.when(pl.col("user_account_age_days") > 0)
            .then(pl.col("user_unique_cards_count") / pl.col("user_account_age_days"))
            .otherwise(pl.col("user_unique_cards_count").cast(pl.Float64))
            .alias("user_cards_per_day")
        )

    if post:
        stats = stats.with_columns(post)

    # Rule flags.
    rules: list[pl.Expr] = []

    if {"user_amount_min", "user_fail_count"}.issubset(stats.columns):
        rules.append(
            (
                (pl.col("user_amount_min") < micro_payment_threshold) &
                (pl.col("user_fail_count") >= 2)
            ).cast(pl.Int8).alias("user_micro_payment_flag")
        )

    if {"user_unique_cards_count", "user_fail_rate", "user_avg_hours_between_tx"}.issubset(stats.columns):
        rules.append(
            (
                (pl.col("user_unique_cards_count") >= unique_cards_threshold) &
                (pl.col("user_fail_rate") > fail_rate_threshold) &
                (pl.col("user_avg_hours_between_tx") < avg_hours_between_tx_threshold)
            ).cast(pl.Int8).alias("user_card_testing_flag")
        )

    if rules:
        stats = stats.with_columns(rules)

    # Composite rule score.
    score_parts = [
        pl.col(col).fill_null(0).cast(pl.Int64)
        for col in [
            "user_has_antifraud_error", "user_has_fraud_error",
            "user_has_triple_geo_mismatch", "user_micro_payment_flag",
            "user_card_testing_flag",
        ]
        if col in stats.columns
    ]
    if "user_unique_cards_count" in stats.columns:
        score_parts.append((pl.col("user_unique_cards_count") >= unique_cards_threshold).cast(pl.Int64))
    if "user_fail_rate" in stats.columns:
        score_parts.append((pl.col("user_fail_rate") > fail_rate_threshold).cast(pl.Int64))

    if score_parts:
        stats = stats.with_columns(sum(score_parts).cast(pl.Int16).alias("user_rule_score"))

    # Drop temporary timestamp columns used only for derived calculations.
    ts_temp = [c for c in ["user_first_tx_ts", "user_last_tx_ts", "user_reg_ts"] if c in stats.columns]
    if ts_temp:
        stats = stats.drop(ts_temp)

    return stats


class PolarsUserAggregator(BaseEstimator, TransformerMixin):
    """
    Stateful user-level aggregator.

    WHEN TO CALL: inside each fold, after PolarsLogicalFeatures.
    fit()       — computes per-user stats from the training fold only.
    transform() — joins those stats onto any split (val, test).

    Val/test users not seen during fit() receive median fill values.
    This mirrors the production scenario where new users have no history.

    Call order inside fold:
        1. PolarsLogicalFeatures.transform(X)      ← creates flag columns
        2. PolarsUserAggregator.fit(X_train)       ← learn from train
        3. PolarsUserAggregator.transform(X_train) ← enrich train
        4. PolarsUserAggregator.transform(X_val)   ← enrich val with train stats
    """

    def __init__(self, user_col: str = "id_user"):
        self.user_col = user_col
        self.user_stats_: pl.DataFrame | None = None
        self.defaults_: dict[str, float | int] = {}

    def fit(self, X: pl.DataFrame, y=None):
        if self.user_col not in X.columns:
            self.user_stats_ = None
            return self

        self.user_stats_ = build_user_level_features(X, user_col=self.user_col)

        self.defaults_ = {}
        for col in self.user_stats_.columns:
            if col == self.user_col:
                continue
            series = self.user_stats_.get_column(col)
            median = series.median()
            if series.dtype in FLOAT_DTYPES:
                self.defaults_[col] = 0.0 if median is None else float(median)
            elif series.dtype in INT_DTYPES:
                self.defaults_[col] = 0 if median is None else int(median)
            else:
                self.defaults_[col] = 0

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.user_stats_ is None or self.user_col not in X.columns:
            return X

        df = X.join(self.user_stats_, on=self.user_col, how="left")

        fill_exprs = [
            pl.col(col).fill_null(val).alias(col)
            for col, val in self.defaults_.items()
            if col in df.columns
        ]
        return df.with_columns(fill_exprs) if fill_exprs else df


# ── Step 3: target encoding (stateful) ────────────────────────────────────────

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