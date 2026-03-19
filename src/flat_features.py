from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

SUSPICIOUS_STATUSES = {"declined", "failed", "chargeback", "fail"}

DROP_COLS = {
    "id_user",
    "is_fraud",
    "timestamp_reg",
    "email",
    "gender",
    "reg_country",
    "traffic_type",
}

NUMERIC_DTYPES = {
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}


def _parse_datetime(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df

    dtype = df.schema[col]
    if dtype == pl.Datetime:
        return df
    if dtype == pl.Date:
        return df.with_columns(pl.col(col).cast(pl.Datetime).alias(col))

    return df.with_columns(
        pl.col(col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_datetime(time_zone="UTC", strict=False)
        .dt.replace_time_zone(None)
        .alias(col)
    )


def build_features(users: pl.DataFrame, trx: pl.DataFrame) -> pl.DataFrame:
    users_df = users.clone()
    trx_df = trx.clone()

    users_df = _parse_datetime(users_df, "timestamp_reg")
    trx_df = _parse_datetime(trx_df, "timestamp_tr")

    if "gender" in users_df.columns:
        users_df = users_df.with_columns(
            (pl.col("gender") == "male").cast(pl.Int8).alias("gender_enc")
        )

    for col in ("reg_country", "traffic_type"):
        if col in users_df.columns:
            freq = users_df.group_by(col).agg(pl.len().alias(f"{col}_freq"))
            users_df = users_df.join(freq, on=col, how="left")

    if not trx_df.is_empty():
        if "status" in trx_df.columns and "is_susp" not in trx_df.columns:
            trx_df = trx_df.with_columns(
                pl.col("status")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(list(SUSPICIOUS_STATUSES))
                .cast(pl.Int8)
                .alias("is_susp")
            )

        agg_exprs: list[pl.Expr] = [
            pl.len().alias("trx_count"),
        ]

        if "amount" in trx_df.columns:
            agg_exprs.extend([
                pl.col("amount").sum().alias("total_amount"),
                pl.col("amount").mean().alias("mean_amount"),
                pl.col("amount").std().alias("std_amount"),
                pl.col("amount").max().alias("max_amount"),
                pl.col("amount").min().alias("min_amount"),
            ])

        if "is_susp" in trx_df.columns:
            agg_exprs.extend([
                pl.col("is_susp").sum().alias("n_suspicious"),
                pl.col("is_susp").mean().alias("suspicious_rate"),
            ])

        for col, alias in [
            ("card_country", "unique_card_countries"),
            ("currency", "unique_currencies"),
            ("card_mask_hash", "unique_cards"),
            ("payment_country", "unique_payment_countries"),
            ("transaction_type", "unique_trx_types"),
        ]:
            if col in trx_df.columns:
                agg_exprs.append(pl.col(col).n_unique().alias(alias))

        if "timestamp_tr" in trx_df.columns:
            agg_exprs.extend([
                pl.col("timestamp_tr").min().alias("first_trx_ts"),
                pl.col("timestamp_tr").max().alias("last_trx_ts"),
            ])

        agg = trx_df.group_by("id_user").agg(agg_exprs)

        derived = []
        if {"std_amount", "mean_amount"}.issubset(agg.columns):
            derived.append(
                (pl.col("std_amount") / (pl.col("mean_amount") + 1e-9)).alias("amount_cv")
            )
        if {"n_suspicious", "trx_count"}.issubset(agg.columns):
            derived.append(
                (pl.col("n_suspicious") / (pl.col("trx_count") + 1)).alias("suspicious_per_trx")
            )
        if {"unique_cards", "trx_count"}.issubset(agg.columns):
            derived.append(
                (pl.col("unique_cards") / (pl.col("trx_count") + 1)).alias("cards_per_trx")
            )
        if {"first_trx_ts", "last_trx_ts"}.issubset(agg.columns):
            derived.append(
                ((pl.col("last_trx_ts") - pl.col("first_trx_ts")).dt.total_seconds() / 3600)
                .fill_null(0.0)
                .alias("trx_window_hours")
            )

        if derived:
            agg = agg.with_columns(derived)

        users_df = users_df.join(agg, on="id_user", how="left")

    if {"timestamp_reg", "first_trx_ts"}.issubset(users_df.columns):
        users_df = users_df.with_columns(
            ((pl.col("first_trx_ts") - pl.col("timestamp_reg")).dt.total_seconds() / 3600)
            .clip(lower_bound=0)
            .fill_null(0.0)
            .alias("hours_to_first_trx")
        )

    numeric_cols = [
        c for c, t in users_df.schema.items()
        if t in NUMERIC_DTYPES and c != "is_fraud"
    ]
    if numeric_cols:
        users_df = users_df.with_columns([pl.col(c).fill_null(0).alias(c) for c in numeric_cols])

    return users_df


def _feature_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in DROP_COLS and df[c].dtype in NUMERIC_DTYPES
    ]


def to_model_matrix(df: pl.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feat_cols = _feature_columns(df)
    X = df.select(feat_cols).to_pandas()
    X = X.astype(np.float32)
    return X, feat_cols


def align_to_features(df: pl.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.clone()

    missing = [col for col in feat_cols if col not in out.columns]
    if missing:
        out = out.with_columns([pl.lit(0.0).alias(col) for col in missing])

    return out.select(feat_cols).to_pandas().astype(np.float32)