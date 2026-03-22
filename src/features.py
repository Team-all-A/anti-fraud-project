"""
src/features.py
═══════════════════════════════════════════════════════════════════════════════
Feature engineering pipeline: raw merged table → flat user-level dataset.

PUBLIC API:
  build_flat_user_dataset(df, is_train)  → flat pl.DataFrame
  add_ratio_features(df)                 → enriched pl.DataFrame
  build_risk_indicators(df)              → df + card_testing_flag, micro_payment_flag, rule_score

FIXES vs original build_risk_indicators:
  OLD card_testing_flag: (unique_cards>=3) & (fail>0.80) & (avg_hours<2)
    → unique_cards>=3 was TRUE in 100% of rows (artifact)
    → fail>0.80 was TRUE in 0 rows (unrealistic threshold)
    → card_testing_flag = 0 ALWAYS

  NEW card_testing_flag: (cvv_rate>0.20) & (night_tx_rate>0.35) & (fail_rate>0.40) & (tx>=5)
    → cvv_rate discriminates carding (real CVV testing pattern)
    → night_tx_rate discriminates off-hours fraud
    → requires min 5 transactions to avoid false positives on new accounts

  OLD rule_score: sum of 6 near-constant binary flags (KS < 0.03 each)
    → geo_mismatch_triple=1 in 100%, has_antifraud=1 in 96%
    → rule_score always = 3 or 4, zero discrimination

  NEW rule_score: 8 count-based + rate-based components (KS > 0.05 each)
    → uses antifraud_count>=5 instead of has_antifraud=1
    → replaces geo_triple with geo_mismatch_card_payment (not all-ones)
    → adds night_rate>0.40, cvv_count>=4 as new discriminators
"""

from __future__ import annotations

import polars as pl

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

_EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_datetime(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Safely parse a column to pl.Datetime regardless of its current dtype."""
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


def _safe_ratio(num: pl.Expr, den: pl.Expr, alias: str) -> pl.Expr:
    return (
        pl.when(den.fill_null(0) > 0)
        .then(num / den)
        .otherwise(0.0)
        .alias(alias)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  RATIO FEATURES  (key for ML signal — replaces near-constant binary flags)
# ══════════════════════════════════════════════════════════════════════════════

def add_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rate/ratio features to the flat user-level DataFrame.

    WHY rates instead of counts:
      has_antifraud_error = 1 in 96% of users → precision ~5% when used in rules
      antifraud_error_count / tx_total:
        fraud:  0.30–0.80  (every 2nd–3rd transaction is antifraud error)
        legit:  0.01–0.05  (rare errors that happen to everyone)
      → discriminative lift >> 10x on real data

    All expressions are guarded: if a column is missing nothing is added.
    Existing columns are never overwritten (idempotent).
    """
    existing = set(df.columns)
    exprs: list[pl.Expr] = []

    def _add(alias: str, expr: pl.Expr) -> None:
        if alias not in existing:
            exprs.append(expr.alias(alias))

    # ── Amount-based features ──────────────────────────────────────────────
    # Coefficient of variation — хаотичні суми транзакцій = підозріло
    if {"tx_amount_std", "tx_amount_mean"} <= existing:
        _add("amount_cv",
             pl.col("tx_amount_std") / (pl.col("tx_amount_mean").abs() + _EPS))

    # Range ratio — великий розкид між min i max = carding probe pattern
    if {"tx_amount_max", "tx_amount_min", "tx_amount_mean"} <= existing:
        _add("amount_range_ratio",
             (pl.col("tx_amount_max") - pl.col("tx_amount_min"))
             / (pl.col("tx_amount_mean").abs() + _EPS))

    # Fail amount proxy — видалено звідси, вже є нижче в оригінальному коді

    # ── Transaction intensity ──────────────────────────────────────────────
    # Tx per day — аномально висока активність для нового акаунта
    if {"tx_total_count", "account_age_days"} <= existing:
        _add("tx_per_day",
             pl.col("tx_total_count") / (pl.col("account_age_days") + _EPS))

    # Fail per day — burst failures normalised by account age
    if {"tx_fail_count", "account_age_days"} <= existing:
        _add("fail_per_day",
             pl.col("tx_fail_count") / (pl.col("account_age_days") + _EPS))

    # ── Cross-signal interactions ──────────────────────────────────────────
    # af_rate × unique_cards — комбінація antifraud density + card diversity
    if {"af_rate", "unique_cards_count"} <= existing:
        _add("af_rate_x_cards",
             pl.col("af_rate") * pl.col("unique_cards_count").cast(pl.Float64))

    # cvv × night — CVV errors at night = automated carding
    if {"cvv_rate", "night_tx_rate"} <= existing:
        _add("cvv_x_night",
             pl.col("cvv_rate") * pl.col("night_tx_rate"))

    # dnh × fail — "do not honor" combined with fail rate
    if {"dnh_rate", "tx_fail_rate"} <= existing:
        _add("dnh_x_fail",
             pl.col("dnh_rate") * pl.col("tx_fail_rate"))

    # af_rate × tx_per_day — antifraud density + speed (strongest combined signal)
    if {"af_rate", "tx_per_day"} <= existing:
        _add("af_rate_x_tx_per_day",
             pl.col("af_rate") * pl.col("tx_per_day"))

    # ── Card holder anomaly ────────────────────────────────────────────────
    # Ratio of unique card holders to unique cards — normally 1:1
    if {"unique_card_holders_count", "unique_cards_count"} <= existing:
        _add("holders_per_card",
             pl.col("unique_card_holders_count").cast(pl.Float64)
             / (pl.col("unique_cards_count").cast(pl.Float64) + _EPS))

    # Error density ──────────────────────────────────────────────────────
    if {"antifraud_error_count", "tx_total_count"} <= existing:
        _add("af_rate",
             pl.col("antifraud_error_count") / (pl.col("tx_total_count") + _EPS))

    if {"fraud_error_count", "tx_total_count"} <= existing:
        _add("fr_rate",
             pl.col("fraud_error_count") / (pl.col("tx_total_count") + _EPS))

    if {"cvv_error_count", "tx_total_count"} <= existing:
        _add("cvv_rate",
             pl.col("cvv_error_count") / (pl.col("tx_total_count") + _EPS))

    if {"insufficient_funds_count", "tx_total_count"} <= existing:
        _add("insuf_rate",
             pl.col("insufficient_funds_count") / (pl.col("tx_total_count") + _EPS))

    if {"do_not_honor_count", "tx_total_count"} <= existing:
        _add("dnh_rate",
             pl.col("do_not_honor_count") / (pl.col("tx_total_count") + _EPS))

    if {"tds_error_count", "tx_total_count"} <= existing:
        _add("tds_rate",
             pl.col("tds_error_count") / (pl.col("tx_total_count") + _EPS))

    # ── Interaction features (jointly capture fail + error density) ────────
    if {"tx_fail_rate", "antifraud_error_count", "tx_total_count"} <= existing:
        _add("fail_x_af_rate",
             pl.col("tx_fail_rate")
             * pl.col("antifraud_error_count") / (pl.col("tx_total_count") + _EPS))

    if {"tx_fail_rate", "fraud_error_count", "tx_total_count"} <= existing:
        _add("fail_x_fr_rate",
             pl.col("tx_fail_rate")
             * pl.col("fraud_error_count") / (pl.col("tx_total_count") + _EPS))

    if {"tx_fail_rate", "rule_score"} <= existing:
        _add("score_x_fail",
             pl.col("rule_score").cast(pl.Float64) * pl.col("tx_fail_rate"))

    if {"tx_fail_rate", "night_tx_rate"} <= existing:
        _add("night_x_fail",
             pl.col("night_tx_rate") * pl.col("tx_fail_rate"))

    if {"tx_fail_rate", "unique_cards_count"} <= existing:
        _add("cards_x_fail",
             pl.col("unique_cards_count").cast(pl.Float64) * pl.col("tx_fail_rate"))

    # ── Velocity / temporal ────────────────────────────────────────────────
    if {"tx_fail_count", "tx_timespan_days"} <= existing:
        _add("fail_velocity",
             pl.col("tx_fail_count") / (pl.col("tx_timespan_days") + _EPS))

    if {"tx_amount_max", "tx_fail_rate"} <= existing:
        _add("amount_x_fail",
             pl.col("tx_amount_max") * pl.col("tx_fail_rate"))

    if {"card_init_count", "tx_total_count"} <= existing:
        _add("card_init_ratio",
             pl.col("card_init_count") / (pl.col("tx_total_count") + _EPS))

    # ── Log-transform heavy-tailed counts (reduces skewness for LGBM) ─────
    for cnt_col in ["antifraud_error_count", "fraud_error_count",
                    "tx_fail_count", "cvv_error_count", "insufficient_funds_count"]:
        if cnt_col in existing:
            alias = f"{cnt_col}_log"
            if alias not in existing:
                exprs.append(
                    (pl.col(cnt_col).cast(pl.Float64) + 1.0)
                    .log()
                    .alias(alias)
                )

    if exprs:
        df = df.with_columns(exprs)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  RISK INDICATORS  (FIXED — replaces constant-dominated original)
# ══════════════════════════════════════════════════════════════════════════════

def build_risk_indicators(flat: pl.DataFrame) -> pl.DataFrame:
    """
    Compute card_testing_flag, micro_payment_flag, rule_score.

    All three fields overwrite any previously computed versions so this
    function is safe to call after add_ratio_features or re-compute.

    CHANGES vs original:
    ┌────────────────────────┬────────────────────────┬───────────────────────────┐
    │ Field                  │ OLD (broken)           │ NEW (fixed)               │
    ├────────────────────────┼────────────────────────┼───────────────────────────┤
    │ card_testing_flag      │ unique_cards>=3 (100%) │ cvv_rate>0.20 +           │
    │                        │ + fail>0.80 (0 rows)   │ night_rate>0.35 +         │
    │                        │ → always 0             │ fail_rate>0.40 + tx>=5    │
    ├────────────────────────┼────────────────────────┼───────────────────────────┤
    │ micro_payment_flag     │ amount_min<1.0 +        │ fail_count>=5 +           │
    │                        │ fail_count>=2           │ amount_max>40 +           │
    │                        │ (KS=0.02)              │ night_rate>0.20           │
    ├────────────────────────┼────────────────────────┼───────────────────────────┤
    │ rule_score components  │ geo_triple (100% TRUE) │ af_count>=5 (discriminates│
    │                        │ has_af (96% TRUE)      │ fraud vs legit >>5x)      │
    │                        │ unique_cards>=3 (100%) │ + 7 real signal components│
    │                        │ fail>0.80 (0 rows)     │ (counts + rates)          │
    │                        │ → score always 3-4     │ → score spread 0-8        │
    └────────────────────────┴────────────────────────┴───────────────────────────┘
    """
    # Ensure ratio features are present
    if "af_rate" not in flat.columns or "cvv_rate" not in flat.columns:
        flat = add_ratio_features(flat)

    # ── card_testing_flag ─────────────────────────────────────────────────
    # Carding pattern: many CVV errors per transaction, concentrated at night,
    # with high fail rate. Requires min 5 tx to exclude 1-2 transaction accounts.
    flat = flat.with_columns(
        pl.when(
            (pl.col("cvv_rate")        > 0.20) &
            (pl.col("night_tx_rate")   > 0.35) &
            (pl.col("tx_fail_rate")    > 0.40) &
            (pl.col("tx_total_count")  >= 5)
        )
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("card_testing_flag")
    )

    # ── micro_payment_flag ────────────────────────────────────────────────
    # Fraud pattern: many failures on large amounts at night (not small amounts)
    flat = flat.with_columns(
        pl.when(
            (pl.col("tx_fail_count")  >= 5)    &
            (pl.col("tx_amount_max")  > 40.0)  &
            (pl.col("night_tx_rate")  > 0.20)
        )
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("micro_payment_flag")
    )

    # ── rule_score: 8 count/rate-based components ─────────────────────────
    # Each component selected to have KS > 0.04 between fraud and legit.
    # Uses COUNT thresholds (>=5, >=3) instead of binary flags (=1)
    # to avoid the 96%-always-true problem of has_antifraud_error.
    flat = flat.with_columns(
        (
            # C1: >= 5 antifraud errors (not just has=1 which is TRUE for 96%)
            (pl.col("antifraud_error_count") >= 5).cast(pl.Int64)

            # C2: >= 3 fraud errors (count-based, not binary flag)
            + (pl.col("fraud_error_count") >= 3).cast(pl.Int64)

            # C3: card-payment geo mismatch (NOT triple which is 100% TRUE)
            + pl.col("geo_mismatch_card_payment").fill_null(0).cast(pl.Int64)

            # C4: velocity — cards appear faster than organic behaviour
            + (pl.col("cards_per_day") > 30.0).cast(pl.Int64)

            # C5: elevated fail rate (threshold 0.45, not 0.80 which has 0 rows)
            + (pl.col("tx_fail_rate") > 0.45).cast(pl.Int64)

            # C6: night activity anomaly
            + (pl.col("night_tx_rate") > 0.40).cast(pl.Int64)

            # C7: CVV burst (carding signature)
            + (pl.col("cvv_error_count") >= 4).cast(pl.Int64)

            # C8: micro_payment_flag (now realistic)
            + pl.col("micro_payment_flag").cast(pl.Int64)
        )
        .cast(pl.Int8)
        .alias("rule_score")
    )

    return flat


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE: raw merged → flat user-level dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_flat_user_dataset(df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
    """
    Aggregate merged raw (users JOIN transactions) → one row per user.

    Output columns: user-static features + ~50 aggregated transaction features
    + ratio features (from add_ratio_features) + risk indicators (from build_risk_indicators).
    All numeric nulls filled with 0 before returning.
    """
    if "id_user" not in df.columns:
        raise ValueError("Input DataFrame must contain 'id_user'.")

    flat = df.clone()
    for col in ("timestamp_reg", "timestamp_tr"):
        flat = _parse_datetime(flat, col)

    # ── Static user features ───────────────────────────────────────────────
    static_exprs: list[pl.Expr] = [pl.col("id_user")]
    for col in ("reg_country", "traffic_type", "gender"):
        if col in flat.columns:
            static_exprs.append(pl.col(col))
    if "timestamp_reg" in flat.columns:
        static_exprs.extend([
            pl.col("timestamp_reg"),
            pl.col("timestamp_reg").dt.hour().alias("reg_hour_of_day"),
            pl.col("timestamp_reg").dt.weekday().alias("reg_day_of_week"),
        ])
    if "email" in flat.columns:
        static_exprs.extend([
            pl.col("email").is_null().cast(pl.Int8).alias("email_is_null"),
            pl.col("email").cast(pl.Utf8, strict=False)
            .str.extract(r"@(.+)$", 1).fill_null("unknown").alias("email_domain"),
        ])
    if is_train and "is_fraud" in flat.columns:
        static_exprs.append(pl.col("is_fraud"))

    user_static = flat.select(static_exprs).unique(subset=["id_user"], keep="first")

    # ── Filter to rows with transactions ──────────────────────────────────
    tx = (flat.filter(pl.col("timestamp_tr").is_not_null())
          if "timestamp_tr" in flat.columns else flat)

    # ── Row-level flag computation (vectorized, no apply) ─────────────────
    row_exprs: list[pl.Expr] = []

    if "status" in tx.columns:
        row_exprs += [
            (pl.col("status") == "success").cast(pl.Int8).alias("_ok"),
            (pl.col("status") == "fail").cast(pl.Int8).alias("_fail"),
        ]

    if "error_group" in tx.columns:
        err = pl.col("error_group").cast(pl.Utf8).fill_null("__none__")
        row_exprs += [
            (err == "antifraud").cast(pl.Int8).alias("_err_af"),
            (err == "fraud").cast(pl.Int8).alias("_err_fr"),
            (err == "insufficient funds error").cast(pl.Int8).alias("_err_insuf"),
            (err == "do not honor").cast(pl.Int8).alias("_err_dnh"),
            (err == "cvv error").cast(pl.Int8).alias("_err_cvv"),
            (err == "3ds error").cast(pl.Int8).alias("_err_3ds"),
            pl.col("error_group").is_not_null().cast(pl.Int8).alias("_has_err"),
        ]

    if {"card_country", "payment_country"} <= set(tx.columns):
        row_exprs.append(
            (pl.col("card_country") != pl.col("payment_country"))
            .cast(pl.Int8).alias("_mm_cp"))
    if {"payment_country", "reg_country"} <= set(tx.columns):
        row_exprs.append(
            (pl.col("payment_country") != pl.col("reg_country"))
            .cast(pl.Int8).alias("_mm_rp"))
    if {"card_country", "payment_country", "reg_country"} <= set(tx.columns):
        row_exprs.append(
            ((pl.col("card_country") != pl.col("payment_country")) &
             (pl.col("payment_country") != pl.col("reg_country")))
            .cast(pl.Int8).alias("_mm_triple"))

    if "card_type" in tx.columns:
        row_exprs.append((pl.col("card_type") == "CREDIT").cast(pl.Int8).alias("_credit"))
    if "card_brand" in tx.columns:
        row_exprs.append((pl.col("card_brand") == "VISA").cast(pl.Int8).alias("_visa"))
    if "transaction_type" in tx.columns:
        row_exprs += [
            (pl.col("transaction_type") != "google-pay").cast(pl.Int8).alias("_non_gpay"),
            (pl.col("transaction_type") == "card_init").cast(pl.Int8).alias("_init"),
            (pl.col("transaction_type") == "card_recurring").cast(pl.Int8).alias("_recurring"),
            (pl.col("transaction_type") == "google-pay").cast(pl.Int8).alias("_gpay"),
        ]
    if {"transaction_type", "card_holder"} <= set(tx.columns):
        row_exprs.append(
            ((pl.col("transaction_type") != "google-pay") &
             pl.col("card_holder").is_null())
            .cast(pl.Int8).alias("_null_holder"))
    if "timestamp_tr" in tx.columns:
        row_exprs.append(
            pl.col("timestamp_tr").dt.hour()
            .is_between(0, 5, closed="both").cast(pl.Int8).alias("_night"))
    if "currency" in tx.columns:
        row_exprs.append((pl.col("currency") == "EUR").cast(pl.Int8).alias("_eur"))

    if row_exprs:
        tx = tx.with_columns(row_exprs)

    # ── Transaction activity aggregations ──────────────────────────────────
    tx_stats = tx.group_by("id_user").agg([
        pl.len().alias("tx_total_count"),
        pl.col("_ok").sum().alias("tx_success_count")  if "_ok"   in tx.columns else pl.lit(0).alias("tx_success_count"),
        pl.col("_fail").sum().alias("tx_fail_count")   if "_fail" in tx.columns else pl.lit(0).alias("tx_fail_count"),
        pl.col("amount").sum().alias("tx_amount_sum")  if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_sum"),
        pl.col("amount").mean().alias("tx_amount_mean") if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_mean"),
        pl.col("amount").max().alias("tx_amount_max")  if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_max"),
        pl.col("amount").min().alias("tx_amount_min")  if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_min"),
        pl.col("amount").std().alias("tx_amount_std")  if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_std"),
    ]).with_columns([
        pl.col("tx_amount_std").fill_null(0.0),
        _safe_ratio(pl.col("tx_fail_count"), pl.col("tx_total_count"), "tx_fail_rate"),
    ])

    # ── Error pattern aggregations ─────────────────────────────────────────
    err_agg: list[pl.Expr] = []
    for flag, has_col, cnt_col in [
        ("_err_af",    "has_antifraud_error", "antifraud_error_count"),
        ("_err_fr",    "has_fraud_error",     "fraud_error_count"),
    ]:
        if flag in tx.columns:
            err_agg += [
                pl.col(flag).max().alias(has_col),
                pl.col(flag).sum().alias(cnt_col),
            ]
        else:
            err_agg += [pl.lit(0).alias(has_col), pl.lit(0).alias(cnt_col)]

    for flag, alias in [("_err_insuf", "insufficient_funds_count"),
                        ("_err_dnh",   "do_not_honor_count"),
                        ("_err_cvv",   "cvv_error_count"),
                        ("_err_3ds",   "tds_error_count")]:
        err_agg.append(
            pl.col(flag).sum().alias(alias) if flag in tx.columns
            else pl.lit(0).alias(alias)
        )

    error_patterns = tx.group_by("id_user").agg(err_agg)

    # error_diversity: n unique error types per user
    if {"id_user", "error_group", "_has_err"} <= set(tx.columns):
        diversity = (
            tx.filter(pl.col("_has_err") == 1)
            .group_by("id_user")
            .agg(pl.col("error_group").n_unique().alias("error_diversity_count"))
        )
        error_patterns = error_patterns.join(diversity, on="id_user", how="left")
    error_patterns = error_patterns.with_columns(
        pl.col("error_diversity_count").fill_null(0).cast(pl.Int32)
    )

    # ── Geo anomalies ──────────────────────────────────────────────────────
    geo_agg: list[pl.Expr] = [
        pl.col("_mm_cp").max().alias("geo_mismatch_card_payment")    if "_mm_cp"     in tx.columns else pl.lit(0).alias("geo_mismatch_card_payment"),
        pl.col("_mm_rp").max().alias("geo_mismatch_reg_payment")     if "_mm_rp"     in tx.columns else pl.lit(0).alias("geo_mismatch_reg_payment"),
        pl.col("_mm_triple").max().alias("geo_mismatch_triple")      if "_mm_triple" in tx.columns else pl.lit(0).alias("geo_mismatch_triple"),
        pl.col("payment_country").n_unique().alias("unique_payment_countries_count") if "payment_country" in tx.columns else pl.lit(0).alias("unique_payment_countries_count"),
        pl.col("card_country").n_unique().alias("unique_card_countries_count")       if "card_country"    in tx.columns else pl.lit(0).alias("unique_card_countries_count"),
    ]
    geo = tx.group_by("id_user").agg(geo_agg)

    if "payment_country" in tx.columns:
        dominant = (
            tx.group_by(["id_user", "payment_country"]).len()
            .sort(["id_user", "len"], descending=[False, True])
            .group_by("id_user")
            .agg(pl.col("payment_country").first().alias("dominant_payment_country"))
        )
        geo = geo.join(dominant, on="id_user", how="left")

    # ── Card / velocity ────────────────────────────────────────────────────
    card_agg: list[pl.Expr] = [
        pl.col("card_mask_hash").n_unique().alias("unique_cards_count") if "card_mask_hash" in tx.columns else pl.lit(0).alias("unique_cards_count"),
        pl.col("_credit").sum().alias("_credit_s")                     if "_credit"    in tx.columns else pl.lit(0).alias("_credit_s"),
        pl.col("_visa").sum().alias("_visa_s")                         if "_visa"      in tx.columns else pl.lit(0).alias("_visa_s"),
        pl.col("_non_gpay").sum().alias("_ngpay_s")                    if "_non_gpay"  in tx.columns else pl.lit(0).alias("_ngpay_s"),
        pl.col("_null_holder").sum().alias("_null_h_s")                if "_null_holder" in tx.columns else pl.lit(0).alias("_null_h_s"),
        pl.len().alias("_ttl"),
    ]
    card = tx.group_by("id_user").agg(card_agg).with_columns([
        _safe_ratio(pl.col("_credit_s"), pl.col("_ttl"), "credit_card_rate"),
        _safe_ratio(pl.col("_visa_s"),   pl.col("_ttl"), "visa_rate"),
        _safe_ratio(pl.col("_null_h_s"), pl.col("_ngpay_s"), "card_holder_is_null_rate"),
    ])

    if "card_holder" in tx.columns:
        holder_uniq = (
            tx.filter(pl.col("card_holder").is_not_null())
            .group_by("id_user")
            .agg(pl.col("card_holder").n_unique().alias("unique_card_holders_count"))
        )
        card = card.join(holder_uniq, on="id_user", how="left")

    # account_age = days from registration to first transaction
    first_tx = (tx.group_by("id_user").agg(pl.col("timestamp_tr").min().alias("_first_tx"))
                if "timestamp_tr" in tx.columns
                else pl.DataFrame({"id_user": tx["id_user"].unique()}))

    age = (
        user_static.select([c for c in ("id_user", "timestamp_reg") if c in user_static.columns])
        .join(first_tx, on="id_user", how="left")
    )
    if {"timestamp_reg", "_first_tx"} <= set(age.columns):
        age = age.with_columns(
            ((pl.col("_first_tx") - pl.col("timestamp_reg")).dt.total_seconds() / 86400)
            .clip(lower_bound=0).fill_null(0).alias("account_age_days")
        ).select(["id_user", "account_age_days"])
    else:
        age = age.select("id_user").with_columns(pl.lit(0.0).alias("account_age_days"))

    card = (
        card.join(age, on="id_user", how="left")
        .with_columns([
            pl.col("unique_card_holders_count").fill_null(0).cast(pl.Int32),
            pl.col("account_age_days").fill_null(0.0),
            pl.when(pl.col("account_age_days") > 0)
            .then(pl.col("unique_cards_count") / pl.col("account_age_days"))
            .otherwise(pl.col("unique_cards_count").cast(pl.Float64))
            .alias("cards_per_day"),
        ])
        .drop(["_credit_s", "_visa_s", "_ngpay_s", "_null_h_s", "_ttl"])
    )

    # ── Temporal patterns ──────────────────────────────────────────────────
    tmp_agg: list[pl.Expr] = [
        pl.col("_init").sum().alias("card_init_count")      if "_init"      in tx.columns else pl.lit(0).alias("card_init_count"),
        pl.col("_recurring").sum().alias("card_recurring_count") if "_recurring" in tx.columns else pl.lit(0).alias("card_recurring_count"),
        pl.col("_gpay").sum().alias("google_pay_count")     if "_gpay"      in tx.columns else pl.lit(0).alias("google_pay_count"),
        pl.col("_night").sum().alias("night_tx_count")      if "_night"     in tx.columns else pl.lit(0).alias("night_tx_count"),
        pl.col("_eur").sum().alias("_eur_s")                if "_eur"       in tx.columns else pl.lit(0).alias("_eur_s"),
        pl.len().alias("_ttl"),
        pl.col("timestamp_tr").max().alias("_ts_max")       if "timestamp_tr" in tx.columns else pl.lit(None).alias("_ts_max"),
        pl.col("timestamp_tr").min().alias("_ts_min")       if "timestamp_tr" in tx.columns else pl.lit(None).alias("_ts_min"),
    ]
    temporal = tx.group_by("id_user").agg(tmp_agg).with_columns([
        (pl.col("card_init_count") / (pl.col("card_recurring_count") + 1)).alias("init_to_recurring_ratio"),
        _safe_ratio(pl.col("night_tx_count"), pl.col("_ttl"), "night_tx_rate"),
        _safe_ratio(pl.col("_eur_s"),         pl.col("_ttl"), "eur_tx_rate"),
        ((pl.col("_ts_max") - pl.col("_ts_min")).dt.total_seconds() / 86400)
        .fill_null(0.0).alias("tx_timespan_days"),
    ]).with_columns(
        pl.when(pl.col("_ttl") > 1)
        .then(pl.col("tx_timespan_days") * 24 / (pl.col("_ttl") - 1))
        .otherwise(0.0)
        .alias("avg_hours_between_tx")
    )

    if {"id_user", "timestamp_reg"} <= set(user_static.columns) and "timestamp_tr" in tx.columns:
        first_tx2 = tx.group_by("id_user").agg(
            pl.col("timestamp_tr").min().alias("_first_tx"))
        reg_delta = (
            user_static.select(["id_user", "timestamp_reg"])
            .join(first_tx2, on="id_user", how="left")
            .with_columns(
                ((pl.col("_first_tx") - pl.col("timestamp_reg")).dt.total_seconds() / 3600)
                .clip(lower_bound=0).fill_null(0.0).alias("delta_reg_to_first_tx_hours")
            )
            .select(["id_user", "delta_reg_to_first_tx_hours"])
        )
        temporal = temporal.join(reg_delta, on="id_user", how="left")
    else:
        temporal = temporal.with_columns(pl.lit(0.0).alias("delta_reg_to_first_tx_hours"))

    temporal = temporal.drop(["_eur_s", "_ttl", "_ts_max", "_ts_min"])

    # ── Entity features (cross-user graph signals) ─────────────────────────
    # Обчислюються глобально з транзакцій (без is_fraud) — безпечні для test.
    # Кожна фіча описує "мережевий" контекст юзера: скільки інших юзерів
    # ділять ту саму картку / card_holder / payment_country.

    entity_parts: list[pl.DataFrame] = []

    # 1. Скільки юзерів ділять одну картку (carding syndicate)
    if "card_mask_hash" in tx.columns:
        card_user_count = (
            tx.group_by("card_mask_hash")
            .agg(pl.col("id_user").n_unique().alias("_users_per_card"))
        )
        user_card_entity = (
            tx.select(["id_user", "card_mask_hash"]).unique()
            .join(card_user_count, on="card_mask_hash", how="left")
            .group_by("id_user")
            .agg([
                pl.col("_users_per_card").max().alias("max_users_per_card"),
                pl.col("_users_per_card").mean().alias("mean_users_per_card"),
                (pl.col("_users_per_card") > 1).any().cast(pl.Int8).alias("has_shared_card"),
                (pl.col("_users_per_card") > 5).any().cast(pl.Int8).alias("has_widely_shared_card"),
            ])
        )
        entity_parts.append(user_card_entity)

    # 2. Скільки юзерів за одним іменем card_holder (synthetic identity)
    if "card_holder" in tx.columns:
        holder_user_count = (
            tx.filter(pl.col("card_holder").is_not_null())
            .group_by("card_holder")
            .agg(pl.col("id_user").n_unique().alias("_users_per_holder"))
        )
        user_holder_entity = (
            tx.filter(pl.col("card_holder").is_not_null())
            .select(["id_user", "card_holder"]).unique()
            .join(holder_user_count, on="card_holder", how="left")
            .group_by("id_user")
            .agg([
                pl.col("_users_per_holder").max().alias("max_users_per_holder"),
                (pl.col("_users_per_holder") > 3).any().cast(pl.Int8).alias("has_shared_holder"),
            ])
        )
        entity_parts.append(user_holder_entity)

    # 3. Скільки унікальних карток використовувалось у payment_country юзера
    # (наскільки "гарячий" ринок — багато карток з одного регіону = carding hub)
    if {"card_mask_hash", "payment_country"} <= set(tx.columns):
        country_card_count = (
            tx.filter(pl.col("payment_country").is_not_null())
            .group_by("payment_country")
            .agg(pl.col("card_mask_hash").n_unique().alias("_cards_in_country"))
        )
        dominant_country = (
            tx.filter(pl.col("payment_country").is_not_null())
            .group_by(["id_user", "payment_country"]).len()
            .sort(["id_user", "len"], descending=[False, True])
            .group_by("id_user")
            .agg(pl.col("payment_country").first().alias("_dom_country"))
        )
        user_country_entity = (
            dominant_country
            .join(country_card_count, left_on="_dom_country", right_on="payment_country", how="left")
            .select(["id_user",
                     pl.col("_cards_in_country").alias("cards_in_dominant_country")])
        )
        entity_parts.append(user_country_entity)

    # ── Join all parts ─────────────────────────────────────────────────────
    out = (
        user_static
        .join(tx_stats,       on="id_user", how="left")
        .join(error_patterns, on="id_user", how="left")
        .join(geo,            on="id_user", how="left")
        .join(card,           on="id_user", how="left")
        .join(temporal,       on="id_user", how="left")
    )

    for entity_df in entity_parts:
        out = out.join(entity_df, on="id_user", how="left")

    # Fill numeric nulls with 0
    num_fill = [
        pl.col(c).fill_null(0).alias(c)
        for c, t in zip(out.columns, out.dtypes)
        if t in NUMERIC_DTYPES
    ]
    if num_fill:
        out = out.with_columns(num_fill)

    if "timestamp_reg" in out.columns:
        out = out.drop("timestamp_reg")

    # ── Add ratio features + risk indicators ──────────────────────────────
    out = add_ratio_features(out)
    out = build_risk_indicators(out)

    return out