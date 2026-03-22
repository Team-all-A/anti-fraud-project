"""
rule_based_filter.py  ·  SKELAR × mono AI Competition 2026
──────────────────────────────────────────────────────────────────────────────
АРХІТЕКТУРА: тільки WHITELIST як жорстке рішення.
  AUTOBLOCK = діагностика + rule_triggers фіча для ML (не hard block).
  Причина: precision AUTOBLOCK на повному датасеті ~24% → тисячі FP.

НОВІ СИГНАЛИ (з v3, відкалібровано на реальних даних):
  • tds_error_count >= 50   → scripted 3DS attack (у легітимних ≤43)
  • insufficient_funds >= 100 AND fail >= 0.50  → mass card-stuffing
  • antifraud >= 20 AND tds >= 20  → dual-system cross-confirmation
  • fail >= 0.75 AND (fraud_count >= 5 OR tds_count >= 5)  → confirmed high-fail

ПОКРАЩЕНІ ПОРОГИ (v2 → v3):
  A1: antifraud_count >= 10 AND fail >= 0.60  (було: >= 3 AND fail > 0.95)
  A5: score >= 4 AND fail >= 0.50             (було: score >= 6)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import polars as pl


NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}


# ══════════════════════════════════════════════════════════════════════════════
#  КОНФІГУРАЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RuleConfig:

    # ── Діагностичні правила (→ rule_triggers, НЕ блокують) ──────────────────

    # A1: Масовий antifraud + підвищений fail_rate
    a1_antifraud_count: int   = 10
    a1_fail_rate:       float = 0.60

    # A2: Scripted 3DS attack (у легітимних юзерів tds ≤ 43)
    a2_tds_count: int = 50

    # A3: Mass card-stuffing через insufficient funds
    a3_insuf_count: int   = 100
    a3_fail_rate:   float = 0.50

    # A4: Висока fail_rate + незалежне підтвердження
    a4_fail_rate:   float = 0.75
    a4_fraud_count: int   = 5
    a4_tds_count:   int   = 5

    # A5: Критичний rule_score + значуща fail_rate
    a5_score:     int   = 4
    a5_fail_rate: float = 0.50

    # A6: Dual-system cross-confirmation (antifraud + tds одночасно)
    a6_antifraud_count: int = 20
    a6_tds_count:       int = 20

    # ── WHITELIST (жорстке рішення) ───────────────────────────────────────────

    w1_max_score:        int   = 0
    w1_max_fail_rate:    float = 0.05
    w1_min_account_days: float = 30.0
    w1_min_tx_count:     int   = 5

    w2_max_error_div:  int   = 0
    w2_min_recurring:  int   = 10
    w2_max_init_ratio: float = 0.10
    w2_max_fail_rate:  float = 0.02


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_datetime_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
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


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    return (
        pl.when(denominator.fill_null(0) > 0)
        .then(numerator / denominator)
        .otherwise(0.0)
        .alias(alias)
    )


def build_flat_user_dataset(df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
    if "id_user" not in df.columns:
        raise ValueError("Input DataFrame must contain 'id_user'.")

    flat = df.clone()
    for col in ["timestamp_reg", "timestamp_tr"]:
        flat = _parse_datetime_col(flat, col)

    # ── Static user features ───────────────────────────────────────────────
    static_exprs = [pl.col("id_user")]
    if "reg_country"   in flat.columns: static_exprs.append(pl.col("reg_country"))
    if "traffic_type"  in flat.columns: static_exprs.append(pl.col("traffic_type"))
    if "gender"        in flat.columns: static_exprs.append(pl.col("gender"))
    if "timestamp_reg" in flat.columns:
        static_exprs.extend([
            pl.col("timestamp_reg"),
            pl.col("timestamp_reg").dt.hour().alias("reg_hour_of_day"),
            pl.col("timestamp_reg").dt.weekday().alias("reg_day_of_week"),
        ])
    if "email" in flat.columns:
        static_exprs.extend([
            pl.col("email").is_null().cast(pl.Int8).alias("email_is_null"),
            pl.col("email")
            .cast(pl.Utf8, strict=False)
            .str.extract(r"@(.+)$", 1)
            .fill_null("unknown")
            .alias("email_domain"),
        ])
    if is_train and "is_fraud" in flat.columns:
        static_exprs.append(pl.col("is_fraud"))

    user_static = flat.select(static_exprs).unique(subset=["id_user"], keep="first")
    tx = flat.filter(pl.col("timestamp_tr").is_not_null()) if "timestamp_tr" in flat.columns else flat.clone()

    # ── Row-level flags ────────────────────────────────────────────────────
    row_exprs: list[pl.Expr] = []
    if "status" in tx.columns:
        row_exprs.extend([
            (pl.col("status") == "success").cast(pl.Int8).alias("_ok"),
            (pl.col("status") == "fail").cast(pl.Int8).alias("_fail"),
        ])
    if "error_group" in tx.columns:
        err = pl.col("error_group").cast(pl.Utf8).fill_null("__none__")
        row_exprs.extend([
            (err == "antifraud").cast(pl.Int8).alias("_err_antifraud"),
            (err == "fraud").cast(pl.Int8).alias("_err_fraud"),
            (err == "insufficient funds error").cast(pl.Int8).alias("_err_insuf"),
            (err == "do not honor").cast(pl.Int8).alias("_err_dnh"),
            (err == "cvv error").cast(pl.Int8).alias("_err_cvv"),
            (err == "3ds error").cast(pl.Int8).alias("_err_3ds"),
            pl.col("error_group").is_not_null().cast(pl.Int8).alias("_has_error"),
        ])
    if {"card_country", "payment_country"}.issubset(tx.columns):
        row_exprs.append((pl.col("card_country") != pl.col("payment_country")).cast(pl.Int8).alias("_mm_cp"))
    if {"payment_country", "reg_country"}.issubset(tx.columns):
        row_exprs.append((pl.col("payment_country") != pl.col("reg_country")).cast(pl.Int8).alias("_mm_rp"))
    if {"card_country", "payment_country", "reg_country"}.issubset(tx.columns):
        row_exprs.append(
            ((pl.col("card_country") != pl.col("payment_country")) &
             (pl.col("payment_country") != pl.col("reg_country")))
            .cast(pl.Int8).alias("_mm_triple")
        )
    if "card_type"        in tx.columns: row_exprs.append((pl.col("card_type")  == "CREDIT").cast(pl.Int8).alias("_is_credit"))
    if "card_brand"       in tx.columns: row_exprs.append((pl.col("card_brand") == "VISA").cast(pl.Int8).alias("_is_visa"))
    if "transaction_type" in tx.columns:
        row_exprs.extend([
            (pl.col("transaction_type") != "google-pay").cast(pl.Int8).alias("_non_gpay"),
            (pl.col("transaction_type") == "card_init").cast(pl.Int8).alias("_init"),
            (pl.col("transaction_type") == "card_recurring").cast(pl.Int8).alias("_recurring"),
            (pl.col("transaction_type") == "google-pay").cast(pl.Int8).alias("_gpay"),
        ])
    if {"transaction_type", "card_holder"}.issubset(tx.columns):
        row_exprs.append(
            ((pl.col("transaction_type") != "google-pay") & pl.col("card_holder").is_null())
            .cast(pl.Int8).alias("_null_holder_non_gpay")
        )
    if "timestamp_tr" in tx.columns:
        row_exprs.append(
            pl.col("timestamp_tr").dt.hour().is_between(0, 5, closed="both")
            .cast(pl.Int8).alias("_night")
        )
    if "currency" in tx.columns:
        row_exprs.append((pl.col("currency") == "EUR").cast(pl.Int8).alias("_eur"))
    if row_exprs:
        tx = tx.with_columns(row_exprs)

    # ── Transaction stats ──────────────────────────────────────────────────
    tx_stats = tx.group_by("id_user").agg([
        pl.len().alias("tx_total_count"),
        pl.col("_ok").sum().alias("tx_success_count")   if "_ok"    in tx.columns else pl.lit(0).alias("tx_success_count"),
        pl.col("_fail").sum().alias("tx_fail_count")    if "_fail"  in tx.columns else pl.lit(0).alias("tx_fail_count"),
        pl.col("amount").sum().alias("tx_amount_sum")   if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_sum"),
        pl.col("amount").mean().alias("tx_amount_mean") if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_mean"),
        pl.col("amount").max().alias("tx_amount_max")   if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_max"),
        pl.col("amount").min().alias("tx_amount_min")   if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_min"),
        pl.col("amount").std().alias("tx_amount_std")   if "amount" in tx.columns else pl.lit(0.0).alias("tx_amount_std"),
    ]).with_columns([
        pl.col("tx_amount_std").fill_null(0.0),
        _safe_ratio(pl.col("tx_fail_count"), pl.col("tx_total_count"), "tx_fail_rate"),
    ])

    # ── Error patterns ─────────────────────────────────────────────────────
    error_aggs = []
    if "_err_antifraud" in tx.columns:
        error_aggs.extend([pl.col("_err_antifraud").max().alias("has_antifraud_error"),
                           pl.col("_err_antifraud").sum().alias("antifraud_error_count")])
    else:
        error_aggs.extend([pl.lit(0).alias("has_antifraud_error"), pl.lit(0).alias("antifraud_error_count")])

    if "_err_fraud" in tx.columns:
        error_aggs.extend([pl.col("_err_fraud").max().alias("has_fraud_error"),
                           pl.col("_err_fraud").sum().alias("fraud_error_count")])
    else:
        error_aggs.extend([pl.lit(0).alias("has_fraud_error"), pl.lit(0).alias("fraud_error_count")])

    error_aggs.extend([
        pl.col("_err_insuf").sum().alias("insufficient_funds_count") if "_err_insuf" in tx.columns else pl.lit(0).alias("insufficient_funds_count"),
        pl.col("_err_dnh").sum().alias("do_not_honor_count")         if "_err_dnh"   in tx.columns else pl.lit(0).alias("do_not_honor_count"),
        pl.col("_err_cvv").sum().alias("cvv_error_count")            if "_err_cvv"   in tx.columns else pl.lit(0).alias("cvv_error_count"),
        pl.col("_err_3ds").sum().alias("tds_error_count")            if "_err_3ds"   in tx.columns else pl.lit(0).alias("tds_error_count"),
    ])

    error_patterns = tx.group_by("id_user").agg(error_aggs)
    if {"id_user", "error_group", "_has_error"}.issubset(tx.columns):
        diversity = (
            tx.filter(pl.col("_has_error") == 1)
            .group_by("id_user")
            .agg(pl.col("error_group").n_unique().alias("error_diversity_count"))
        )
        error_patterns = error_patterns.join(diversity, on="id_user", how="left")
    error_patterns = error_patterns.with_columns(
        pl.col("error_diversity_count").fill_null(0).cast(pl.Int32)
    )

    # ── Geo ────────────────────────────────────────────────────────────────
    geo_aggs = [
        pl.col("_mm_cp").max().alias("geo_mismatch_card_payment")    if "_mm_cp"     in tx.columns else pl.lit(0).alias("geo_mismatch_card_payment"),
        pl.col("_mm_rp").max().alias("geo_mismatch_reg_payment")     if "_mm_rp"     in tx.columns else pl.lit(0).alias("geo_mismatch_reg_payment"),
        pl.col("_mm_triple").max().alias("geo_mismatch_triple")      if "_mm_triple" in tx.columns else pl.lit(0).alias("geo_mismatch_triple"),
        pl.col("payment_country").n_unique().alias("unique_payment_countries_count") if "payment_country" in tx.columns else pl.lit(0).alias("unique_payment_countries_count"),
        pl.col("card_country").n_unique().alias("unique_card_countries_count")       if "card_country"    in tx.columns else pl.lit(0).alias("unique_card_countries_count"),
    ]
    geo = tx.group_by("id_user").agg(geo_aggs)
    if "payment_country" in tx.columns:
        dominant_payment = (
            tx.group_by(["id_user", "payment_country"]).len()
            .sort(["id_user", "len"], descending=[False, True])
            .group_by("id_user")
            .agg(pl.col("payment_country").first().alias("dominant_payment_country"))
        )
        geo = geo.join(dominant_payment, on="id_user", how="left")

    # ── Card / velocity ────────────────────────────────────────────────────
    card_aggs = [
        pl.col("card_mask_hash").n_unique().alias("unique_cards_count")          if "card_mask_hash"        in tx.columns else pl.lit(0).alias("unique_cards_count"),
        pl.col("_is_credit").sum().alias("_credit_sum")                          if "_is_credit"            in tx.columns else pl.lit(0).alias("_credit_sum"),
        pl.col("_is_visa").sum().alias("_visa_sum")                              if "_is_visa"              in tx.columns else pl.lit(0).alias("_visa_sum"),
        pl.col("_non_gpay").sum().alias("_non_gpay_total")                       if "_non_gpay"             in tx.columns else pl.lit(0).alias("_non_gpay_total"),
        pl.col("_null_holder_non_gpay").sum().alias("_null_holder_non_gpay_sum") if "_null_holder_non_gpay" in tx.columns else pl.lit(0).alias("_null_holder_non_gpay_sum"),
        pl.len().alias("_total"),
    ]
    card = tx.group_by("id_user").agg(card_aggs).with_columns([
        _safe_ratio(pl.col("_credit_sum"),               pl.col("_total"),          "credit_card_rate"),
        _safe_ratio(pl.col("_visa_sum"),                 pl.col("_total"),          "visa_rate"),
        _safe_ratio(pl.col("_null_holder_non_gpay_sum"), pl.col("_non_gpay_total"), "card_holder_is_null_rate"),
    ])
    if "card_holder" in tx.columns:
        holder_uniq = (
            tx.filter(pl.col("card_holder").is_not_null())
            .group_by("id_user")
            .agg(pl.col("card_holder").n_unique().alias("unique_card_holders_count"))
        )
        card = card.join(holder_uniq, on="id_user", how="left")

    first_tx = (
        tx.group_by("id_user").agg(pl.col("timestamp_tr").min().alias("_first_tx"))
        if "timestamp_tr" in tx.columns
        else pl.DataFrame({"id_user": tx.get_column("id_user").unique()})
    )
    age = user_static.select([c for c in ["id_user", "timestamp_reg"] if c in user_static.columns]).join(first_tx, on="id_user", how="left")
    if {"timestamp_reg", "_first_tx"}.issubset(age.columns):
        age = age.with_columns(
            (((pl.col("_first_tx") - pl.col("timestamp_reg")).dt.total_seconds() / 86400)
             .clip(lower_bound=0).fill_null(0))
            .alias("account_age_days")
        ).select(["id_user", "account_age_days"])
    else:
        age = age.select("id_user").with_columns(pl.lit(0.0).alias("account_age_days"))

    card = card.join(age, on="id_user", how="left").with_columns([
        pl.col("unique_card_holders_count").fill_null(0).cast(pl.Int32),
        pl.col("account_age_days").fill_null(0.0),
        pl.when(pl.col("account_age_days") > 0)
        .then(pl.col("unique_cards_count") / pl.col("account_age_days"))
        .otherwise(pl.col("unique_cards_count").cast(pl.Float64))
        .alias("cards_per_day"),
    ]).drop(["_credit_sum", "_visa_sum", "_non_gpay_total", "_null_holder_non_gpay_sum", "_total"])

    # ── Temporal ───────────────────────────────────────────────────────────
    temporal_aggs = [
        pl.col("_init").sum().alias("card_init_count")           if "_init"      in tx.columns else pl.lit(0).alias("card_init_count"),
        pl.col("_recurring").sum().alias("card_recurring_count") if "_recurring" in tx.columns else pl.lit(0).alias("card_recurring_count"),
        pl.col("_gpay").sum().alias("google_pay_count")          if "_gpay"      in tx.columns else pl.lit(0).alias("google_pay_count"),
        pl.col("_night").sum().alias("night_tx_count")           if "_night"     in tx.columns else pl.lit(0).alias("night_tx_count"),
        pl.col("_eur").sum().alias("_eur_sum")                   if "_eur"       in tx.columns else pl.lit(0).alias("_eur_sum"),
        pl.len().alias("_total"),
        pl.col("timestamp_tr").max().alias("_ts_max")            if "timestamp_tr" in tx.columns else pl.lit(None).alias("_ts_max"),
        pl.col("timestamp_tr").min().alias("_ts_min")            if "timestamp_tr" in tx.columns else pl.lit(None).alias("_ts_min"),
    ]
    temporal = tx.group_by("id_user").agg(temporal_aggs).with_columns([
        (pl.col("card_init_count") / (pl.col("card_recurring_count") + 1)).alias("init_to_recurring_ratio"),
        _safe_ratio(pl.col("night_tx_count"), pl.col("_total"), "night_tx_rate"),
        _safe_ratio(pl.col("_eur_sum"),        pl.col("_total"), "eur_tx_rate"),
        (((pl.col("_ts_max") - pl.col("_ts_min")).dt.total_seconds() / 86400).fill_null(0.0)).alias("tx_timespan_days"),
    ]).with_columns([
        pl.when(pl.col("_total") > 1)
        .then(pl.col("tx_timespan_days") * 24 / (pl.col("_total") - 1))
        .otherwise(0.0)
        .alias("avg_hours_between_tx")
    ])

    if {"id_user", "timestamp_reg"}.issubset(user_static.columns) and "timestamp_tr" in tx.columns:
        first_tx2 = tx.group_by("id_user").agg(pl.col("timestamp_tr").min().alias("_first_tx"))
        reg_delta = user_static.select(["id_user", "timestamp_reg"]).join(first_tx2, on="id_user", how="left")
        reg_delta = reg_delta.with_columns(
            (((pl.col("_first_tx") - pl.col("timestamp_reg")).dt.total_seconds() / 3600)
             .clip(lower_bound=0).fill_null(0.0))
            .alias("delta_reg_to_first_tx_hours")
        ).select(["id_user", "delta_reg_to_first_tx_hours"])
        temporal = temporal.join(reg_delta, on="id_user", how="left")
    else:
        temporal = temporal.with_columns(pl.lit(0.0).alias("delta_reg_to_first_tx_hours"))

    temporal = temporal.drop(["_eur_sum", "_total", "_ts_max", "_ts_min"])

    # ── Join ───────────────────────────────────────────────────────────────
    out = (
        user_static
        .join(tx_stats,       on="id_user", how="left")
        .join(error_patterns, on="id_user", how="left")
        .join(geo,            on="id_user", how="left")
        .join(card,           on="id_user", how="left")
        .join(temporal,       on="id_user", how="left")
    )
    numeric_fill = [
        pl.col(col).fill_null(0).alias(col)
        for col, dtype in zip(out.columns, out.dtypes)
        if dtype in NUMERIC_DTYPES
    ]
    if numeric_fill:
        out = out.with_columns(numeric_fill)
    if "timestamp_reg" in out.columns:
        out = out.drop("timestamp_reg")

    return build_risk_indicators(out)


def build_risk_indicators(flat: pl.DataFrame) -> pl.DataFrame:
    df = flat.with_columns([
        (
            (pl.col("tx_amount_min") < 1.0) &
            (pl.col("tx_fail_count") >= 2)
        ).cast(pl.Int8).alias("micro_payment_flag"),
        (
            (pl.col("unique_cards_count") >= 3) &
            (pl.col("tx_fail_rate") > 0.80) &
            (pl.col("avg_hours_between_tx") < 2.0)
        ).cast(pl.Int8).alias("card_testing_flag"),
    ]).with_columns([
        (
            pl.col("has_antifraud_error").fill_null(0).cast(pl.Int64) +
            pl.col("has_fraud_error").fill_null(0).cast(pl.Int64) +
            pl.col("geo_mismatch_triple").fill_null(0).cast(pl.Int64) +
            (pl.col("unique_cards_count") >= 3).cast(pl.Int64) +
            (pl.col("tx_fail_rate") > 0.80).cast(pl.Int64) +
            pl.col("micro_payment_flag").cast(pl.Int64)
        ).cast(pl.Int8).alias("rule_score")
    ])
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED FILTER
# ══════════════════════════════════════════════════════════════════════════════

def apply_rule_based_filter(
    df: pl.DataFrame,
    cfg: RuleConfig | None = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Routing: тільки WHITELIST як жорстке рішення.
    Всі інші (включно з AUTOBLOCK-кандидатами) → SEND_TO_ML.
    rule_triggers — аудит + потенційна фіча для ML.
    """
    if cfg is None:
        cfg = RuleConfig()

    out = df.with_columns([

        # ── Діагностичні AUTOBLOCK правила (v3 пороги) ────────────────────
        pl.when(
            (pl.col("antifraud_error_count") >= cfg.a1_antifraud_count) &
            (pl.col("tx_fail_rate")          >= cfg.a1_fail_rate)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a1"),

        pl.when(
            pl.col("tds_error_count") >= cfg.a2_tds_count
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a2"),

        pl.when(
            (pl.col("insufficient_funds_count") >= cfg.a3_insuf_count) &
            (pl.col("tx_fail_rate")             >= cfg.a3_fail_rate)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a3"),

        pl.when(
            (pl.col("tx_fail_rate") >= cfg.a4_fail_rate) &
            (
                (pl.col("fraud_error_count") >= cfg.a4_fraud_count) |
                (pl.col("tds_error_count")   >= cfg.a4_tds_count)
            )
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a4"),

        pl.when(
            (pl.col("rule_score")   >= cfg.a5_score) &
            (pl.col("tx_fail_rate") >= cfg.a5_fail_rate)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a5"),

        pl.when(
            (pl.col("antifraud_error_count") >= cfg.a6_antifraud_count) &
            (pl.col("tds_error_count")       >= cfg.a6_tds_count)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_a6"),

        # ── WHITELIST ──────────────────────────────────────────────────────
        pl.when(
            (pl.col("rule_score")       == cfg.w1_max_score) &
            (pl.col("tx_fail_rate")     <= cfg.w1_max_fail_rate) &
            (pl.col("account_age_days") >= cfg.w1_min_account_days) &
            (pl.col("tx_total_count")   >= cfg.w1_min_tx_count)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_w1"),

        pl.when(
            (pl.col("error_diversity_count")   == cfg.w2_max_error_div) &
            (pl.col("card_recurring_count")    >= cfg.w2_min_recurring) &
            (pl.col("init_to_recurring_ratio") <= cfg.w2_max_init_ratio) &
            (pl.col("tx_fail_rate")            <= cfg.w2_max_fail_rate)
        ).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8)).alias("_w2"),
    ])

    autoblock_cols = ["_a1", "_a2", "_a3", "_a4", "_a5", "_a6"]
    whitelist_cols = ["_w1", "_w2"]

    label_map: dict[str, str] = {
        "_a1": "A1:mass_antifraud",
        "_a2": "A2:scripted_3ds",
        "_a3": "A3:insuf_stuffing",
        "_a4": "A4:fail_confirmed",
        "_a5": "A5:score_plus_fail",
        "_a6": "A6:dual_system",
        "_w1": "W1:stable_clean",
        "_w2": "W2:subscriber",
    }

    trigger_exprs = [
        pl.when(pl.col(col) == 1).then(pl.lit(lbl)).otherwise(pl.lit(""))
        for col, lbl in label_map.items()
    ]

    out = out.with_columns([
        pl.sum_horizontal(autoblock_cols).alias("_ab_total"),
        pl.sum_horizontal(whitelist_cols).alias("_wl_total"),
        pl.concat_str(trigger_exprs, separator="|")
        .str.replace_all(r"\|+", "|")
        .str.strip_chars("|")
        .alias("rule_triggers"),
    ])

    # Тільки WHITELIST — жорстке рішення. Всі інші → SEND_TO_ML.
    out = out.with_columns(
        pl.when(pl.col("_wl_total") > 0)
        .then(pl.lit("WHITELIST"))
        .otherwise(pl.lit("SEND_TO_ML"))
        .alias("rule_decision")
    )

    out = out.drop(autoblock_cols + whitelist_cols + ["_ab_total", "_wl_total"])

    if verbose:
        total = out.height
        dist  = out.group_by("rule_decision").len().sort("rule_decision")

        print("\n[RULE FILTER] Routing distribution")
        for decision, count in dist.iter_rows():
            pct = 100.0 * count / max(total, 1)
            bar = "█" * int(pct / 3)
            print(f"  {decision:<12} {count:>8,} ({pct:5.1f}%)  {bar}")

        if "is_fraud" in out.columns:
            total_fraud = out.filter(pl.col("is_fraud") == 1).height

            ab_would = out.filter(pl.col("rule_triggers").str.len_chars() > 0)
            if ab_would.height > 0:
                tp        = ab_would.filter(pl.col("is_fraud") == 1).height
                fp        = ab_would.filter(pl.col("is_fraud") == 0).height
                precision = tp / max(tp + fp, 1)
                recall    = tp / max(total_fraud, 1)
                print(f"\n[RULE FILTER] AUTOBLOCK diagnostic (NOT applied): precision={precision:.4f}  recall={recall:.4f}  (TP={tp} FP={fp})")
                print("  → All routed to SEND_TO_ML")

                print(f"\n[RULE FILTER] Per-trigger breakdown:")
                print(f"  {'Rule':<24} {'Fired':>7} {'TP':>7} {'FP':>7} {'Prec':>7}")
                for col, label in label_map.items():
                    if label.startswith("W"):
                        continue
                    fired = out.filter(pl.col("rule_triggers").str.contains(label, literal=True))
                    if fired.height == 0:
                        print(f"  {label:<24} {'0':>7}")
                        continue
                    t_tp   = fired.filter(pl.col("is_fraud") == 1).height
                    t_fp   = fired.filter(pl.col("is_fraud") == 0).height
                    t_prec = t_tp / max(t_tp + t_fp, 1)
                    print(f"  {label:<24} {fired.height:>7,} {t_tp:>7,} {t_fp:>7,} {t_prec:>7.3f}")

            wl = out.filter(pl.col("rule_decision") == "WHITELIST")
            if wl.height > 0:
                wl_fn   = wl.filter(pl.col("is_fraud") == 1).height
                wl_tn   = wl.filter(pl.col("is_fraud") == 0).height
                wl_prec = wl_tn / max(wl.height, 1)
                print(f"\n[RULE FILTER] WHITELIST: precision(legit)={wl_prec:.4f}  (TN={wl_tn} FN={wl_fn})")

            ml = out.filter(pl.col("rule_decision") == "SEND_TO_ML")
            if ml.height > 0:
                gfr  = float(out["is_fraud"].mean())
                mfr  = float(ml["is_fraud"].mean())
                dir_ = "↑" if mfr >= gfr else "↓"
                print(f"\n[RULE FILTER] Fraud rate: global={gfr:.4f} → ML-zone={mfr:.4f} {dir_}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ML ZONE + SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_ml_zone(
    df: pl.DataFrame,
    is_train: bool = True,
) -> tuple[pl.DataFrame, np.ndarray | None]:
    ml_df = df.filter(pl.col("rule_decision") == "SEND_TO_ML")
    if ml_df.height == 0:
        return ml_df, None
    y: np.ndarray | None = None
    if is_train and "is_fraud" in ml_df.columns:
        y = ml_df.get_column("is_fraud").to_numpy()
    return ml_df, y


def build_submission(
    test_filtered: pl.DataFrame,
    ml_proba: np.ndarray | None,
    best_threshold: float = 0.5,
) -> pl.DataFrame:
    """WHITELIST → 0, SEND_TO_ML → ML predictions."""
    parts: list[pl.DataFrame] = []

    parts.append(
        test_filtered
        .filter(pl.col("rule_decision") == "WHITELIST")
        .select("id_user")
        .with_columns(pl.lit(0, dtype=pl.Int32).alias("is_fraud"))
    )

    ml_ids = test_filtered.filter(pl.col("rule_decision") == "SEND_TO_ML").select("id_user")
    if ml_ids.height > 0:
        if ml_proba is None:
            raise ValueError(f"ml_proba is None but {ml_ids.height} users routed to SEND_TO_ML.")
        parts.append(
            ml_ids.with_columns(
                pl.Series("is_fraud", (ml_proba >= best_threshold).astype(np.int32), dtype=pl.Int32)
            )
        )

    submission = pl.concat(parts).sort("id_user")
    print(
        f"\n[SUBMISSION] {submission.height:,} rows | "
        f"predicted fraud rate = {submission['is_fraud'].mean():.4f}"
    )
    return submission