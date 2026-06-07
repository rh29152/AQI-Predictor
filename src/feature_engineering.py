"""
feature_engineering.py — Feature construction from raw air-quality records.

Transforms raw pollutant and weather fields into model inputs: calendar features,
per-pollutant lags, rolling means, and change rates. Final EPA AQI is derived
downstream in aqi_utils.py rather than modelled directly.

Two modes coexist:
  • Batch — full history with 12 supervised targets (backfill / rebuild path).
  • Incremental — one live row from recent raw context; targets omitted when
    future concentrations are unavailable (hourly path).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import (
    get_collection,
    get_raw_history,
    get_raw_history_up_to,
    insert_features,
    insert_features_batch,
    ensure_indexes,
)
from src.config import RAW_COLLECTION, POLLUTANTS_TO_FORECAST, FORECAST_HORIZONS
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Minimum raw context: 48 h lag + 24 h rolling window + current row = 73
LAG_CONTEXT_ROWS = 73


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """Load ALL raw records from MongoDB (for batch/backfill use)."""
    col = get_collection(RAW_COLLECTION)
    docs = list(col.find({}, {"_id": 0}).sort("datetime", 1))
    if not docs:
        raise ValueError("No raw data in MongoDB. Run backfill.py first.")
    df = pd.DataFrame(docs)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d raw records (batch).", len(df))
    return df


# ── Core feature builders ─────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / time-of-day features. Mutates df in-place."""
    df["hour"]       = df["datetime"].dt.hour
    df["day"]        = df["datetime"].dt.day
    df["month"]      = df["datetime"].dt.month
    df["weekday"]    = df["datetime"].dt.weekday   # 0 = Monday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def _add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps OpenWeather `aqi` (1–5) to `aqi_category` as a model input feature.
    The original `aqi` column is retained for raw_data and EDA compatibility.
    """
    if "aqi" in df.columns:
        df["aqi_category"] = df["aqi"]
    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pollutant lag, rolling-mean, and change-rate features for pm2_5, pm10,
    o3, and no2. Rolling windows use shift(1) to avoid same-hour leakage.
    Requires ascending datetime order; mutates the frame in place.
    """
    for poll in POLLUTANTS_TO_FORECAST:
        if poll not in df.columns:
            logger.warning("Pollutant '%s' not in DataFrame; skipping lag features.", poll)
            continue

        df[f"{poll}_lag_1"]  = df[poll].shift(1)
        df[f"{poll}_lag_24"] = df[poll].shift(24)
        df[f"{poll}_lag_48"] = df[poll].shift(48)

        shifted = df[poll].shift(1)
        df[f"{poll}_rolling_6_mean"]  = shifted.rolling(window=6,  min_periods=3).mean()
        df[f"{poll}_rolling_12_mean"] = shifted.rolling(window=12, min_periods=6).mean()
        df[f"{poll}_rolling_24_mean"] = shifted.rolling(window=24, min_periods=12).mean()

        df[f"{poll}_change_rate"] = (
            (df[poll] - df[f"{poll}_lag_24"])
            / df[f"{poll}_lag_24"].replace(0, float("nan"))
        )
    return df


def _add_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-shifted pollutant concentrations as supervised targets (batch mode).

    Trailing 72 rows lack future values and are dropped in build_features_batch.
    Produces twelve columns: four pollutants × 24/48/72 h horizons.
    """
    for poll in POLLUTANTS_TO_FORECAST:
        if poll not in df.columns:
            continue
        for h in FORECAST_HORIZONS:
            df[f"target_{poll}_{h}h"] = df[poll].shift(-h)
    return df


# ── Batch mode (backfill / rebuild) ───────────────────────────────────────────

def build_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature matrix from raw history including lags, rolling stats, and targets.

    Rows at the window edges (first ~48 h, last ~72 h) are dropped when lags
    or forward targets cannot be computed.
    """
    feat = df.copy()
    feat = _add_time_features(feat)
    feat = _add_aqi_category(feat)
    feat = _add_lag_rolling_features(feat)
    feat = _add_target_features(feat)

    # Rows missing any required lag or target column are dropped
    must_be_present: list[str] = []
    for poll in POLLUTANTS_TO_FORECAST:
        must_be_present += [
            f"{poll}_lag_1", f"{poll}_lag_24", f"{poll}_lag_48",
            f"{poll}_rolling_24_mean",
        ]
        for h in FORECAST_HORIZONS:
            must_be_present.append(f"target_{poll}_{h}h")

    before = len(feat)
    feat.dropna(subset=must_be_present, inplace=True)
    feat.reset_index(drop=True, inplace=True)
    logger.info(
        "Batch: dropped %d rows (NaN lag/target); %d usable rows remain.",
        before - len(feat), len(feat),
    )
    return feat


def store_features(feat: pd.DataFrame) -> None:
    """Upsert a DataFrame of feature rows into MongoDB."""
    records = []
    for _, row in feat.iterrows():
        doc = row.to_dict()
        if hasattr(doc["datetime"], "to_pydatetime"):
            doc["datetime"] = doc["datetime"].to_pydatetime()
        # Omit NaN fields from stored documents
        doc = {k: v for k, v in doc.items() if v == v}  # filters out float NaN
        records.append(doc)
    count = insert_features_batch(records)
    logger.info("Upserted %d feature records into MongoDB.", count)


def run_feature_pipeline() -> pd.DataFrame:
    """
    Batch pipeline: materialise labelled features for the full raw history.

    Used by backfill and feature-rebuild flows; the hourly path relies on
    run_incremental_pipeline instead.
    """
    raw_df  = load_raw_data()
    feat_df = build_features_batch(raw_df)
    store_features(feat_df)
    return feat_df


# ── Incremental mode (hourly) ──────────────────────────────────────────────────

def build_features_incremental(context_df: pd.DataFrame) -> dict:
    """
    Single feature dict for the newest row in context_df.

    Expects ascending datetime order and at least LAG_CONTEXT_ROWS rows.
    Target columns are omitted — future concentrations are unknown at ingest time.
    """
    feat = context_df.copy()
    feat = _add_time_features(feat)
    feat = _add_aqi_category(feat)
    feat = _add_lag_rolling_features(feat)
    # Supervised targets excluded in incremental mode

    latest = feat.iloc[-1].to_dict()
    if hasattr(latest["datetime"], "to_pydatetime"):
        latest["datetime"] = latest["datetime"].to_pydatetime()
    # Omit NaN fields from stored document
    latest = {k: v for k, v in latest.items() if v == v}
    return latest


def _floor_to_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def _build_incremental_from_context(context_df: pd.DataFrame) -> dict:
    """Shared helper: build and upsert one incremental feature row."""
    feature_row = build_features_incremental(context_df)
    insert_features(feature_row)
    return feature_row


def run_incremental_for_datetime(target_dt: datetime) -> dict | None:
    """
    Feature row for a specific hourly timestamp (catch-up path).

    Loads raw context up to target_dt; returns None when context is insufficient
    or no raw row exists at that hour.
    """
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    target_hour = _floor_to_hour(target_dt)

    context = get_raw_history_up_to(target_hour, n=LAG_CONTEXT_ROWS)
    if len(context) < 2:
        logger.warning(
            "Catch-up: insufficient raw context for %s — skipping feature row.",
            target_hour,
        )
        return None

    context_df = pd.DataFrame(context)
    context_df["datetime"] = pd.to_datetime(context_df["datetime"], utc=True)
    context_df.sort_values("datetime", inplace=True)
    context_df.reset_index(drop=True, inplace=True)

    last_dt = context_df.iloc[-1]["datetime"]
    if _floor_to_hour(last_dt.to_pydatetime()) != target_hour:
        logger.warning(
            "Catch-up: no raw row at %s (latest raw in context=%s) — skipping.",
            target_hour,
            last_dt,
        )
        return None

    feature_row = _build_incremental_from_context(context_df)
    logger.info(
        "Catch-up: upserted feature row — datetime=%s  PM2.5=%.1f",
        feature_row.get("datetime"),
        feature_row.get("pm2_5", float("nan")),
    )
    return feature_row


def run_incremental_pipeline() -> dict:
    """
    Incremental pipeline for the latest raw snapshot.

    Pulls LAG_CONTEXT_ROWS of history, engineers the trailing row, and upserts
    one document into the features collection.
    """
    context = get_raw_history(n=LAG_CONTEXT_ROWS)
    if len(context) < 2:
        raise ValueError(
            "Not enough raw data to compute lag features. "
            "Run backfill.py first, or wait for more hourly records."
        )

    context_df = pd.DataFrame(context)
    context_df["datetime"] = pd.to_datetime(context_df["datetime"], utc=True)
    context_df.sort_values("datetime", inplace=True)
    context_df.reset_index(drop=True, inplace=True)

    feature_row = _build_incremental_from_context(context_df)

    logger.info(
        "Incremental: upserted 1 feature row — datetime=%s  PM2.5=%.1f",
        feature_row.get("datetime"),
        feature_row.get("pm2_5", float("nan")),
    )
    return feature_row


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_indexes()
    result = run_incremental_pipeline()
    logger.info("Incremental feature pipeline complete for %s.", result.get("datetime"))
