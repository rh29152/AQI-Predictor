"""
feature_engineering.py — Build ML features from raw stored data.

Pollutant-first approach
------------------------
This module creates lag, rolling, and target features for the 4 forecast
pollutants (PM2.5, PM10, O3, NO2).  Final AQI is NOT a target here — it is
computed in aqi_utils.py from predicted pollutant concentrations.

Two operating modes
-------------------
Batch (backfill)
    Called once during initial backfill / rebuild.  Engineers features + all
    12 target labels (4 pollutants × 3 horizons) for the full historical
    dataset.  Rows where any lag or target value is missing are dropped.

    Entry point: run_feature_pipeline()
    Caller:      backfill.py

Incremental (hourly)
    Called every hour after fetch_openweather.py stores the latest raw record.
    Reads only the last ~73 raw records (enough for 48-h lags + 24-h rolling)
    and engineers ONE feature row for the newest timestamp.  Target columns
    are NOT added because future concentrations are unknown.

    Entry point: run_incremental_pipeline()
    Caller:      hourly_pipeline.py

Run standalone (incremental):
    python src/feature_engineering.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import (
    get_collection,
    get_raw_history,
    insert_features,
    insert_features_batch,
    ensure_indexes,
)
from src.config import RAW_COLLECTION, POLLUTANTS_TO_FORECAST, FORECAST_HORIZONS
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Context size ───────────────────────────────────────────────────────────────
# lag_48 needs 48 prior rows; rolling_24 needs 24 prior rows.
# 48 (lag) + 24 (rolling window) + 1 (new row) = 73 minimum.
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
    Rename the OpenWeather 'aqi' column to 'aqi_category' so it is clearly
    an input feature, not the ML target.  The original 'aqi' column is kept
    for backward-compat raw_data displays (e.g. EDA).
    """
    if "aqi" in df.columns:
        df["aqi_category"] = df["aqi"]
    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features for all 4 forecast pollutants.

    For each pollutant (pm2_5, pm10, o3, no2):
      - lag_1  : concentration 1 hour ago
      - lag_24 : concentration 24 hours ago
      - lag_48 : concentration 48 hours ago
      - rolling_6_mean  : 6-hour rolling mean (shift-1 to avoid leakage)
      - rolling_12_mean : 12-hour rolling mean
      - rolling_24_mean : 24-hour rolling mean
      - change_rate     : % change vs 24 h ago

    df must be sorted ascending by datetime.  Mutates df in-place.
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
    Add future pollutant target columns by shifting concentrations forward.

    Only valid when the full dataset is available (batch/backfill mode).
    The last 72 rows will have NaN targets and are dropped afterwards.

    Creates 12 targets: 4 pollutants × 3 horizons (24h, 48h, 72h).
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
    Build a complete feature DataFrame from a full raw history.

    Includes all lag, rolling, and target columns for all 4 pollutants.
    Drops rows where any required lag or target value is missing (the first
    48 and last 72 hours of the window cannot be fully populated).

    Parameters
    ----------
    df : pd.DataFrame
        All raw records sorted ascending by datetime.

    Returns
    -------
    pd.DataFrame
        Training-ready feature rows with all 12 target columns.
    """
    feat = df.copy()
    feat = _add_time_features(feat)
    feat = _add_aqi_category(feat)
    feat = _add_lag_rolling_features(feat)
    feat = _add_target_features(feat)

    # Require all lag and target columns to be non-null
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
        # Remove NaN values to keep documents clean
        doc = {k: v for k, v in doc.items() if v == v}  # filters out float NaN
        records.append(doc)
    count = insert_features_batch(records)
    logger.info("Upserted %d feature records into MongoDB.", count)


def run_feature_pipeline() -> pd.DataFrame:
    """
    BATCH entry point — called by backfill.py (full build or --rebuild-features).

    Loads ALL raw records → engineers full features + all 12 pollutant targets
    → stores all rows in the features collection.
    Do NOT call this from the hourly pipeline.
    """
    raw_df  = load_raw_data()
    feat_df = build_features_batch(raw_df)
    store_features(feat_df)
    return feat_df


# ── Incremental mode (hourly) ──────────────────────────────────────────────────

def build_features_incremental(context_df: pd.DataFrame) -> dict:
    """
    Build ONE feature row for the newest record in context_df.

    context_df must be sorted ascending by datetime and contain at least
    LAG_CONTEXT_ROWS entries.  The last row is the new record.
    No target columns are added — future concentrations are not known yet.

    Parameters
    ----------
    context_df : pd.DataFrame
        Last LAG_CONTEXT_ROWS raw records, sorted oldest → newest.

    Returns
    -------
    dict
        Feature dict for the newest timestamp, ready for MongoDB upsert.
    """
    feat = context_df.copy()
    feat = _add_time_features(feat)
    feat = _add_aqi_category(feat)
    feat = _add_lag_rolling_features(feat)
    # No target features — future pollutant values are unavailable

    latest = feat.iloc[-1].to_dict()
    if hasattr(latest["datetime"], "to_pydatetime"):
        latest["datetime"] = latest["datetime"].to_pydatetime()
    # Remove NaN values to keep the document clean
    latest = {k: v for k, v in latest.items() if v == v}
    return latest


def run_incremental_pipeline() -> dict:
    """
    INCREMENTAL entry point — called by hourly_pipeline.py.

    1. Reads the last LAG_CONTEXT_ROWS raw records from MongoDB.
    2. Computes features for the newest (latest) record only.
    3. Upserts that single feature row into the features collection.

    Returns
    -------
    dict
        The feature row that was upserted.
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

    feature_row = build_features_incremental(context_df)
    insert_features(feature_row)

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
