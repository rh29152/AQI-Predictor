"""
feature_engineering.py — Build ML features from raw stored data.

Two operating modes
-------------------
Batch (backfill)
    Called once during initial backfill to engineer features + target labels
    for the full 90-day historical dataset.  Every row gets target_aqi_24/48/72h
    because we have future data available.  Rows where lag/target values are
    still missing (first 48 h and last 72 h of the window) are dropped.

    Entry point: run_feature_pipeline()
    Caller:      backfill.py

Incremental (hourly)
    Called every hour by GitHub Actions after fetch_openweather.py stores the
    latest raw record.  Reads only the last ~73 raw records (enough to compute
    48-hour lags + 24-hour rolling stats) and engineers ONE feature row for
    the newest timestamp.  Target columns are NOT added because future AQI
    values are unknown.  The resulting row is used for live prediction.

    Entry point: run_incremental_pipeline()
    Caller:      hourly_pipeline.py  (via __main__ below)

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
from src.config import RAW_COLLECTION
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Number of raw rows needed as context for accurate lag/rolling computation.
# 48 h lag  + 24 h rolling window + 1 new row = 73 minimum.
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


# ── Core feature builders (shared by both modes) ──────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar / time-of-day features. Mutates df in-place."""
    df["hour"]       = df["datetime"].dt.hour
    df["day"]        = df["datetime"].dt.day
    df["month"]      = df["datetime"].dt.month
    df["weekday"]    = df["datetime"].dt.weekday   # 0 = Monday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features.  Requires df sorted ascending by datetime.
    Mutates df in-place.
    """
    df["aqi_lag_1"]           = df["aqi"].shift(1)
    df["aqi_lag_24"]          = df["aqi"].shift(24)
    df["aqi_lag_48"]          = df["aqi"].shift(48)
    df["pm25_lag_24"]         = df["pm2_5"].shift(24)
    df["pm10_lag_24"]         = df["pm10"].shift(24)

    df["aqi_rolling_24_mean"] = (
        df["aqi"].shift(1).rolling(window=24, min_periods=12).mean()
    )
    df["pm25_rolling_24_mean"] = (
        df["pm2_5"].shift(1).rolling(window=24, min_periods=12).mean()
    )

    df["aqi_change_rate"] = (
        (df["aqi"] - df["aqi_lag_24"])
        / df["aqi_lag_24"].replace(0, float("nan"))
    )
    return df


def _add_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add future-AQI target columns by shifting backward.

    Only valid when the full dataset is available (batch/backfill mode).
    The last 72 rows will have NaN targets and are dropped afterwards.
    """
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)
    return df


# ── Batch mode (backfill) ──────────────────────────────────────────────────────

def build_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete feature DataFrame from a full raw history.

    Includes all lag, rolling, and target columns.
    Drops rows where any lag or target value is missing (the first 48 and
    last 72 hours of the window cannot be fully populated).

    Parameters
    ----------
    df : pd.DataFrame
        All raw records sorted ascending by datetime.

    Returns
    -------
    pd.DataFrame
        Training-ready feature rows with target columns.
    """
    feat = df.copy()
    feat = _add_time_features(feat)
    feat = _add_lag_rolling_features(feat)
    feat = _add_target_features(feat)

    must_be_present = [
        "aqi_lag_1", "aqi_lag_24", "aqi_lag_48",
        "pm25_lag_24", "pm10_lag_24",
        "aqi_rolling_24_mean", "pm25_rolling_24_mean",
        "aqi_change_rate",
        "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
    ]
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
        records.append(doc)
    count = insert_features_batch(records)
    logger.info("Upserted %d feature records into MongoDB.", count)


def run_feature_pipeline() -> pd.DataFrame:
    """
    BATCH entry point — called by backfill.py.

    Loads ALL raw records → engineers full features + targets → stores all.
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
    No target columns are added — future AQI is not known yet.

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
    feat = _add_lag_rolling_features(feat)
    # ── No target features: future data is unavailable ─────────────────────────

    latest = feat.iloc[-1].to_dict()
    if hasattr(latest["datetime"], "to_pydatetime"):
        latest["datetime"] = latest["datetime"].to_pydatetime()
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
        "Incremental: upserted 1 feature row — datetime=%s  AQI=%s",
        feature_row.get("datetime"), feature_row.get("aqi"),
    )
    return feature_row


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # When called by hourly_pipeline.py (or directly), run incremental only.
    ensure_indexes()
    result = run_incremental_pipeline()
    logger.info("Incremental feature pipeline complete for %s.", result.get("datetime"))
