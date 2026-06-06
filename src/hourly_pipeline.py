"""
hourly_pipeline.py — Incremental hourly data + feature pipeline.

Called by GitHub Actions every hour.  Performs exactly two steps:

  1. Fetch the latest Karachi AQI + weather record from OpenWeather and
     insert/upsert it into MongoDB → raw_data collection.

  2. Read the last ~73 raw records (enough lag context) from MongoDB,
     compute features for the NEWEST record only, and upsert ONE row into
     MongoDB → features collection.

This script NEVER rebuilds features for the whole 90-day history.
Target columns (target_aqi_24h / 48h / 72h) are NOT created here because
future AQI values are unknown.  Only the daily train.py uses rows with
target labels (which came from the one-time backfill).

Run standalone:
    python src/hourly_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import ensure_indexes
from src.fetch_openweather import fetch_and_store_current
from src.feature_engineering import run_incremental_pipeline
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_hourly_pipeline() -> None:
    """
    Full incremental hourly pipeline:

    Step 1 — API fetch
        Calls OpenWeather air pollution + weather endpoints for Karachi.
        Upserts one record into raw_data keyed by (datetime, city).
        Idempotent: re-running at the same hour will overwrite, not duplicate.

    Step 2 — Incremental feature engineering
        Reads the last 73 raw records from raw_data (for lag context).
        Computes all feature columns for the newest timestamp only.
        Upserts that single row into the features collection.
        Idempotent: re-running produces the same row and overwrites it.
    """
    ensure_indexes()

    # ── Step 1: Fetch latest raw record ───────────────────────────────────────
    logger.info("=== STEP 1: Fetching latest AQI + weather from OpenWeather ===")
    raw_record = fetch_and_store_current()
    logger.info(
        "Raw record stored — datetime=%s  AQI=%s",
        raw_record.get("datetime"), raw_record.get("aqi"),
    )

    # ── Step 2: Incremental feature engineering ────────────────────────────────
    logger.info("=== STEP 2: Incremental feature engineering ===")
    feature_row = run_incremental_pipeline()
    logger.info(
        "Feature row upserted — datetime=%s  AQI=%s  aqi_lag_1=%s",
        feature_row.get("datetime"),
        feature_row.get("aqi"),
        feature_row.get("aqi_lag_1"),
    )

    logger.info("=== HOURLY PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run_hourly_pipeline()
