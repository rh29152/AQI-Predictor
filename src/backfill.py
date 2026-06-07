"""
backfill.py — Historical backfill and feature rebuild for the AQI Predictor.

Normal run (first time setup)
------------------------------
    python src/backfill.py

Steps:
  1. Fetch 90 days of air pollution history from OpenWeather API.
  2. Fetch matching weather from Open-Meteo (free, no key required).
  3. Merge + store all records in MongoDB → raw_data collection.
  4. Run BATCH feature engineering:
     - lag, rolling, time features for pm2_5, pm10, o3, no2
     - 12 pollutant target columns (4 pollutants × 3 horizons)
     - Store all usable rows in MongoDB → features collection.

Rebuild features only (after target column migration)
------------------------------------------------------
    python src/backfill.py --rebuild-features

Skips the raw data fetch.  Clears the features collection and rebuilds it
from the existing raw_data records.  Useful after changing FEATURE_COLUMNS
or TARGET_COLUMNS without re-fetching data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import KARACHI_LAT, KARACHI_LON, CITY_NAME
from src.database import insert_raw_batch, ensure_indexes
from src.fetch_openweather import fetch_air_pollution_historical
from src.feature_engineering import run_feature_pipeline   # BATCH version — creates targets
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

BACKFILL_DAYS = 90          # 3 months
CHUNK_DAYS = 7              # OpenWeather history max chunk recommended size
RATE_LIMIT_SLEEP = 1.0      # seconds between API calls to avoid rate limits


# ── Open-Meteo weather fallback ────────────────────────────────────────────────

def fetch_open_meteo_weather(date_str: str) -> list[dict]:
    """
    Fetch hourly weather for a single day from Open-Meteo (free, no key needed).

    Parameters
    ----------
    date_str : str
        Date in 'YYYY-MM-DD' format.

    Returns
    -------
    list[dict]
        One dict per hour with weather fields.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m,cloudcover",
        "timezone": "UTC",
        "windspeed_unit": "ms",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        results = []
        for i, ts_str in enumerate(times):
            dt_utc = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            results.append(
                {
                    "datetime": dt_utc,
                    "temperature": hourly.get("temperature_2m", [None])[i],
                    "humidity": hourly.get("relativehumidity_2m", [None])[i],
                    "pressure": hourly.get("surface_pressure", [None])[i],
                    "wind_speed": hourly.get("windspeed_10m", [None])[i],
                    "clouds": hourly.get("cloudcover", [None])[i],
                }
            )
        return results
    except Exception as exc:
        logger.warning("Open-Meteo fetch failed for %s: %s", date_str, exc)
        return []


def build_weather_lookup(start_dt: datetime, end_dt: datetime) -> dict[datetime, dict]:
    """
    Build a datetime → weather-dict lookup for the backfill window
    using Open-Meteo (free fallback).
    """
    lookup: dict[datetime, dict] = {}
    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        logger.info("Fetching Open-Meteo weather for %s…", date_str)
        hourly = fetch_open_meteo_weather(date_str)
        for row in hourly:
            lookup[row["datetime"]] = row
        current += timedelta(days=1)
        time.sleep(0.3)
    return lookup


# ── Main backfill logic ────────────────────────────────────────────────────────

def run_backfill(days: int = BACKFILL_DAYS) -> int:
    """
    Fetch historical air pollution + weather data and store raw records.

    Parameters
    ----------
    days : int
        Number of past days to backfill.

    Returns
    -------
    int
        Total records inserted.
    """
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    logger.info(
        "Starting backfill: %s → %s (%d days)",
        start_dt.date(),
        end_dt.date(),
        days,
    )

    # ── Build weather lookup using Open-Meteo ──────────────────────────────────
    logger.info("Fetching weather data from Open-Meteo…")
    weather_lookup = build_weather_lookup(start_dt, end_dt)
    logger.info("Weather lookup built: %d hourly entries.", len(weather_lookup))

    # ── Fetch air pollution in weekly chunks ───────────────────────────────────
    all_records: list[dict] = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt)
        logger.info(
            "Fetching air pollution: %s → %s",
            chunk_start.date(),
            chunk_end.date(),
        )
        try:
            records = fetch_air_pollution_historical(
                int(chunk_start.timestamp()),
                int(chunk_end.timestamp()),
            )
            logger.info("  Got %d records.", len(records))
            all_records.extend(records)
        except Exception as exc:
            logger.error("Failed to fetch chunk %s→%s: %s", chunk_start.date(), chunk_end.date(), exc)
        chunk_start = chunk_end
        time.sleep(RATE_LIMIT_SLEEP)

    # ── Merge weather into pollution records ───────────────────────────────────
    enriched: list[dict] = []
    for rec in all_records:
        dt_key = rec["datetime"].replace(minute=0, second=0, microsecond=0)
        weather = weather_lookup.get(dt_key, {})
        merged = {
            **rec,
            "temperature": weather.get("temperature"),
            "humidity": weather.get("humidity"),
            "pressure": weather.get("pressure"),
            "wind_speed": weather.get("wind_speed"),
            "clouds": weather.get("clouds"),
            "city": CITY_NAME,
        }
        enriched.append(merged)

    logger.info("Storing %d merged records in MongoDB…", len(enriched))
    count = insert_raw_batch(enriched)
    logger.info("Stored %d raw records.", count)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI Predictor backfill / feature rebuild")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help=(
            "Skip raw data fetch and only rebuild the features collection "
            "from existing raw_data records. Use after changing feature/target "
            "definitions without re-fetching data."
        ),
    )
    args = parser.parse_args()

    ensure_indexes()

    if args.rebuild_features:
        # ── Rebuild-only mode: skip raw data fetch ─────────────────────────────
        from src.database import get_collection  # noqa: PLC0415
        from src.config import FEATURES_COLLECTION  # noqa: PLC0415

        logger.info("=== REBUILD-FEATURES MODE ===")
        logger.info("Clearing features collection...")
        n_deleted = get_collection(FEATURES_COLLECTION).delete_many({}).deleted_count
        logger.info("Deleted %d old feature rows.", n_deleted)

        logger.info("=== STEP: BATCH FEATURE ENGINEERING (with pollutant targets) ===")
        feat_df = run_feature_pipeline()
        logger.info(
            "Feature rebuild complete. %d training rows stored in features.",
            len(feat_df),
        )
        logger.info("=== REBUILD COMPLETE — ready for train.py ===")

    else:
        # ── Full backfill mode ─────────────────────────────────────────────────
        logger.info("=== STEP 1: RAW DATA BACKFILL ===")
        total = run_backfill(days=BACKFILL_DAYS)
        logger.info("Raw backfill complete: %d records stored in raw_data.", total)

        logger.info("=== STEP 2: BATCH FEATURE ENGINEERING (with pollutant targets) ===")
        feat_df = run_feature_pipeline()
        logger.info(
            "Feature engineering complete. %d training rows stored in features.",
            len(feat_df),
        )
        logger.info("=== BACKFILL COMPLETE — ready for train.py ===")
