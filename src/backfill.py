"""
backfill.py — One-time historical backfill for Karachi AQI + weather data.

Runs ONCE before anything else to pre-populate MongoDB with 90 days of
hourly records.  After this, the hourly pipeline handles incremental updates.

Steps performed
---------------
1. Fetches 90 days of air pollution history from OpenWeather (free tier).
2. Fetches matching weather data from Open-Meteo (free, no key required).
3. Merges and store ALL records in MongoDB → raw_data collection.
4. Runs the BATCH feature pipeline on the full history:
   - Compute lag, rolling, and time features.
   - Compute target labels (target_aqi_24/48/72h) using future-shifted AQI.
   - Store all usable rows in MongoDB → features collection.

After backfill the features collection contains rows WITH target columns,
which the daily training pipeline (train.py) uses for model fitting.
"""

from __future__ import annotations

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
    ensure_indexes()

    # ── Step 1: Fetch & store 90 days of raw data ──────────────────────────────
    logger.info("=== STEP 1: RAW DATA BACKFILL ===")
    total = run_backfill(days=BACKFILL_DAYS)
    logger.info("Raw backfill complete: %d records stored in raw_data.", total)

    # ── Step 2: Batch feature engineering (creates target labels) ─────────────
    # run_feature_pipeline() is the BATCH version — it loads all raw records,
    # builds lag/rolling/time features + target_aqi_24/48/72h columns, and
    # stores training-ready rows in the features collection.
    # This is NOT called by the hourly pipeline.
    logger.info("=== STEP 2: BATCH FEATURE ENGINEERING (with target labels) ===")
    feat_df = run_feature_pipeline()
    logger.info(
        "Feature engineering complete. %d training rows stored in features.",
        len(feat_df),
    )

    logger.info("=== BACKFILL COMPLETE — ready for train.py ===")
