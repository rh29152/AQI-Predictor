"""
hourly_pipeline.py — Incremental hourly ingestion and feature engineering.

Orchestrates the live data path: optional gap fill (up to MAX_CATCHUP_HOURS),
current OpenWeather fetch, and single-row incremental feature upsert. All writes
use MongoDB upserts keyed by datetime for idempotent re-runs.

CLI flags: --no-catchup (snapshot only), --catchup-hours N (gap limit).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backfill import fetch_open_meteo_weather
from src.config import CITY_NAME
from src.database import (
    ensure_indexes,
    get_latest_feature_datetime,
    insert_raw_batch,
)
from src.fetch_openweather import (
    fetch_air_pollution_historical,
    fetch_and_store_current,
)
from src.feature_engineering import (
    run_incremental_for_datetime,
    run_incremental_pipeline,
)
from src.utils import setup_logging, utc_now

setup_logging()
logger = logging.getLogger(__name__)

MAX_CATCHUP_HOURS = 48
RATE_LIMIT_SLEEP = 1.0


def floor_to_hour(dt: datetime) -> datetime:
    """Floor a timezone-aware datetime to the start of its UTC hour."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def compute_missing_hours(latest_feature: datetime, now: datetime) -> list[datetime]:
    """
    Hourly timestamps strictly after the latest feature hour through now (floored).

    Example: latest=2026-06-07 04:15, now=2026-06-07 09:22
      -> [05:00, 06:00, 07:00, 08:00, 09:00]
    """
    latest_h = floor_to_hour(latest_feature)
    now_h = floor_to_hour(now)
    start = latest_h + timedelta(hours=1)
    if start > now_h:
        return []

    missing: list[datetime] = []
    current = start
    while current <= now_h:
        missing.append(current)
        current += timedelta(hours=1)
    return missing


def _weather_lookup_for_range(start: datetime, end: datetime) -> dict[datetime, dict]:
    """Build datetime -> weather dict for a date range via Open-Meteo."""
    lookup: dict[datetime, dict] = {}
    day = floor_to_hour(start).replace(hour=0)
    end_day = floor_to_hour(end).replace(hour=0)
    while day <= end_day:
        date_str = day.strftime("%Y-%m-%d")
        logger.info("Catch-up: fetching Open-Meteo weather for %s", date_str)
        for row in fetch_open_meteo_weather(date_str):
            lookup[row["datetime"]] = row
        day += timedelta(days=1)
        time.sleep(0.3)
    return lookup


def fetch_and_store_historical_hours(missing_hours: list[datetime]) -> int:
    """
    Fetch historical pollution for missing hourly slots and upsert raw rows.

    Returns the number of raw records upserted.
    """
    if not missing_hours:
        return 0

    start = missing_hours[0]
    end = missing_hours[-1]
    end_ts = int((end + timedelta(hours=1)).timestamp()) - 1
    start_ts = int(start.timestamp())

    logger.info(
        "Catch-up: fetching historical pollution %s -> %s (%d hours)",
        start.isoformat(),
        end.isoformat(),
        len(missing_hours),
    )
    pollution_records = fetch_air_pollution_historical(start_ts, end_ts)
    logger.info("Catch-up: API returned %d pollution records.", len(pollution_records))

    weather_lookup = _weather_lookup_for_range(start, end)
    missing_set = set(missing_hours)

    enriched: list[dict] = []
    seen_hours: set[datetime] = set()
    for rec in pollution_records:
        hour_dt = floor_to_hour(rec["datetime"])
        if hour_dt not in missing_set:
            continue
        if hour_dt in seen_hours:
            continue
        seen_hours.add(hour_dt)

        weather = weather_lookup.get(hour_dt, {})
        enriched.append(
            {
                **rec,
                "datetime": hour_dt,
                "city": CITY_NAME,
                "temperature": weather.get("temperature"),
                "humidity": weather.get("humidity"),
                "pressure": weather.get("pressure"),
                "wind_speed": weather.get("wind_speed"),
                "clouds": weather.get("clouds"),
            }
        )

    unfetched = missing_set - seen_hours
    if unfetched:
        logger.warning(
            "Catch-up: API did not return data for %d hour(s): %s",
            len(unfetched),
            ", ".join(sorted(h.isoformat() for h in unfetched)[:5])
            + ("..." if len(unfetched) > 5 else ""),
        )

    if not enriched:
        logger.warning("Catch-up: no historical raw rows to store.")
        return 0

    count = insert_raw_batch(enriched)
    logger.info("Catch-up: upserted %d historical raw row(s).", count)
    return count


def run_catchup(max_hours: int = MAX_CATCHUP_HOURS) -> int:
    """
    Fill missing hourly raw + feature rows since the latest feature timestamp.

    Returns the number of feature rows successfully upserted during catch-up.
    """
    latest = get_latest_feature_datetime()
    if latest is None:
        logger.warning(
            "Catch-up skipped: no features in MongoDB. Run backfill.py first."
        )
        return 0

    now = utc_now()
    missing = compute_missing_hours(latest, now)
    gap = len(missing)

    if gap == 0:
        logger.info(
            "Catch-up: no hourly gap (latest feature=%s, now=%s).",
            latest.isoformat(),
            floor_to_hour(now).isoformat(),
        )
        return 0

    if gap > max_hours:
        logger.warning(
            "Gap too large for hourly catch-up (%d hours > %d). "
            "Run backfill/rebuild manually.",
            gap,
            max_hours,
        )
        return 0

    logger.info(
        "Catch-up: filling %d missing hour(s) from %s to %s.",
        gap,
        missing[0].isoformat(),
        missing[-1].isoformat(),
    )

    fetch_and_store_historical_hours(missing)
    time.sleep(RATE_LIMIT_SLEEP)

    features_built = 0
    for hour_dt in missing:
        result = run_incremental_for_datetime(hour_dt)
        if result is not None:
            features_built += 1

    logger.info(
        "Catch-up complete: %d/%d feature row(s) upserted.",
        features_built,
        gap,
    )
    return features_built


def run_hourly_pipeline(
    enable_catchup: bool = True,
    catchup_hours: int = MAX_CATCHUP_HOURS,
) -> None:
    """Hourly pipeline: optional gap fill, live raw ingest, incremental features."""
    ensure_indexes()

    if enable_catchup:
        logger.info("=== STEP 0: Catch-up missing hours (max=%d) ===", catchup_hours)
        run_catchup(max_hours=catchup_hours)
    else:
        logger.info("=== STEP 0: Catch-up disabled (--no-catchup) ===")

    logger.info("=== STEP 1: Fetching latest AQI + weather from OpenWeather ===")
    raw_record = fetch_and_store_current()
    logger.info(
        "Raw record stored — datetime=%s  AQI=%s",
        raw_record.get("datetime"),
        raw_record.get("aqi"),
    )

    logger.info("=== STEP 2: Incremental feature engineering (current snapshot) ===")
    feature_row = run_incremental_pipeline()
    logger.info(
        "Feature row upserted — datetime=%s  pm2_5=%s  pm2_5_lag_1=%s",
        feature_row.get("datetime"),
        feature_row.get("pm2_5"),
        feature_row.get("pm2_5_lag_1"),
    )

    logger.info("=== HOURLY PIPELINE COMPLETE ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental hourly AQI pipeline with optional catch-up."
    )
    parser.add_argument(
        "--no-catchup",
        action="store_true",
        help="Skip catch-up; fetch and engineer current snapshot only.",
    )
    parser.add_argument(
        "--catchup-hours",
        type=int,
        default=MAX_CATCHUP_HOURS,
        metavar="N",
        help=f"Max missing hours to fill automatically (default: {MAX_CATCHUP_HOURS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_hourly_pipeline(
        enable_catchup=not args.no_catchup,
        catchup_hours=args.catchup_hours,
    )
