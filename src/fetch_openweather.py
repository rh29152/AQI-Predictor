"""
fetch_openweather.py — OpenWeather air pollution and weather ingestion for Karachi.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# Supports direct execution from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    OPENWEATHER_API_KEY,
    KARACHI_LAT,
    KARACHI_LON,
    CITY_NAME,
)
from src.database import insert_raw_data, ensure_indexes
from src.utils import setup_logging, ts_to_utc

setup_logging()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org"
TIMEOUT = 15  # seconds


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict) -> dict:
    """Perform a GET request and return parsed JSON, or raise on failure."""
    params["appid"] = OPENWEATHER_API_KEY
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.error("Request timed out: %s", url)
        raise
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP error %s for %s: %s", exc.response.status_code, url, exc)
        raise
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        raise


# ── Current air pollution ──────────────────────────────────────────────────────

def fetch_air_pollution_current() -> dict:
    """
    Fetch current air pollution data for Karachi.

    Returns a flat dict ready for MongoDB insertion.
    """
    data = _get(
        "/data/2.5/air_pollution",
        {"lat": KARACHI_LAT, "lon": KARACHI_LON},
    )
    item = data["list"][0]
    components = item["components"]
    dt_utc = ts_to_utc(item["dt"])

    return {
        "datetime": dt_utc,
        "city": CITY_NAME,
        "aqi": item["main"]["aqi"],
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
    }


# ── Current weather ────────────────────────────────────────────────────────────

def fetch_weather_current() -> dict:
    """
    Fetch current weather for Karachi.

    Returns a flat dict with meteorological fields.
    """
    data = _get(
        "/data/2.5/weather",
        {
            "lat": KARACHI_LAT,
            "lon": KARACHI_LON,
            "units": "metric",
        },
    )
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "clouds": data["clouds"]["all"],
    }


# ── Historical air pollution (last N hours via timestamp range) ────────────────

def fetch_air_pollution_historical(start_ts: int, end_ts: int) -> list[dict]:
    """
    Fetch historical air pollution data between two UNIX timestamps.

    OpenWeather /air_pollution/history endpoint supports arbitrary ranges.

    Parameters
    ----------
    start_ts : int
        Start UNIX timestamp (UTC).
    end_ts : int
        End UNIX timestamp (UTC).

    Returns
    -------
    list[dict]
        List of flat records ready for insertion.
    """
    data = _get(
        "/data/2.5/air_pollution/history",
        {
            "lat": KARACHI_LAT,
            "lon": KARACHI_LON,
            "start": start_ts,
            "end": end_ts,
        },
    )
    records = []
    for item in data.get("list", []):
        components = item["components"]
        records.append(
            {
                "datetime": ts_to_utc(item["dt"]),
                "city": CITY_NAME,
                "aqi": item["main"]["aqi"],
                "co": components.get("co"),
                "no": components.get("no"),
                "no2": components.get("no2"),
                "o3": components.get("o3"),
                "so2": components.get("so2"),
                "pm2_5": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "nh3": components.get("nh3"),
            }
        )
    return records


# ── Historical weather (One Call 3.0 / Timemachine) ───────────────────────────

def fetch_weather_historical(timestamp_ts: int) -> dict | None:
    """
    Fetch historical weather for a specific UNIX timestamp using
    One Call API 3.0 timemachine endpoint.

    NOTE: One Call 3.0 requires a paid OpenWeather subscription.
    If unavailable, weather fields will be None for historical records.

    TODO: Replace with an alternative free historical weather source
          (e.g., Open-Meteo historical API) if One Call 3.0 is not available.
    """
    try:
        data = _get(
            "/data/3.0/onecall/timemachine",
            {
                "lat": KARACHI_LAT,
                "lon": KARACHI_LON,
                "dt": timestamp_ts,
                "units": "metric",
            },
        )
        hourly = data.get("data", [{}])[0]
        return {
            "temperature": hourly.get("temp"),
            "humidity": hourly.get("humidity"),
            "pressure": hourly.get("pressure"),
            "wind_speed": hourly.get("wind_speed"),
            "clouds": hourly.get("clouds"),
        }
    except Exception as exc:
        logger.warning(
            "Could not fetch historical weather for ts=%d: %s. "
            "Weather fields will be None.",
            timestamp_ts,
            exc,
        )
        return None


# ── Combined current fetch + store ─────────────────────────────────────────────

def fetch_and_store_current() -> dict:
    """
    Fetch current AQI + weather, merge into one record, and store in MongoDB.

    Returns the merged record dict.
    """
    logger.info("Fetching current air pollution data for %s…", CITY_NAME)
    pollution = fetch_air_pollution_current()

    logger.info("Fetching current weather data for %s…", CITY_NAME)
    weather = fetch_weather_current()

    record = {**pollution, **weather}
    insert_raw_data(record)
    logger.info(
        "Stored record for %s — AQI=%d", record["datetime"].isoformat(), record["aqi"]
    )
    return record


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_indexes()
    record = fetch_and_store_current()
    logger.info("Done. Record datetime: %s", record["datetime"])