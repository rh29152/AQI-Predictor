"""
utils.py — Shared utility functions.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a standard format."""
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def ts_to_utc(timestamp: int) -> datetime:
    """Convert a UNIX timestamp (seconds) to UTC datetime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def aqi_label(aqi: int | float) -> str:
    """
    Return a human-readable label for an EPA AQI value (0-500 scale).
    Delegates to aqi_utils.aqi_label for the 0-500 EPA scale.
    """
    from src.aqi_utils import aqi_label as _epa_label  # noqa: PLC0415
    return _epa_label(aqi)


def aqi_color(aqi: int | float) -> str:
    """Return a hex color for an EPA AQI value (0-500 scale)."""
    from src.aqi_utils import aqi_color as _epa_color  # noqa: PLC0415
    return _epa_color(aqi)


def openweather_aqi_label(aqi: int | float) -> str:
    """
    Map OpenWeather AQI integer (1-5) to a human-readable label.

    Kept for displaying raw_data 'aqi_category' field in the EDA notebook
    and historical trend charts.  Do NOT use for forecast output.
    """
    mapping = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return mapping.get(int(round(aqi)), "Unknown")


def openweather_aqi_color(aqi: int | float) -> str:
    """Return a hex color for OpenWeather AQI (1-5 scale)."""
    colors = {1: "#00e400", 2: "#ffff00", 3: "#ff7e00", 4: "#ff0000", 5: "#8f3f97"}
    return colors.get(int(round(aqi)), "#999999")


def sanitize_doc(doc: dict) -> dict:
    """
    Make a MongoDB document JSON-serializable by converting datetime objects
    to ISO-format strings and removing the '_id' field.
    """
    clean = {}
    for k, v in doc.items():
        if k == "_id":
            continue
        if isinstance(v, datetime):
            clean[k] = v.isoformat()
        else:
            clean[k] = v
    return clean
