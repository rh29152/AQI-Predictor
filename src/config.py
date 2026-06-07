"""
config.py — Central configuration loaded from environment variables.

Local development reads `.env`; CI injects the same keys via repository secrets.
Missing required keys raise at import time so misconfiguration surfaces early.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenWeather ────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
if not OPENWEATHER_API_KEY:
    raise EnvironmentError(
        "OPENWEATHER_API_KEY is not set. "
        "Add it to .env file or GitHub Actions secrets."
    )

# ── MongoDB Atlas ──────────────────────────────────────────────────────────────
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
if not MONGODB_URI:
    raise EnvironmentError(
        "MONGODB_URI is not set. "
        "Add it to your .env file or GitHub Actions secrets."
    )

DB_NAME: str = os.getenv("DB_NAME") or "aqi_predictor"

# ── Hugging Face Hub ───────────────────────────────────────────────────────────
HF_TOKEN:   str | None = os.getenv("HF_TOKEN")
HF_REPO_ID: str | None = os.getenv("HF_REPO_ID")

HF_ENABLED: bool = bool(HF_TOKEN and HF_REPO_ID)

if not HF_ENABLED:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "HF_TOKEN or HF_REPO_ID not set. "
        "Models will be saved locally only (no HF Hub upload)."
    )

# ── Karachi coordinates ────────────────────────────────────────────────────────
KARACHI_LAT: float = 24.8607
KARACHI_LON: float = 67.0011
CITY_NAME: str = "Karachi"

# ── MongoDB collection names ───────────────────────────────────────────────────
RAW_COLLECTION:         str = "raw_data"
FEATURES_COLLECTION:    str = "features"
MODELS_COLLECTION:      str = "model_registry"
PREDICTIONS_COLLECTION: str = "predictions"

# ── Pollutant-first AQI forecasting ───────────────────────────────────────────
# We forecast these 4 pollutant concentrations (μg/m³) at 3 horizons.
# Final AQI is derived via EPA-style breakpoint interpolation in aqi_utils.py.
POLLUTANTS_TO_FORECAST: list[str] = ["pm2_5", "pm10", "o3", "no2"]
FORECAST_HORIZONS: list[int] = [24, 48, 72]

# Twelve supervised targets: four pollutants × three forecast horizons
TARGET_COLUMNS: list[str] = [
    f"target_{pollutant}_{horizon}h"
    for pollutant in POLLUTANTS_TO_FORECAST
    for horizon in FORECAST_HORIZONS
]

# Model input schema — ordered feature list shared by training and inference
FEATURE_COLUMNS: list[str] = [
    # Raw pollutant concentrations (μg/m³)
    "pm2_5", "pm10", "o3", "no2", "co", "so2", "nh3",
    # Meteorology
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
    # Calendar / time-of-day
    "hour", "day", "month", "weekday", "is_weekend",
    # OpenWeather AQI category (1–5); input signal only, not the forecast target
    "aqi_category",
    # PM2.5 lag & rolling features
    "pm2_5_lag_1", "pm2_5_lag_24", "pm2_5_lag_48",
    "pm2_5_rolling_6_mean", "pm2_5_rolling_12_mean", "pm2_5_rolling_24_mean",
    "pm2_5_change_rate",
    # PM10 lag & rolling features
    "pm10_lag_1", "pm10_lag_24", "pm10_lag_48",
    "pm10_rolling_6_mean", "pm10_rolling_12_mean", "pm10_rolling_24_mean",
    "pm10_change_rate",
    # O3 lag & rolling features
    "o3_lag_1", "o3_lag_24", "o3_lag_48",
    "o3_rolling_6_mean", "o3_rolling_12_mean", "o3_rolling_24_mean",
    "o3_change_rate",
    # NO2 lag & rolling features
    "no2_lag_1", "no2_lag_24", "no2_lag_48",
    "no2_rolling_6_mean", "no2_rolling_12_mean", "no2_rolling_24_mean",
    "no2_change_rate",
]

# OpenWeather 1–5 scale labels (raw_data display); EPA 0–500 AQI lives in aqi_utils
AQI_CATEGORIES: dict[int, str] = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}
