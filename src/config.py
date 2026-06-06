"""
config.py — Central configuration loader.

Reads all settings from environment variables.
  - Set them in .env for local development.
  - Inject them as GitHub Actions repository secrets in CI/CD.
"""

import os
from dotenv import load_dotenv

# Load .env when running locally; no-op in CI/CD where vars are injected
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

DB_NAME: str = os.getenv("DB_NAME", "aqi_predictor")

# ── Hugging Face Hub ───────────────────────────────────────────────────────────
HF_TOKEN:   str | None = os.getenv("HF_TOKEN")
HF_REPO_ID: str | None = os.getenv("HF_REPO_ID")

# Gate flag:  all HF Hub calls are skipped when either variable is absent.
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
# raw_data        : one hourly record from OpenWeather API (pollution + weather)
# features        : ML-ready rows; backfill rows have target columns,
#                   hourly pipeline rows do NOT (future data unavailable)
# model_registry  : metadata mirror of every HF Hub model upload
#                   (hf_repo_id, hf_model_path, metrics, feature_columns …)
# predictions     : saved forecast outputs from predict.py (optional)
RAW_COLLECTION: str = "raw_data"
FEATURES_COLLECTION: str = "features"
MODELS_COLLECTION: str = "model_registry"
PREDICTIONS_COLLECTION: str = "predictions"

# ── Feature columns used during training ──────────────────────────────────────
FEATURE_COLUMNS: list[str] = [
    "hour", "day", "month", "weekday", "is_weekend",
    "aqi_lag_1", "aqi_lag_24", "aqi_lag_48",
    "pm25_lag_24", "pm10_lag_24",
    "aqi_rolling_24_mean", "pm25_rolling_24_mean",
    "aqi_change_rate",
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
]

TARGET_COLUMNS: list[str] = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]

# ── AQI category mapping (OpenWeather 1-5 scale) ──────────────────────────────
AQI_CATEGORIES: dict[int, str] = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}
