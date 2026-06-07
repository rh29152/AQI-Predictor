"""
predict.py — Pollutant-first AQI forecasting for Karachi.

Approach
--------
1. Load the latest feature row from MongoDB (written by the hourly pipeline).
2. For each of the 12 targets (4 pollutants × 3 horizons):
   - Load the latest best model from model_registry / HF Hub cache.
   - Predict the pollutant concentration (μg/m³).
   - Clip negative predictions to 0 (concentrations cannot be negative).
3. Group predictions by horizon (24h, 48h, 72h).
4. For each horizon, pass the pollutant concentration dict to
   aqi_utils.calculate_final_aqi() to compute the EPA-style AQI.
5. Return and optionally persist full forecast documents.

Run standalone:
    python src/predict.py
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_COLUMNS, POLLUTANTS_TO_FORECAST, FORECAST_HORIZONS, CITY_NAME
from src.database import get_latest_features, save_prediction, ensure_indexes
from src.model_registry import load_model
from src.aqi_utils import calculate_final_aqi
from src.utils import setup_logging, utc_now

setup_logging()
logger = logging.getLogger(__name__)


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_input(features: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    """
    Convert a list of feature dicts to a 2-D numpy array aligned to the
    model's expected feature order, using the most recent row.

    Parameters
    ----------
    features : list[dict]
        Latest feature rows from MongoDB (ordered oldest → newest).
    feature_columns : list[str]
        Ordered feature list stored in model metadata.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    df = pd.DataFrame(features)
    latest = df.iloc[-1]
    missing = set(feature_columns) - set(latest.index)
    if missing:
        logger.warning("Missing features in latest data: %s. Filling with 0.", missing)
    row = {col: float(latest.get(col, 0.0) or 0.0) for col in feature_columns}
    return np.array([[row[c] for c in feature_columns]])


# ── Core prediction ────────────────────────────────────────────────────────────

def predict_all_horizons(save_to_db: bool = False) -> dict[str, Any]:
    """
    Predict pollutant concentrations for all horizons, then derive AQI.

    Parameters
    ----------
    save_to_db : bool
        If True, persist each horizon's forecast to the predictions collection.

    Returns
    -------
    dict keyed by horizon label ('24h', '48h', '72h'), each containing:
        {
            "horizon_hours": 24,
            "predicted_pollutants": {"pm2_5": float, "pm10": float, ...},
            "sub_indices": {"pm2_5": int, "pm10": int, ...},
            "predicted_aqi": int,
            "aqi_category": str,
            "dominant_pollutant": str,
            "color": str,
            "predicted_at": datetime,
            "target_time": datetime,
        }
    """
    features = get_latest_features(n=72)
    if not features:
        raise ValueError(
            "No feature data in MongoDB. "
            "Run backfill.py first, then wait for the hourly pipeline to run."
        )

    predicted_at = utc_now()

    # ── Latest timestamp for deriving target_time ──────────────────────────────
    latest_dt = pd.to_datetime(features[-1]["datetime"], utc=True)

    results: dict[str, Any] = {}

    for horizon_h in FORECAST_HORIZONS:
        target_time = latest_dt + timedelta(hours=horizon_h)
        pollutant_predictions: dict[str, float] = {}

        for pollutant in POLLUTANTS_TO_FORECAST:
            target_key = f"target_{pollutant}_{horizon_h}h"
            try:
                # load_model() prints full verification log (target, algorithm,
                # trained_at, hf_path, active flag) before returning the model.
                model, meta = load_model(target=target_key)
                feature_columns = meta.get("feature_columns", FEATURE_COLUMNS)
                X = prepare_input(features, feature_columns)
                raw_pred = float(model.predict(X)[0])
                concentration = max(0.0, raw_pred)   # clip negatives
                pollutant_predictions[pollutant] = round(concentration, 2)
                logger.info(
                    "  [+%dh] %s = %.2f ug/m3",
                    horizon_h, pollutant, concentration,
                )
            except FileNotFoundError as exc:
                logger.error("Model not found for %s: %s", target_key, exc)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", target_key, exc)

        aqi_result = calculate_final_aqi(pollutant_predictions)

        horizon_label = f"{horizon_h}h"
        results[horizon_label] = {
            "city": CITY_NAME,
            "horizon_hours": horizon_h,
            "predicted_at": predicted_at,
            "target_time": target_time,
            "predicted_pollutants": pollutant_predictions,
            "sub_indices": aqi_result["sub_indices"],
            "predicted_aqi": aqi_result["aqi"],
            "aqi_category": aqi_result["category"],
            "dominant_pollutant": aqi_result["dominant_pollutant"],
            "color": aqi_result["color"],
        }

        logger.info(
            "[+%dh] AQI=%s  Category=%s  Dominant=%s",
            horizon_h,
            aqi_result["aqi"],
            aqi_result["category"],
            aqi_result["dominant_pollutant"],
        )

        if save_to_db and aqi_result["aqi"] is not None:
            save_prediction({
                "city": CITY_NAME,
                "horizon_hours": horizon_h,
                "predicted_at": predicted_at,
                "target_time": target_time,
                "predicted_pollutants": pollutant_predictions,
                "sub_indices": aqi_result["sub_indices"],
                "predicted_aqi": aqi_result["aqi"],
                "aqi_category": aqi_result["category"],
                "dominant_pollutant": aqi_result["dominant_pollutant"],
            })

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_indexes()
    logger.info("=== PREDICTION PIPELINE STARTED ===")
    preds = predict_all_horizons(save_to_db=True)
    print("\n" + "=" * 65)
    print(f"  Karachi AQI Forecast")
    print("=" * 65)
    for horizon_label, info in preds.items():
        pollutants = info.get("predicted_pollutants", {})
        print(
            f"  +{horizon_label:<5}  AQI={info.get('predicted_aqi', 'N/A'):<5}  "
            f"Category: {info.get('aqi_category', 'N/A'):<35}  "
            f"Dominant: {info.get('dominant_pollutant', 'N/A')}"
        )
        for poll, conc in pollutants.items():
            sub = info.get("sub_indices", {}).get(poll, "?")
            print(f"            {poll:<8} = {conc:>7.2f} ug/m3  (sub-AQI: {sub})")
        print()
    print("=" * 65)
