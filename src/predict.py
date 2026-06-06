"""
predict.py — Generate 24h, 48h, 72h AQI predictions for Karachi.

Data source
-----------
Loads the latest feature rows from MongoDB (features collection).
The most recent row — written by the hourly pipeline — does NOT need target
columns; only the feature values (lags, rolling stats, weather, etc.) are
needed to feed into the trained model.

Model source
------------
Loads the best saved model for each forecast horizon from the model registry
(model_registry collection in MongoDB + the .pkl file on disk).

Run standalone:
    python src/predict.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.database import get_latest_features, save_prediction, ensure_indexes
from src.model_registry import load_model
from src.utils import setup_logging, aqi_label, utc_now

setup_logging()
logger = logging.getLogger(__name__)


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_input(features: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    """
    Convert a list of feature dicts to a 2-D numpy array aligned to
    the model's expected feature order.

    Parameters
    ----------
    features : list[dict]
        Latest feature rows from MongoDB (ordered oldest → newest).
    feature_columns : list[str]
        Ordered feature list stored in model metadata.

    Returns
    -------
    np.ndarray of shape (1, n_features) using the most recent row.
    """
    df = pd.DataFrame(features)
    latest = df.iloc[-1]
    available = [c for c in feature_columns if c in latest.index]
    missing = set(feature_columns) - set(available)
    if missing:
        logger.warning("Missing features in latest data: %s. Filling with 0.", missing)

    row = {}
    for col in feature_columns:
        row[col] = latest.get(col, 0.0)

    return np.array([[row[c] for c in feature_columns]])


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_all_horizons(save_to_db: bool = False) -> dict[str, Any]:
    """
    Load each target's best model and predict AQI for 24h, 48h, 72h.

    Uses the most recent feature row from MongoDB (written by the hourly
    pipeline).  Target columns are not required — only feature values matter.

    Parameters
    ----------
    save_to_db : bool
        If True, persist each forecast to the predictions collection.

    Returns
    -------
    dict with structure:
        {
            "target_aqi_24h": {"aqi": <float>, "label": <str>, ...},
            "target_aqi_48h": {...},
            "target_aqi_72h": {...},
        }
    """
    # Get the latest feature rows — these are from the hourly pipeline
    # and contain all feature values needed for inference.
    features = get_latest_features(n=72)
    if not features:
        raise ValueError(
            "No feature data in MongoDB. "
            "Run backfill.py first, then wait for the hourly pipeline to run."
        )

    predicted_at = utc_now()
    predictions: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        try:
            model, meta = load_model(target=target)
            feature_columns = meta.get("feature_columns", FEATURE_COLUMNS)
            X = prepare_input(features, feature_columns)
            raw_pred = model.predict(X)[0]
            # Clip predictions to the valid OpenWeather AQI range [1, 5]
            aqi_pred = float(np.clip(round(raw_pred), 1, 5))
            label = aqi_label(aqi_pred)

            predictions[target] = {
                "aqi": aqi_pred,
                "label": label,
                "model_name": meta.get("model_name"),
                "trained_at": meta.get("trained_at"),
                "predicted_at": predicted_at,
            }
            logger.info("Prediction [%s]: AQI=%.2f (%s)", target, aqi_pred, label)

            if save_to_db:
                save_prediction({
                    "target": target,
                    "aqi": aqi_pred,
                    "label": label,
                    "model_name": meta.get("model_name"),
                    "predicted_at": predicted_at,
                })

        except FileNotFoundError as exc:
            logger.error("Could not load model for %s: %s", target, exc)
            predictions[target] = {"aqi": None, "label": "N/A", "error": str(exc)}
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", target, exc)
            predictions[target] = {"aqi": None, "label": "N/A", "error": str(exc)}

    return predictions


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_indexes()
    preds = predict_all_horizons()
    print("\n=== AQI Predictions for Karachi ===")
    for horizon, info in preds.items():
        label = info.get("label", "N/A")
        aqi = info.get("aqi")
        hours = horizon.replace("target_aqi_", "").replace("h", "")
        print(f"  +{hours}h: AQI = {aqi}  ({label})")
