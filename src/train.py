"""
train.py — Daily training pipeline for 4 ML models × 3 forecast horizons.

Data source
-----------
Loads feature rows from MongoDB (features collection) that have ALL three
target columns populated.  These rows were created by the one-time backfill
(backfill.py) and grow daily as the hourly pipeline adds new records whose
targets become known 24–72 h later.

Rows inserted by the incremental hourly pipeline (no target columns) are
automatically excluded by get_training_data().

This script does NOT call the API, fetch raw data, or run feature engineering.
It purely trains on what is already in MongoDB.

Per target (24h / 48h / 72h):
  1. Time-based train/test split (no random shuffle)
  2. Train: LinearRegression, RandomForestRegressor,
            GradientBoostingRegressor, XGBRegressor
  3. Evaluate: MAE, RMSE, R²
  4. For best model (lowest RMSE) via model_registry.save_model():
       a. Save .pkl locally (models/ folder)
       b. Upload .pkl to GCS (gs://<GCS_MODEL_BUCKET>/models/<target>/<ts>/)
       c. Register artifact in Vertex AI Model Registry
       d. Mirror metadata in MongoDB model_registry collection

Run standalone:
    python src/train.py

Required environment variables (in addition to MONGODB_URI etc.):
    GCP_PROJECT_ID, GCP_REGION, GCS_MODEL_BUCKET
    (See .env.example or config.py for the full list)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.database import get_training_data, ensure_indexes
from src.model_registry import save_model
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TEST_RATIO = 0.2   # last 20% of data used for testing (time-ordered)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Load training-eligible feature rows from MongoDB.

    get_training_data() filters for rows that have ALL three target columns
    (target_aqi_24h, target_aqi_48h, target_aqi_72h) populated.  Rows
    added by the incremental hourly pipeline (which have no targets) are
    excluded automatically.
    """
    docs = get_training_data()
    if not docs:
        raise ValueError(
            "No training data found in the features collection.\n"
            "Make sure backfill.py has been run — it stores rows WITH target labels.\n"
            "Hourly pipeline rows do not have targets and cannot be used for training."
        )
    df = pd.DataFrame(docs)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d training-eligible feature rows (with target columns).", len(df))
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R² and return as a dict."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


# ── Model catalogue ────────────────────────────────────────────────────────────

def get_models() -> dict[str, Any]:
    """
    Return a dict of model_name → sklearn-compatible estimator.

    LinearRegression is wrapped in a Pipeline with StandardScaler
    since it is sensitive to feature scale.
    """
    return {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        ),
    }


# ── Per-target training ────────────────────────────────────────────────────────

def train_for_target(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """
    Train all 4 models for a single target column.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame with all target columns.
    target : str
        One of 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'.

    Returns
    -------
    dict
        Summary of results for each model.
    """
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        logger.warning("Features missing from DataFrame (will be skipped): %s", missing)

    sub = df[available_features + [target]].dropna()
    X = sub[available_features].values
    y = sub[target].values

    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(
        "[%s] Train=%d  Test=%d", target, len(X_train), len(X_test)
    )

    models = get_models()
    results: dict[str, dict] = {}
    best_name: str | None = None
    best_rmse = float("inf")
    best_model: Any = None

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            results[name] = metrics
            logger.info(
                "  [%s] %s → MAE=%.4f  RMSE=%.4f  R²=%.4f",
                target, name, metrics["mae"], metrics["rmse"], metrics["r2"],
            )
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_name = name
                best_model = model
        except Exception as exc:
            logger.error("  [%s] %s failed: %s", target, name, exc)
            results[name] = {"error": str(exc)}

    if best_model is not None and best_name is not None:
        logger.info(
            "[%s] Best model: %s (RMSE=%.4f)", target, best_name, best_rmse
        )
        save_model(
            model=best_model,
            model_name=best_name,
            target=target,
            metrics=results[best_name],
            feature_columns=available_features,
        )

    return {
        "target": target,
        "best_model": best_name,
        "best_rmse": best_rmse,
        "all_results": results,
    }


# ── Main training pipeline ─────────────────────────────────────────────────────

def run_training() -> list[dict[str, Any]]:
    """Run the full training pipeline across all three forecast horizons."""
    df = load_features()
    summary = []
    for target in TARGET_COLUMNS:
        result = train_for_target(df, target)
        summary.append(result)
    return summary


if __name__ == "__main__":
    ensure_indexes()
    logger.info("=== TRAINING PIPELINE STARTED ===")
    summary = run_training()
    logger.info("=== TRAINING COMPLETE ===")
    for s in summary:
        logger.info(
            "Target: %-20s | Best: %-28s | RMSE: %.4f",
            s["target"], s["best_model"], s["best_rmse"],
        )
