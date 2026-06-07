"""
train.py — Daily training pipeline: 4 ML models × 12 pollutant targets.

Data source
-----------
Loads ALL feature rows from MongoDB (features collection) sorted by datetime.
Each call to train_for_target() drops only the rows where *that specific*
target column is missing, so each of the 12 targets gets the maximum possible
training set.  Hourly incremental rows (no target columns at all) are excluded
automatically by the per-target .dropna() step.

This script does NOT call the API, fetch raw data, or run feature engineering.
It purely trains on what is already in MongoDB.

12 targets: 4 pollutants (pm2_5, pm10, o3, no2) × 3 horizons (24h, 48h, 72h)
48 candidate trainings total (12 targets × 4 algorithms).

Per target:
  1. Time-based train/test split — last 20% held out (no random shuffle)
  2. Train: Ridge (+ StandardScaler), RandomForestRegressor,
            GradientBoostingRegressor, XGBRegressor
  3. Evaluate train + test: MAE, RMSE, R²
  4. Overfitting rule: overfit_ratio > 1.5 AND r2_gap > 0.4
  5. Select best model: prefer non-overfitting, then lowest test RMSE
  6. Save winning model via model_registry.save_model():
       a. Save .pkl locally (models/ folder)
       b. Upload model.pkl + metadata.json to Hugging Face Hub
       c. Mirror metadata (metrics, pollutant, horizon_hours) in MongoDB

Run standalone:
    python src/train.py

Required environment variables (in addition to MONGODB_URI etc.):
    HF_TOKEN, HF_REPO_ID  (see .env.example or config.py)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.database import get_all_features, ensure_indexes
from src.model_registry import save_model
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TEST_RATIO = 0.2   # last 20% of data used for testing (time-ordered)
OVERFIT_RATIO_THRESHOLD = 1.5
OVERFIT_R2_GAP_THRESHOLD = 0.4

# Hyperparameters for ~2k hourly rows, ~46 features, 80/20 time split.
# Lighter than previous settings to reduce training time and avoid overfitting
# on the relatively small dataset (~1,575 train / ~395 test rows).
MODEL_CONFIG = {
    "ridge_alpha": 1.0,
    "rf_n_estimators": 100,
    "rf_max_depth": 8,
    "rf_min_samples_leaf": 10,
    "rf_min_samples_split": 20,
    "gbr_n_estimators": 100,
    "gbr_learning_rate": 0.05,
    "gbr_max_depth": 3,
    "gbr_min_samples_leaf": 10,
    "xgb_n_estimators": 100,
    "xgb_learning_rate": 0.05,
    "xgb_max_depth": 3,
    "xgb_min_child_weight": 8,
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Load ALL feature rows from MongoDB, sorted by datetime.

    Does NOT filter by target column presence — each train_for_target() call
    drops rows where its specific target is NaN via .dropna().  This means
    every target gets the maximum usable training set, and partial rows
    (e.g. a row that has 24h targets but whose 72h future is not yet known)
    contribute correctly to the shorter-horizon models.

    Raises ValueError if the collection is completely empty.
    """
    docs = get_all_features()
    if not docs:
        raise ValueError(
            "No feature rows found in MongoDB.\n"
            "Run:  python src/backfill.py --rebuild-features\n"
            "to populate the features collection first."
        )
    df = pd.DataFrame(docs)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Loaded %d feature rows from MongoDB.", len(df))
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, R² and return as a dict."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def compute_model_metrics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict[str, float | bool]:
    """Compute train/test metrics and overfitting diagnostics."""
    train = evaluate(y_train, y_train_pred)
    test = evaluate(y_test, y_test_pred)

    train_rmse = train["rmse"]
    test_rmse = test["rmse"]
    overfit_gap = round(test_rmse - train_rmse, 4)
    if train_rmse == 0:
        overfit_ratio = float("inf") if test_rmse > 0 else 1.0
    else:
        overfit_ratio = round(test_rmse / train_rmse, 4)

    r2_gap = round(train["r2"] - test["r2"], 4)

    # AND condition: both ratio AND r2_gap must exceed thresholds.
    # Using OR was too aggressive and flagged mildly regularized models.
    is_overfitting = (
        overfit_ratio > OVERFIT_RATIO_THRESHOLD
        and r2_gap > OVERFIT_R2_GAP_THRESHOLD
    )

    return {
        "train_mae": train["mae"],
        "train_rmse": train_rmse,
        "train_r2": train["r2"],
        "test_mae": test["mae"],
        "test_rmse": test_rmse,
        "test_r2": test["r2"],
        "overfit_gap": overfit_gap,
        "overfit_ratio": overfit_ratio,
        "r2_gap": r2_gap,
        "is_overfitting": is_overfitting,
        # Backward-compatible aliases (test metrics used by dashboard)
        "mae": test["mae"],
        "rmse": test_rmse,
        "r2": test["r2"],
    }


def is_overfitting(metrics: dict[str, float | bool]) -> bool:
    """Return True when model exceeds overfitting thresholds."""
    return bool(metrics["is_overfitting"])


def select_best_model(results: dict[str, dict]) -> tuple[str | None, dict | None, bool]:
    """
    Pick the best model for a target.

    Prefer non-overfitting models sorted by lowest test RMSE.
    If all overfit, pick lowest test RMSE and return all_overfitting=True.
    """
    valid = {name: m for name, m in results.items() if "error" not in m}
    if not valid:
        return None, None, False

    non_overfit = {name: m for name, m in valid.items() if not is_overfitting(m)}
    pool = non_overfit if non_overfit else valid
    all_overfitting = not bool(non_overfit)

    best_name = min(pool, key=lambda name: pool[name]["test_rmse"])
    return best_name, valid[best_name], all_overfitting


def print_model_comparison_table(target: str, results: dict[str, dict]) -> None:
    """Print a formatted comparison table for all trained models."""
    lines = [
        "",
        "=" * 105,
        f"Model comparison — {target}",
        "=" * 105,
        f"{'Model':<28} {'Train RMSE':>11} {'Test RMSE':>10} "
        f"{'Train R²':>9} {'Test R²':>8} {'Overfit Ratio':>14} {'Overfitting':>12}",
        "-" * 105,
    ]

    for name, metrics in results.items():
        if "error" in metrics:
            lines.append(f"{name:<28} FAILED — {metrics['error']}")
            continue
        lines.append(
            f"{name:<28} "
            f"{metrics['train_rmse']:>11.4f} "
            f"{metrics['test_rmse']:>10.4f} "
            f"{metrics['train_r2']:>9.4f} "
            f"{metrics['test_r2']:>8.4f} "
            f"{metrics['overfit_ratio']:>14.4f} "
            f"{str(metrics['is_overfitting']):>12}"
        )

    lines.append("=" * 105)
    table = "\n".join(lines)
    print(table)
    logger.info(table)


# ── Model catalogue ────────────────────────────────────────────────────────────

def get_models() -> dict[str, Any]:
    """
    Return a dict of model_name → sklearn-compatible estimator.

    Configurations are sized for ~2k rows and ~46 features.  Tree models
    use shallow depth (max_depth 3-8), moderate leaf constraints, and
    subsampling to limit overfitting on the relatively small dataset.
    Ridge with StandardScaler handles the highly correlated lag features.
    """
    cfg = MODEL_CONFIG
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                # Ridge avoids unstable coefficients from correlated lag/pollutant features
                ("model", Ridge(alpha=cfg["ridge_alpha"])),
            ]
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=cfg["rf_n_estimators"],
            max_depth=cfg["rf_max_depth"],
            min_samples_leaf=cfg["rf_min_samples_leaf"],
            min_samples_split=cfg["rf_min_samples_split"],
            max_features="sqrt",
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=cfg["gbr_n_estimators"],
            learning_rate=cfg["gbr_learning_rate"],
            max_depth=cfg["gbr_max_depth"],
            min_samples_leaf=cfg["gbr_min_samples_leaf"],
            subsample=0.75,
            max_features=0.8,
            random_state=42,
        ),
        "XGBRegressor": XGBRegressor(
            n_estimators=cfg["xgb_n_estimators"],
            learning_rate=cfg["xgb_learning_rate"],
            max_depth=cfg["xgb_max_depth"],
            min_child_weight=cfg["xgb_min_child_weight"],
            subsample=0.75,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        ),
    }


# ── Per-target training ────────────────────────────────────────────────────────

def _parse_target(target: str) -> tuple[str, int]:
    """
    Extract pollutant name and horizon from a target column name.

    Example: 'target_pm2_5_24h' → ('pm2_5', 24)
    """
    # target format: target_<pollutant>_<hours>h
    # e.g. target_pm2_5_24h, target_no2_72h
    parts = target.removeprefix("target_").rsplit("_", 1)
    pollutant = parts[0]
    horizon_h = int(parts[1].rstrip("h"))
    return pollutant, horizon_h


def train_for_target(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """
    Train all 4 candidate models for a single pollutant target column.

    Only rows where *this specific target* is non-null are used for training.
    This means each of the 12 targets gets its own optimally-sized training
    set without requiring all 12 targets to be present in every row.

    Parameters
    ----------
    df : pd.DataFrame
        All feature rows from MongoDB (loaded by load_features()).
    target : str
        One of the 12 pollutant target columns, e.g. 'target_pm2_5_24h'.

    Returns
    -------
    dict
        Summary: target, best_model, best_rmse, all_overfitting, all_results.
    """
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        logger.warning("Features missing from DataFrame (will be skipped): %s", missing)

    if target not in df.columns:
        logger.error("Target column '%s' not found in DataFrame. Skipping.", target)
        return {"target": target, "best_model": None, "best_rmse": float("inf"),
                "all_overfitting": False, "all_results": {}}

    # Drop only rows where THIS target is missing — other targets may be NaN
    sub = df[available_features + [target]].dropna(subset=[target])
    # Fill any remaining NaN in feature columns with 0 (can occur for the
    # first few lag rows or if an optional feature is absent)
    X = sub[available_features].fillna(0).values
    y = sub[target].values

    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(
        "[%s] Train=%d  Test=%d", target, len(X_train), len(X_test)
    )

    models = get_models()
    results: dict[str, dict] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            metrics = compute_model_metrics(y_train, y_train_pred, y_test, y_test_pred)
            results[name] = metrics
            fitted_models[name] = model
            logger.info(
                "  [%s] %s — train RMSE=%.4f  test RMSE=%.4f  "
                "train R²=%.4f  test R²=%.4f  overfit_ratio=%.4f  overfitting=%s",
                target,
                name,
                metrics["train_rmse"],
                metrics["test_rmse"],
                metrics["train_r2"],
                metrics["test_r2"],
                metrics["overfit_ratio"],
                metrics["is_overfitting"],
            )
        except Exception as exc:
            logger.error("  [%s] %s failed: %s", target, name, exc)
            results[name] = {"error": str(exc)}

    print_model_comparison_table(target, results)

    best_name, best_metrics, all_overfitting = select_best_model(results)
    best_model = fitted_models.get(best_name) if best_name else None
    best_rmse = best_metrics["test_rmse"] if best_metrics else float("inf")

    if best_model is not None and best_name is not None and best_metrics is not None:
        if all_overfitting:
            logger.warning(
                "[%s] All models are overfitting — selected %s with lowest test RMSE=%.4f",
                target,
                best_name,
                best_rmse,
            )
        else:
            logger.info(
                "[%s] Best model: %s (test RMSE=%.4f, not overfitting)",
                target,
                best_name,
                best_rmse,
            )
        pollutant, horizon_h = _parse_target(target)
        save_model(
            model=best_model,
            model_name=best_name,
            target=target,
            metrics=best_metrics,
            feature_columns=available_features,
            extra_meta={
                "pollutant": pollutant,
                "horizon_hours": horizon_h,
                "prediction_target_type": "pollutant_concentration",
                "unit": "ug/m3",
            },
        )

    return {
        "target": target,
        "best_model": best_name,
        "best_rmse": best_rmse,
        "all_overfitting": all_overfitting,
        "all_results": results,
    }


# ── Main training pipeline ─────────────────────────────────────────────────────

def run_training() -> list[dict[str, Any]]:
    """Run the full training pipeline across all 12 pollutant targets."""
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
        overfit_note = " (all overfitting)" if s.get("all_overfitting") else ""
        logger.info(
            "Target: %-20s | Best: %-28s | Test RMSE: %.4f%s",
            s["target"],
            s["best_model"],
            s["best_rmse"],
            overfit_note,
        )
