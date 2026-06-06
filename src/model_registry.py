"""
model_registry.py — Unified model save / load interface (HF Hub backend).

Save workflow
-------------
1. Serialise the best model to models/<filename>.pkl (local cache).
2. Upload model.pkl to HF Hub at:
       models/<target>/<timestamp>/model.pkl
3. Upload metadata.json to HF Hub at:
       models/<target>/<timestamp>/metadata.json
4. Persist a metadata document to MongoDB model_registry:
       {
         registry        : "huggingface",
         hf_repo_id      : str,
         hf_model_path   : str,   # path inside HF repo
         hf_metadata_path: str,
         model_name      : str,
         target          : str,
         metrics         : {mae, rmse, r2},
         feature_columns : [str, ...],
         trained_at      : datetime,
         local_model_path: str,   # local cache (may be absent on new runners)
       }

Load workflow
-------------
1. Query MongoDB for the latest metadata document matching the target.
2. Fast path  — if local_model_path still exists on disk, load directly.
3. Slow path  — download model.pkl from HF Hub using hf_model_path stored
               in metadata, cache to models/ folder, then load.

Graceful degradation
--------------------
When HF_ENABLED is False (token or repo_id missing), HF upload is skipped.
The model is saved locally only; metadata is stored in MongoDB with
hf_model_path=None.  Loading still works from the local cache.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.database import save_model_metadata, get_latest_model_metadata
from src.hf_model_registry import upload_model_to_hf, download_model_from_hf

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Save ───────────────────────────────────────────────────────────────────────

def save_model(
    model: Any,
    model_name: str,
    target: str,
    metrics: dict[str, float],
    feature_columns: list[str],
) -> Path:
    """
    Persist a trained model locally and register it on HF Hub.

    Parameters
    ----------
    model : sklearn / XGBoost estimator
        A fitted model (or Pipeline) exposing fit() / predict().
    model_name : str
        Algorithm class name, e.g. 'RandomForestRegressor'.
    target : str
        Forecast horizon key, e.g. 'target_aqi_24h'.
    metrics : dict
        {'mae': float, 'rmse': float, 'r2': float}
    feature_columns : list[str]
        Ordered feature column names the model was trained on.

    Returns
    -------
    Path
        Absolute path to the locally saved .pkl file.
    """
    trained_at = datetime.now(timezone.utc)
    timestamp  = trained_at.strftime("%Y%m%d_%H%M%S")
    filename   = f"best_model_{target}_{timestamp}.pkl"
    local_path = MODELS_DIR / filename

    # ── Step 1: Save locally ───────────────────────────────────────────────────
    joblib.dump(model, local_path)
    logger.info("Model saved locally: %s", local_path)

    # ── Step 2 & 3: Upload to HF Hub (model.pkl + metadata.json) ──────────────
    hf_model_path, hf_metadata_path = upload_model_to_hf(
        local_model_path=local_path,
        target=target,
        timestamp=timestamp,
        metrics=metrics,
        feature_columns=feature_columns,
        model_name=model_name,
    )

    # ── Step 4: Write metadata mirror to MongoDB ───────────────────────────────
    from src.config import HF_REPO_ID  # noqa: PLC0415  (avoid circular at module level)

    metadata: dict[str, Any] = {
        "registry":          "huggingface" if hf_model_path else "local",
        "hf_repo_id":        HF_REPO_ID   if hf_model_path else None,
        "hf_model_path":     hf_model_path,
        "hf_metadata_path":  hf_metadata_path,
        "model_name":        model_name,
        "target":            target,
        "metrics":           metrics,
        "feature_columns":   feature_columns,
        "trained_at":        trained_at,
        "local_model_path":  str(local_path),
    }
    save_model_metadata(metadata)

    logger.info(
        "Model registered — target=%s  algorithm=%s  hf_path=%s",
        target, model_name,
        hf_model_path or "local only",
    )
    return local_path


# ── Load ───────────────────────────────────────────────────────────────────────

def load_model(target: str | None = None) -> tuple[Any, dict[str, Any]]:
    """
    Load the most recently registered model for a given forecast horizon.

    Resolution order
    ----------------
    1. Read the latest metadata document from MongoDB.
    2. Fast path  — local .pkl still on disk → load directly.
    3. Slow path  — download from HF Hub using metadata.hf_model_path,
                    save to the original local_model_path, then load.

    Parameters
    ----------
    target : str or None
        Forecast horizon, e.g. 'target_aqi_24h'.
        None → most recently trained model across all targets.

    Returns
    -------
    (model, metadata) tuple
        model    — fitted sklearn / XGBoost estimator
        metadata — full MongoDB document dict
    """
    meta = get_latest_model_metadata(target=target)
    if meta is None:
        raise FileNotFoundError(
            f"No model metadata in MongoDB for target={target!r}. "
            "Run train.py first."
        )

    local_path = Path(meta.get("local_model_path", ""))

    # ── Fast path ──────────────────────────────────────────────────────────────
    if local_path.exists():
        model = joblib.load(local_path)
        logger.info(
            "Loaded model '%s' from local cache: %s",
            meta["model_name"], local_path,
        )
        return model, meta

    # ── Slow path: download from HF Hub ────────────────────────────────────────
    hf_model_path = meta.get("hf_model_path")
    hf_repo_id    = meta.get("hf_repo_id")

    if not hf_model_path or not hf_repo_id:
        raise FileNotFoundError(
            f"Local .pkl missing at '{local_path}' and no HF Hub path stored "
            f"in metadata for target={target!r}. "
            "Re-run train.py to retrain the model."
        )

    logger.info(
        "Local .pkl not found — downloading from HF Hub: %s/%s",
        hf_repo_id, hf_model_path,
    )
    downloaded = download_model_from_hf(
        repo_id=hf_repo_id,
        filename=hf_model_path,
        local_dest=local_path,   # cache at the same path for next time
    )
    if downloaded is None:
        raise RuntimeError(
            f"HF Hub download failed for {hf_repo_id}/{hf_model_path}. "
            "Check HF_TOKEN and HF_REPO_ID, or re-run train.py."
        )

    model = joblib.load(downloaded)
    logger.info(
        "Loaded model '%s' from HF Hub → cached at %s",
        meta["model_name"], downloaded,
    )
    return model, meta


# ── List all registered models ─────────────────────────────────────────────────

def list_models(target: str | None = None) -> list[dict[str, Any]]:
    """
    Return metadata for all registered models, newest first.

    Parameters
    ----------
    target : str or None
        Filter by forecast horizon.  None → all targets.
    """
    from src.database import get_collection   # noqa: PLC0415
    from src.config import MODELS_COLLECTION  # noqa: PLC0415

    col = get_collection(MODELS_COLLECTION)
    query: dict[str, Any] = {}
    if target:
        query["target"] = target
    return list(col.find(query, {"_id": 0}).sort("trained_at", -1))
