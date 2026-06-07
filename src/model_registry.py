"""
model_registry.py — Unified model persistence and resolution.

Serialises estimators locally, optionally mirrors artefacts to Hugging Face Hub,
and records authoritative metadata in MongoDB. Load resolves the active model
for an exact target name: local cache when present, otherwise Hub download.
When Hub credentials are absent, local-only mode still registers metadata.
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
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """
    Serialise a fitted estimator locally and register it in Hub + MongoDB.

    Prior active records for the same target are deactivated; the new row
    becomes the sole active model for that target.
    """
    trained_at = datetime.now(timezone.utc)
    timestamp  = trained_at.strftime("%Y%m%d_%H%M%S")
    filename   = f"best_model_{target}_{timestamp}.pkl"
    local_path = MODELS_DIR / filename

    # Local serialisation
    joblib.dump(model, local_path)
    logger.info("Model saved locally: %s", local_path)

    # Remote upload (model.pkl + metadata.json) when Hub is configured
    hf_model_path, hf_metadata_path = upload_model_to_hf(
        local_model_path=local_path,
        target=target,
        timestamp=timestamp,
        metrics=metrics,
        feature_columns=feature_columns,
        model_name=model_name,
    )

    # MongoDB metadata mirror (active=True for this target)
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
    if extra_meta:
        metadata.update(extra_meta)
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
    Load the active model for an exact target name.

    Resolution: MongoDB active record → local .pkl if present → Hub download.
    Legacy records without active=True fall back to the newest matching row.
    """
    meta = get_latest_model_metadata(target=target)
    if meta is None:
        raise FileNotFoundError(
            f"No model metadata in MongoDB for target={target!r}. "
            "Run train.py first."
        )

    # Audit log before deserialisation
    trained_at_str = (
        meta["trained_at"].isoformat()
        if hasattr(meta.get("trained_at"), "isoformat")
        else str(meta.get("trained_at", "unknown"))
    )
    logger.info(
        "Loading model | target=%-25s | algorithm=%-28s | trained_at=%s | hf_path=%s | active=%s",
        meta.get("target", "?"),
        meta.get("model_name", "?"),
        trained_at_str,
        meta.get("hf_model_path") or "local-only",
        meta.get("active", "unknown"),
    )

    local_path = Path(meta.get("local_model_path", ""))

    # Local cache hit
    if local_path.exists():
        model = joblib.load(local_path)
        logger.info("  -> Loaded from local cache: %s", local_path.name)
        return model, meta

    # Hub fetch when local artefact is missing (typical on fresh CI runners)
    hf_model_path = meta.get("hf_model_path")
    hf_repo_id    = meta.get("hf_repo_id")

    if not hf_model_path or not hf_repo_id:
        raise FileNotFoundError(
            f"Local .pkl missing at '{local_path}' and no HF Hub path stored "
            f"in metadata for target={target!r}. "
            "Re-run train.py to retrain the model."
        )

    logger.info("  -> Local .pkl absent — downloading from HF Hub: %s/%s", hf_repo_id, hf_model_path)
    downloaded = download_model_from_hf(
        repo_id=hf_repo_id,
        filename=hf_model_path,
        local_dest=local_path,
    )
    if downloaded is None:
        raise RuntimeError(
            f"HF Hub download failed for {hf_repo_id}/{hf_model_path}. "
            "Check HF_TOKEN and HF_REPO_ID, or re-run train.py."
        )

    model = joblib.load(downloaded)
    logger.info("  -> Downloaded and cached at: %s", downloaded)
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
