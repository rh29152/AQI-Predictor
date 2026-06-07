"""
hf_model_registry.py — Hugging Face Hub model registry.

Replaces Vertex AI / GCS as the remote model store.
Uses the free huggingface_hub SDK — no billing account required.

Repository layout on HF Hub
----------------------------
<HF_REPO_ID>/
└── models/
    ├── target_aqi_24h/
    │   └── <timestamp>/
    │       ├── model.pkl        ← serialised sklearn/XGBoost model
    │       └── metadata.json    ← metrics, feature_columns, etc.
    ├── target_aqi_48h/
    │   └── ...
    └── target_aqi_72h/
        └── ...

Graceful degradation
--------------------
Every public function checks HF_ENABLED before calling any HF API.
If HF_TOKEN or HF_REPO_ID is missing the functions log a warning and
return None, so the project still runs in local-only mode (models
stored only in the models/ folder).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Config accessor (avoids circular import at module level) ──────────────────

def _cfg():
    from src import config  # noqa: PLC0415
    return config


# ── Repo management ────────────────────────────────────────────────────────────

def create_or_get_repo() -> str | None:
    """
    Create the HF Hub model repository if it does not already exist.

    Uses exist_ok=True so calling this multiple times is safe (idempotent).
    The repository is created as a *model* repo (not dataset/space).

    Returns
    -------
    str : repo_id if successful, None if HF is not configured or on error.
    """
    cfg = _cfg()
    if not cfg.HF_ENABLED:
        logger.warning(
            "HF_TOKEN or HF_REPO_ID not set — skipping HF repo creation."
        )
        return None

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()
        api.create_repo(
            repo_id=cfg.HF_REPO_ID,
            token=cfg.HF_TOKEN,
            repo_type="model",
            private=False,   # public so Streamlit Cloud can pull without a token
            exist_ok=True,
        )
        logger.info("HF Hub repo ready: https://huggingface.co/%s", cfg.HF_REPO_ID)
        return cfg.HF_REPO_ID
    except Exception as exc:
        logger.error("Could not create/verify HF repo: %s", exc)
        return None


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_model_to_hf(
    local_model_path: Path,
    target: str,
    timestamp: str,
    metrics: dict[str, float],
    feature_columns: list[str],
    model_name: str,
) -> tuple[str, str] | tuple[None, None]:
    """
    Upload a trained model .pkl and its metadata.json to HF Hub.

    Files are committed under:
        models/<target>/<timestamp>/model.pkl
        models/<target>/<timestamp>/metadata.json

    huggingface_hub handles Git LFS automatically for large files,
    so no manual LFS configuration is required.

    Parameters
    ----------
    local_model_path : Path
        Local path to the serialised model (.pkl).
    target : str
        Forecast horizon key, e.g. 'target_aqi_24h'.
    timestamp : str
        Run timestamp, e.g. '20260606_120000'.
    metrics : dict
        Train/test metrics and overfitting diagnostics.
    feature_columns : list[str]
        Ordered feature column names the model was trained on.
    model_name : str
        Algorithm class name, e.g. 'RandomForestRegressor'.

    Returns
    -------
    (hf_model_path, hf_metadata_path)  on success
    (None, None)                       when HF is disabled or on error
    """
    cfg = _cfg()
    if not cfg.HF_ENABLED:
        logger.warning(
            "HF not configured — skipping upload for target=%s.", target
        )
        return None, None

    # Ensure the repo exists before uploading
    if create_or_get_repo() is None:
        return None, None

    hf_model_path    = f"models/{target}/{timestamp}/model.pkl"
    hf_metadata_path = f"models/{target}/{timestamp}/metadata.json"

    # Build metadata dict (must be JSON-serialisable)
    metadata_dict: dict[str, Any] = {
        "registry":          "huggingface",
        "hf_repo_id":        cfg.HF_REPO_ID,
        "hf_model_path":     hf_model_path,
        "hf_metadata_path":  hf_metadata_path,
        "target":            target,
        "model_name":        model_name,
        "timestamp":         timestamp,
        "metrics":           metrics,
        "feature_columns":   feature_columns,
        "trained_at":        datetime.now(timezone.utc).isoformat(),
    }
    metadata_bytes = json.dumps(metadata_dict, indent=2).encode("utf-8")

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()

        # ── Upload model.pkl ───────────────────────────────────────────────────
        api.upload_file(
            path_or_fileobj=str(local_model_path),
            path_in_repo=hf_model_path,
            repo_id=cfg.HF_REPO_ID,
            token=cfg.HF_TOKEN,
            repo_type="model",
            commit_message=f"[{target}] Upload {model_name} — {timestamp}",
        )
        logger.info("Uploaded model to HF Hub: %s/%s", cfg.HF_REPO_ID, hf_model_path)

        # ── Upload metadata.json ───────────────────────────────────────────────
        api.upload_file(
            path_or_fileobj=metadata_bytes,
            path_in_repo=hf_metadata_path,
            repo_id=cfg.HF_REPO_ID,
            token=cfg.HF_TOKEN,
            repo_type="model",
            commit_message=f"[{target}] Upload metadata — {timestamp}",
        )
        logger.info(
            "Uploaded metadata to HF Hub: %s/%s", cfg.HF_REPO_ID, hf_metadata_path
        )

        return hf_model_path, hf_metadata_path

    except Exception as exc:
        logger.error(
            "HF upload failed for target=%s: %s", target, exc
        )
        return None, None


# ── Download ───────────────────────────────────────────────────────────────────

def download_model_from_hf(
    repo_id: str,
    filename: str,
    local_dest: Path,
) -> Path | None:
    """
    Download a single file from HF Hub to a local path.

    Used by model_registry.load_model() when the local .pkl cache is missing
    (e.g. on a fresh Streamlit Cloud instance or after CI runner teardown).

    Parameters
    ----------
    repo_id : str
        HF Hub repository ID, e.g. 'your-username/karachi-aqi-models'.
    filename : str
        Path inside the repo, e.g.
        'models/target_aqi_24h/20260606_120000/model.pkl'.
    local_dest : Path
        Where to save the downloaded file on disk.

    Returns
    -------
    Path to the downloaded file, or None on failure.
    """
    cfg = _cfg()
    # Allow download with or without a token (public repos work without one)
    token = cfg.HF_TOKEN if cfg.HF_ENABLED else None

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        # hf_hub_download caches files in ~/.cache/huggingface/
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            repo_type="model",
        )
        # Copy from cache to the expected local models/ location
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, local_dest)
        logger.info(
            "Downloaded model from HF Hub: %s/%s -> %s", repo_id, filename, local_dest
        )
        return local_dest

    except Exception as exc:
        logger.error(
            "HF download failed — repo=%s  file=%s: %s", repo_id, filename, exc
        )
        return None


# ── Latest model lookup ────────────────────────────────────────────────────────

def get_latest_hf_model_metadata(target: str) -> dict[str, Any] | None:
    """
    Return the latest model metadata for a given target from MongoDB.

    MongoDB model_registry stores a metadata mirror for every registered
    model, including hf_repo_id and hf_model_path.  This function is a
    convenience wrapper used by the Streamlit dashboard to find out which
    HF model to download.

    Parameters
    ----------
    target : str
        Forecast horizon, e.g. 'target_aqi_24h'.

    Returns
    -------
    dict or None
    """
    from src.database import get_latest_model_metadata  # noqa: PLC0415
    return get_latest_model_metadata(target=target)
