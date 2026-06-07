"""
flush_and_reset.py — One-time migration reset for features, models, and predictions.

Clears MongoDB features, model_registry, and predictions collections; removes
local models/*.pkl; optionally wipes Hugging Face Hub model paths. raw_data is
preserved. Intended for schema or target-definition migrations only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_collection, get_client
from src.config import FEATURES_COLLECTION, MODELS_COLLECTION, PREDICTIONS_COLLECTION
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── 1. MongoDB: drop features + model_registry + predictions ───────────────────
log.info("=== Step 1: Clearing MongoDB collections ===")

feats_col = get_collection(FEATURES_COLLECTION)
n_feats = feats_col.count_documents({})
feats_col.delete_many({})
log.info("  Deleted %d documents from '%s'", n_feats, FEATURES_COLLECTION)

models_col = get_collection(MODELS_COLLECTION)
n_models = models_col.count_documents({})
models_col.delete_many({})
log.info("  Deleted %d documents from '%s'", n_models, MODELS_COLLECTION)

preds_col = get_collection(PREDICTIONS_COLLECTION)
n_preds = preds_col.count_documents({})
preds_col.delete_many({})
log.info("  Deleted %d documents from '%s'", n_preds, PREDICTIONS_COLLECTION)

# ── 2. Local models/ folder ────────────────────────────────────────────────────
log.info("=== Step 2: Clearing local models/ folder ===")
models_dir = Path(__file__).resolve().parent.parent / "models"
pkl_files = list(models_dir.glob("*.pkl"))
for f in pkl_files:
    f.unlink()
    log.info("  Deleted %s", f.name)
if not pkl_files:
    log.info("  (no .pkl files found)")

# ── 3. HF Hub: delete all model files ─────────────────────────────────────────
log.info("=== Step 3: Cleaning HF Hub repo ===")

HF_TOKEN  = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if not HF_TOKEN or not HF_REPO_ID:
    log.warning("  HF_TOKEN or HF_REPO_ID not set — skipping HF cleanup.")
else:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type="model")
        model_files = [f for f in files if f.startswith("models/")]
        if model_files:
            api.delete_files(
                repo_id=HF_REPO_ID,
                delete_patterns=["models/**"],
                token=HF_TOKEN,
                repo_type="model",
                commit_message="[migration] Remove old target_aqi_* models",
            )
            log.info("  Deleted %d files from HF Hub repo '%s'", len(model_files), HF_REPO_ID)
        else:
            log.info("  No model files found in HF Hub repo.")
    except Exception as exc:
        log.warning("  HF cleanup skipped/failed: %s", exc)
        log.info("  You can manually delete models via: https://huggingface.co/%s/tree/main", HF_REPO_ID)

log.info("")
log.info("=== FLUSH COMPLETE ===")
log.info("raw_data is untouched (%d records).", get_collection("raw_data").count_documents({}))
log.info("")
log.info("Next steps:")
log.info("  1. py -3 src/backfill.py --rebuild-features")
log.info("  2. py -3 src/train.py")
log.info("  3. py -3 src/predict.py")
