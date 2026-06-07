"""
cleanup_old_models.py — Safe model versioning cleanup utility.

What this script does
---------------------
1. Marks old MongoDB model_registry records as active=False for the
   specified target names (default: old target_aqi_* targets).
2. Optionally deletes matching local .pkl files from the models/ folder.
3. Does NOT touch Hugging Face Hub files (old HF files are harmless
   because prediction code always resolves models via MongoDB metadata,
   never by scanning HF directly).

Usage
-----
# Deactivate old target_aqi_* records in MongoDB (default behaviour)
python src/cleanup_old_models.py

# Deactivate specific targets
python src/cleanup_old_models.py --targets target_aqi_24h target_aqi_48h target_aqi_72h

# Also delete matching local .pkl files
python src/cleanup_old_models.py --delete-local-pkl

# Dry-run — show what would happen without making changes
python src/cleanup_old_models.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.database import get_collection
from src.config import MODELS_COLLECTION
from src.utils import setup_logging

setup_logging()
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DEFAULT_OLD_TARGETS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]


def mark_inactive(targets: list[str], dry_run: bool) -> int:
    """
    Set active=False on all MongoDB model_registry records matching any of
    the given target names.

    Returns the number of documents updated.
    """
    col = get_collection(MODELS_COLLECTION)
    query = {"target": {"$in": targets}, "active": {"$ne": False}}
    docs = list(col.find(query, {"_id": 0, "target": 1, "model_name": 1,
                                  "trained_at": 1, "active": 1}))

    if not docs:
        log.info("No active records found for targets: %s", targets)
        return 0

    log.info("Records to deactivate (%d):", len(docs))
    for d in docs:
        log.info(
            "  target=%-25s  algorithm=%-28s  trained_at=%s  active=%s",
            d.get("target"), d.get("model_name"),
            str(d.get("trained_at", "?"))[:19],
            d.get("active", "unknown"),
        )

    if dry_run:
        log.info("[DRY-RUN] Would mark %d record(s) as active=False.", len(docs))
        return len(docs)

    result = col.update_many(query, {"$set": {"active": False}})
    log.info("Marked %d record(s) as active=False in model_registry.", result.modified_count)
    return result.modified_count


def delete_local_pkls(targets: list[str], dry_run: bool) -> int:
    """
    Delete local .pkl files whose filenames contain any of the given target
    strings (e.g. 'target_aqi_24h').

    Returns the number of files deleted.
    """
    if not MODELS_DIR.exists():
        log.info("models/ directory does not exist — nothing to delete.")
        return 0

    to_delete = [
        f for f in MODELS_DIR.glob("*.pkl")
        if any(t in f.name for t in targets)
    ]

    if not to_delete:
        log.info("No local .pkl files matched targets: %s", targets)
        return 0

    log.info("Local .pkl files to delete (%d):", len(to_delete))
    for f in to_delete:
        log.info("  %s", f.name)

    if dry_run:
        log.info("[DRY-RUN] Would delete %d local .pkl file(s).", len(to_delete))
        return len(to_delete)

    for f in to_delete:
        f.unlink()
        log.info("  Deleted: %s", f.name)
    log.info("Deleted %d local .pkl file(s).", len(to_delete))
    return len(to_delete)


def print_active_summary() -> None:
    """Print a summary of currently active models in MongoDB."""
    col = get_collection(MODELS_COLLECTION)
    active = list(col.find({"active": True}, {"_id": 0, "target": 1,
                                               "model_name": 1, "trained_at": 1}
                           ).sort("trained_at", -1))
    inactive = col.count_documents({"active": False})
    no_flag  = col.count_documents({"active": {"$exists": False}})

    log.info("")
    log.info("=== model_registry summary ===")
    log.info("  Active models   : %d", len(active))
    log.info("  Inactive models : %d", inactive)
    log.info("  No flag (legacy): %d", no_flag)
    if active:
        log.info("  Active targets:")
        for d in active:
            log.info(
                "    target=%-25s  algorithm=%-28s  trained_at=%s",
                d.get("target"), d.get("model_name"),
                str(d.get("trained_at", "?"))[:19],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deactivate old model_registry entries and optionally clean local .pkl files."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_OLD_TARGETS,
        metavar="TARGET",
        help=(
            "Target names to deactivate. "
            f"Default: {DEFAULT_OLD_TARGETS}"
        ),
    )
    parser.add_argument(
        "--delete-local-pkl",
        action="store_true",
        help="Also delete matching local .pkl files from the models/ folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making any changes.",
    )
    args = parser.parse_args()

    log.info("=== CLEANUP UTILITY%s ===", " [DRY-RUN]" if args.dry_run else "")
    log.info("Targets: %s", args.targets)

    print_active_summary()

    log.info("")
    log.info("--- Step 1: Deactivating MongoDB records ---")
    n_db = mark_inactive(args.targets, dry_run=args.dry_run)

    if args.delete_local_pkl:
        log.info("")
        log.info("--- Step 2: Deleting local .pkl files ---")
        n_pkl = delete_local_pkls(args.targets, dry_run=args.dry_run)
    else:
        log.info("")
        log.info("--- Step 2: Skipped (pass --delete-local-pkl to also remove local files) ---")
        n_pkl = 0

    log.info("")
    log.info("=== DONE ===")
    log.info("  MongoDB records deactivated : %d", n_db)
    log.info("  Local .pkl files deleted    : %d", n_pkl)
    log.info("  HF Hub files                : untouched (not required for correctness)")
    log.info("")
    log.info("Note: load_model() resolves models via MongoDB active=True records.")
    log.info("      Old HF files are harmless — prediction code never scans HF directly.")
