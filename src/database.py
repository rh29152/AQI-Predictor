"""
database.py — MongoDB Atlas connection and helper functions.

All MongoDB interactions live here so every other module imports from
a single, testable interface.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

from src.config import (
    MONGODB_URI, DB_NAME,
    RAW_COLLECTION, FEATURES_COLLECTION,
    MODELS_COLLECTION, PREDICTIONS_COLLECTION,
)

logger = logging.getLogger(__name__)

# ── Singleton client ───────────────────────────────────────────────────────────
_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return (or create) a singleton MongoClient."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
            _client.admin.command("ping")
            logger.info("Connected to MongoDB Atlas.")
        except ConnectionFailure as exc:
            logger.error("Could not connect to MongoDB: %s", exc)
            raise
    return _client


def get_collection(name: str) -> Collection:
    """Return a collection from the configured database."""
    return get_client()[DB_NAME][name]


# ── Index helpers ──────────────────────────────────────────────────────────────

def ensure_indexes() -> None:
    """Create indexes that speed up common queries (idempotent)."""
    # raw_data: unique per (datetime, city) — prevents duplicate hourly records
    raw = get_collection(RAW_COLLECTION)
    raw.create_index([("datetime", ASCENDING), ("city", ASCENDING)], unique=True)

    # features: unique per datetime — upsert by this key
    feats = get_collection(FEATURES_COLLECTION)
    feats.create_index([("datetime", ASCENDING)], unique=True)

    # model_registry: fast lookup by (target, active) + recency sort
    models = get_collection(MODELS_COLLECTION)
    models.create_index([("trained_at", ASCENDING)])
    models.create_index([("target", ASCENDING), ("trained_at", ASCENDING)])
    models.create_index([("target", ASCENDING), ("active", ASCENDING), ("trained_at", ASCENDING)])

    # predictions: lookup by horizon and time
    preds = get_collection(PREDICTIONS_COLLECTION)
    preds.create_index([("predicted_at", ASCENDING)])
    preds.create_index([("horizon_hours", ASCENDING), ("predicted_at", ASCENDING)])

    logger.info("MongoDB indexes ensured.")


# ── Raw data ───────────────────────────────────────────────────────────────────

def insert_raw_data(record: dict[str, Any]) -> None:
    """
    Upsert a single raw API record keyed by (datetime, city).

    Parameters
    ----------
    record : dict
        Must contain 'datetime' (datetime) and 'city' (str) fields.
    """
    col = get_collection(RAW_COLLECTION)
    filter_key = {"datetime": record["datetime"], "city": record["city"]}
    try:
        col.update_one(filter_key, {"$set": record}, upsert=True)
    except OperationFailure as exc:
        logger.error("Failed to insert raw record: %s", exc)
        raise


def insert_raw_batch(records: list[dict[str, Any]]) -> int:
    """
    Upsert a list of raw records. Returns the number of records processed.
    """
    for rec in records:
        insert_raw_data(rec)
    return len(records)


# ── Feature store ──────────────────────────────────────────────────────────────

def insert_features(record: dict[str, Any]) -> None:
    """
    Upsert a single engineered feature row keyed by 'datetime'.
    """
    col = get_collection(FEATURES_COLLECTION)
    filter_key = {"datetime": record["datetime"]}
    try:
        col.update_one(filter_key, {"$set": record}, upsert=True)
    except OperationFailure as exc:
        logger.error("Failed to insert feature record: %s", exc)
        raise


def insert_features_batch(records: list[dict[str, Any]]) -> int:
    """Upsert a list of feature rows."""
    for rec in records:
        insert_features(rec)
    return len(records)


def get_training_data(limit: int = 0) -> list[dict[str, Any]]:
    """
    Retrieve feature documents that have ALL 12 pollutant target columns
    populated (4 pollutants × 3 horizons = 12 targets).

    Rows inserted by the incremental (hourly) pipeline do NOT have target
    columns because future concentrations are unknown — they are excluded here
    so the training pipeline never trains on incomplete labels.

    Parameters
    ----------
    limit : int
        Max number of documents to return (0 = no limit).
    """
    from src.config import POLLUTANTS_TO_FORECAST, FORECAST_HORIZONS  # noqa: PLC0415
    col = get_collection(FEATURES_COLLECTION)
    query: dict[str, Any] = {}
    for poll in POLLUTANTS_TO_FORECAST:
        for h in FORECAST_HORIZONS:
            key = f"target_{poll}_{h}h"
            query[key] = {"$exists": True, "$ne": None}
    cursor = col.find(query, {"_id": 0}).sort("datetime", ASCENDING)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def get_raw_history(n: int = 73) -> list[dict[str, Any]]:
    """
    Return the *n* most recent raw records sorted ascending by datetime.

    Used by the incremental feature pipeline to obtain enough historical
    context to compute lag (up to 48 h) and rolling (24 h) features for
    the latest single record.

    Parameters
    ----------
    n : int
        Number of records to fetch.  Default 73 = 72 lag-context rows
        + 1 latest row.
    """
    col = get_collection(RAW_COLLECTION)
    docs = list(col.find({}, {"_id": 0}).sort("datetime", -1).limit(n))
    return list(reversed(docs))  # oldest first


def get_raw_history_up_to(end_dt: datetime, n: int = 73) -> list[dict[str, Any]]:
    """
    Return up to *n* raw records with datetime <= end_dt, sorted ascending.

    Used by catch-up feature engineering to build lag context ending at a
    specific hourly timestamp.
    """
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    col = get_collection(RAW_COLLECTION)
    docs = list(
        col.find({"datetime": {"$lte": end_dt}}, {"_id": 0})
        .sort("datetime", -1)
        .limit(n)
    )
    return list(reversed(docs))


def get_latest_features(n: int = 48) -> list[dict[str, Any]]:
    """
    Return the *n* most recent feature rows (sorted ascending by datetime).
    Used by the prediction pipeline.
    """
    col = get_collection(FEATURES_COLLECTION)
    docs = list(
        col.find({}, {"_id": 0})
        .sort("datetime", -1)
        .limit(n)
    )
    return list(reversed(docs))


def get_latest_feature_datetime() -> datetime | None:
    """Return the datetime of the most recent row in the features collection."""
    col = get_collection(FEATURES_COLLECTION)
    doc = col.find_one({}, {"datetime": 1}, sort=[("datetime", -1)])
    if not doc:
        return None
    dt = doc["datetime"]
    if isinstance(dt, datetime) and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_all_features() -> list[dict[str, Any]]:
    """
    Return ALL feature rows sorted ascending by datetime.

    Unlike get_training_data(), this does NOT filter by target column
    presence.  The training pipeline uses this so each call to
    train_for_target() can drop only the rows missing *that specific*
    target column, maximising the training set for each of the 12 targets.
    Hourly incremental rows (no targets at all) are naturally excluded
    per-target by the .dropna() in train_for_target().
    """
    col = get_collection(FEATURES_COLLECTION)
    return list(col.find({}, {"_id": 0}).sort("datetime", ASCENDING))


# ── Model registry ─────────────────────────────────────────────────────────────

def save_model_metadata(metadata: dict[str, Any]) -> None:
    """
    Persist model training metadata to MongoDB.

    Before inserting the new document, all existing records for the same
    target are marked active=False so that load_model() always resolves to
    exactly one active model per target.  The new document is inserted with
    active=True.

    Expected keys: model_name, model_path, target, metrics,
                   trained_at, feature_columns.
    """
    col = get_collection(MODELS_COLLECTION)
    metadata.setdefault("saved_at", datetime.now(timezone.utc))

    target = metadata.get("target")
    if target:
        # Deactivate all previous models for this exact target
        result = col.update_many(
            {"target": target, "active": {"$ne": False}},
            {"$set": {"active": False}},
        )
        if result.modified_count:
            logger.info(
                "Deactivated %d older model(s) for target='%s'.",
                result.modified_count, target,
            )

    # Mark the incoming document as the active model
    metadata["active"] = True

    try:
        col.insert_one(metadata)
        logger.info("Model metadata saved: %s (active=True)", metadata.get("model_name"))
    except OperationFailure as exc:
        logger.error("Failed to save model metadata: %s", exc)
        raise


# ── Predictions ────────────────────────────────────────────────────────────────

def save_prediction(prediction: dict[str, Any]) -> None:
    """
    Persist a forecast record to the predictions collection.

    Expected keys:
        predicted_at   (datetime) — when the forecast was generated
        target         (str)      — e.g. 'target_aqi_24h'
        aqi            (float)    — predicted AQI
        label          (str)      — human-readable AQI category
        model_name     (str)      — model used
    """
    col = get_collection(PREDICTIONS_COLLECTION)
    prediction.setdefault("predicted_at", datetime.now(timezone.utc))
    try:
        col.insert_one(prediction)
    except OperationFailure as exc:
        logger.error("Failed to save prediction: %s", exc)
        raise


def get_latest_model_metadata(target: str | None = None) -> dict[str, Any] | None:
    """
    Return metadata for the active model for a given target.

    Resolution order:
      1. Exact target name match (e.g. 'target_pm2_5_24h').
      2. active=True preferred; falls back to any doc if no active record
         exists (e.g. legacy models trained before the active flag was added).
      3. Sorted by trained_at descending to always return the newest.

    Parameters
    ----------
    target : str or None
        If provided, filter by exact target column name.
        None → the most recently trained active model across all targets.
    """
    col = get_collection(MODELS_COLLECTION)

    base_query: dict[str, Any] = {}
    if target:
        base_query["target"] = target

    # Prefer active=True; fall back to any record for backward compat
    active_query = {**base_query, "active": True}
    doc = col.find_one(active_query, {"_id": 0}, sort=[("trained_at", -1)])
    if doc is None:
        doc = col.find_one(base_query, {"_id": 0}, sort=[("trained_at", -1)])
    return doc
