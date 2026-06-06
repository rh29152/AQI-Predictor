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

    # model_registry: sort by trained_at to find the latest model quickly
    models = get_collection(MODELS_COLLECTION)
    models.create_index([("trained_at", ASCENDING)])
    models.create_index([("target", ASCENDING), ("trained_at", ASCENDING)])

    # predictions: sort by predicted_at for latest forecast lookup
    preds = get_collection(PREDICTIONS_COLLECTION)
    preds.create_index([("predicted_at", ASCENDING)])

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
    Retrieve feature documents that have ALL three target columns populated.

    Rows inserted by the incremental (hourly) pipeline do NOT have target
    columns because future data is unknown — they are excluded here so the
    training pipeline never trains on incomplete labels.

    Parameters
    ----------
    limit : int
        Max number of documents to return (0 = no limit).
    """
    col = get_collection(FEATURES_COLLECTION)
    query = {
        "target_aqi_24h": {"$exists": True, "$ne": None},
        "target_aqi_48h": {"$exists": True, "$ne": None},
        "target_aqi_72h": {"$exists": True, "$ne": None},
    }
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


# ── Model registry ─────────────────────────────────────────────────────────────

def save_model_metadata(metadata: dict[str, Any]) -> None:
    """
    Persist model training metadata to MongoDB.

    Expected keys: model_name, model_path, target, metrics,
                   trained_at, feature_columns.
    """
    col = get_collection(MODELS_COLLECTION)
    metadata.setdefault("saved_at", datetime.now(timezone.utc))
    try:
        col.insert_one(metadata)
        logger.info("Model metadata saved: %s", metadata.get("model_name"))
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
    Return metadata for the most recently trained model.

    Parameters
    ----------
    target : str or None
        If provided, filter by target column (e.g. 'target_aqi_24h').
    """
    col = get_collection(MODELS_COLLECTION)
    query: dict[str, Any] = {}
    if target:
        query["target"] = target
    doc = col.find_one(query, {"_id": 0}, sort=[("trained_at", -1)])
    return doc
