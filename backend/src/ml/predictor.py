"""
ML prediction logic.
Loads the trained pipeline from backend/models/model.pkl.
Auto-trains on first use if the file is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from backend.src.ml.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_MODEL_PATH = _BACKEND_DIR / "models" / "model.pkl"

# ✅ Use correct feature order from config
FEATURE_ORDER = FEATURE_COLUMNS

# Module-level cache
_pipeline = None


# ──────────────────────────────────────────────────────────────────────────────
def _get_pipeline():
    """Load model (auto-train if missing)."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    if not _MODEL_PATH.exists():
        logger.warning("Model not found — training new model...")
        from backend.src.ml.train_model import train_and_save
        train_and_save()

    _pipeline = joblib.load(_MODEL_PATH)
    logger.info("Model loaded successfully")

    return _pipeline


# ──────────────────────────────────────────────────────────────────────────────
def predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict a single record.
    """

    # Validate features
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    # Convert to array
    X = np.array([[features[k] for k in FEATURE_ORDER]], dtype=float)

    pipeline = _get_pipeline()

    pred = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]
    confidence = float(proba[pred])

    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "label": "Defect" if pred == 1 else "Pass",
    }


# ──────────────────────────────────────────────────────────────────────────────
def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Predict multiple rows.
    """

    # Validate columns
    missing_cols = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    pipeline = _get_pipeline()

    X = frame[FEATURE_ORDER].values

    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    results = frame.copy()

    results["prediction"] = preds
    results["confidence"] = [
        round(float(probas[i][preds[i]]), 4) for i in range(len(preds))
    ]
    results["label"] = ["Defect" if p == 1 else "Pass" for p in preds]

    return results
