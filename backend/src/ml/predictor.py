"""
Predictor — loads the trained model and exposes predict_one().
Auto-trains the model on first use if it is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent        # backend/src/ml/
_MODEL_PATH = _HERE.parents[1] / "models" / "quality_model.joblib"

# Cached pipeline (loaded once per process)
_pipeline = None


def _get_pipeline():
    """Return (and cache) the trained pipeline, auto-training if needed."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not _MODEL_PATH.exists():
        logger.warning(
            "Model file not found at %s — auto-training now …", _MODEL_PATH
        )
        # Lazy import to avoid circular deps
        from src.ml.train_model import train_and_save
        train_and_save()

    _pipeline = joblib.load(_MODEL_PATH)
    logger.info("Model loaded from %s", _MODEL_PATH)
    return _pipeline


def predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference for a single sample.

    Args:
        features: dict with keys matching the training feature order.
                  Expected keys (in order):
                    temperature, pressure, humidity, vibration_level

    Returns:
        {
            "prediction": 0 | 1,
            "confidence": float,   # probability of the predicted class
            "label":      str,     # "Pass" | "Defect"
        }

    Raises:
        ValueError: if required feature keys are missing.
    """
    FEATURE_ORDER = ["temperature", "pressure", "humidity", "vibration_level"]

    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    X = np.array([[features[k] for k in FEATURE_ORDER]], dtype=float)

    pipeline  = _get_pipeline()
    pred      = int(pipeline.predict(X)[0])
    proba     = pipeline.predict_proba(X)[0]
    confidence = float(proba[pred])

    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "label":      "Defect" if pred == 1 else "Pass",
    }
