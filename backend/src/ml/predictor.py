"""
ML prediction logic.
Loads the trained pipeline from backend/models/quality_model.joblib.
Auto-trains on first use if the file is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# __file__ = backend/src/ml/predictor.py
# .parents[2] = backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_MODEL_PATH  = _BACKEND_DIR / "models" / "quality_model.joblib"

# Feature order must match what was used during training
FEATURE_ORDER = ["temperature", "pressure", "humidity", "vibration_level"]

# Module-level cache — loaded once per process
_pipeline = None


def _get_pipeline():
    """Return the cached pipeline, auto-training if the model file is missing."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not _MODEL_PATH.exists():
        logger.warning("Model not found at %s — auto-training now …", _MODEL_PATH)
        from backend.src.ml.train_model import train_and_save
        train_and_save()

    _pipeline = joblib.load(_MODEL_PATH)
    logger.info("Model loaded from %s", _MODEL_PATH)
    return _pipeline


def predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference for a single sample.

    Args:
        features: dict with keys matching FEATURE_ORDER:
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
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    X         = np.array([[features[k] for k in FEATURE_ORDER]], dtype=float)
    pipeline  = _get_pipeline()
    pred      = int(pipeline.predict(X)[0])
    proba     = pipeline.predict_proba(X)[0]
    confidence = float(proba[pred])

    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "label":      "Defect" if pred == 1 else "Pass",
    }
def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in frame.iterrows():
        record = row.to_dict()

        try:
            prediction = predict_one(record)

            result_row = {
                **record,
                **prediction,
                "error": None,
            }

        except Exception as e:
            result_row = {
                **record,
                "defect_prediction": None,
                "defect_label": None,
                "defect_probability": None,
                "risk_level": None,
                "recommendations": None,
                "error": str(e),
            }

        results.append(result_row)

    return pd.DataFrame(results)
