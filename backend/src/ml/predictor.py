from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from backend.src.defect_prevention.recommender import (
    prevention_recommendations,
    risk_level,
)
from backend.src.ml.config import MODEL_PATH
from backend.src.ml.preprocessing import payload_to_frame


# -------------------------------
# Load Model (cached)
# -------------------------------
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m backend.src.ml.train_model` first."
        )
    return joblib.load(MODEL_PATH)


# -------------------------------
# Single Prediction
# -------------------------------
def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload_to_frame(payload)
    model = load_model()

    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(probability >= 0.5)

    input_record = frame.iloc[0].to_dict()

    return {
        "defect_prediction": prediction,
        "defect_label": "Defective" if prediction else "Good",
        "defect_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "recommendations": prevention_recommendations(input_record, probability),
    }


# -------------------------------
# Batch Prediction
# -------------------------------
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
