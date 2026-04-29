from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.defect_prevention.recommender import prevention_recommendations, risk_level
from src.ml.config import MODEL_PATH
from src.ml.preprocessing import payload_to_frame, validate_input_frame


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/ml/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


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


def predict_batch(data: pd.DataFrame) -> pd.DataFrame:
    frame = validate_input_frame(data)
    model = load_model()
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = data.copy()
    result["defect_prediction"] = predictions
    result["defect_probability"] = probabilities.round(4)
    result["risk_level"] = [risk_level(float(probability)) for probability in probabilities]
    return result
