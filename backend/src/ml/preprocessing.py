from __future__ import annotations

from typing import Any
import pandas as pd

from backend.src.ml.config import FEATURE_COLUMNS, NUMERIC_FEATURES


# ─────────────────────────────────────────────
# Validate DataFrame input
# ─────────────────────────────────────────────
def validate_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures input DataFrame has correct columns and types.
    """

    # Check missing columns
    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Select only required columns
    frame = data[FEATURE_COLUMNS].copy()

    # Convert numeric columns safely
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    return frame


# ─────────────────────────────────────────────
# Convert API payload → DataFrame
# ─────────────────────────────────────────────
def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Converts single JSON payload into validated DataFrame.
    """
    return validate_input_frame(pd.DataFrame([payload]))


# ─────────────────────────────────────────────
# TEST COMPATIBILITY FUNCTION (IMPORTANT)
# ─────────────────────────────────────────────
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function required for tests.
    """
    return validate_input_frame(data)
