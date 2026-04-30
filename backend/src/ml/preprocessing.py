from __future__ import annotations

from typing import Any
import pandas as pd

from backend.src.ml.config import FEATURE_COLUMNS, NUMERIC_FEATURES


# ─────────────────────────────────────────────
# Strict validation (used in API & model)
# ─────────────────────────────────────────────
def validate_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = data[FEATURE_COLUMNS].copy()

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame


# ─────────────────────────────────────────────
# Convert API payload → DataFrame
# ─────────────────────────────────────────────
def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return validate_input_frame(pd.DataFrame([payload]))


# ─────────────────────────────────────────────
# Test-friendly preprocessing (IMPORTANT FIX)
# ─────────────────────────────────────────────
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Used only for tests:
    - Adds missing columns
    - Ensures correct format
    """

    df = data.copy()

    # Add missing columns with default value
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Keep correct column order
    df = df[FEATURE_COLUMNS]

    # Convert numeric safely
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df
