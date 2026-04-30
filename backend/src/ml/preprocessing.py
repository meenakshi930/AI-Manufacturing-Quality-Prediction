from __future__ import annotations
from typing import Any
import pandas as pd
import sys
import os

# FIX: Automatically adds the project root to the path so imports work everywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from backend.src.ml.config import FEATURE_COLUMNS, NUMERIC_FEATURES
except ImportError:
    # Local fallback if running from within the ml folder
    from config import FEATURE_COLUMNS, NUMERIC_FEATURES

# ─────────────────────────────────────────────
# Strict validation (used in API & model)
# ─────────────────────────────────────────────

def validate_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validates columns and types. Used by the API to ensure 
    incoming data matches the model's requirements.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    # Force correct column order as expected by the model
    frame = data[FEATURE_COLUMNS].copy()
    
    try:
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    except Exception as e:
        raise ValueError(f"Invalid numeric data provided: {str(e)}")
        
    return frame

# ─────────────────────────────────────────────
# Convert API payload → DataFrame
# ─────────────────────────────────────────────

def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Converts a single JSON dictionary into a validated DataFrame."""
    return validate_input_frame(pd.DataFrame([payload]))

# ─────────────────────────────────────────────
# Test-friendly preprocessing
# ─────────────────────────────────────────────

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Used for tests and data generation:
    - Fills missing columns with 0
    - Coerces bad data to 0 instead of crashing
    """
    df = data.copy()
    
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            
    df = df[FEATURE_COLUMNS]
    
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
    return df
