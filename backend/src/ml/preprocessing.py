from __future__ import annotations

from typing import Any

import pandas as pd

from src.ml.config import FEATURE_COLUMNS, NUMERIC_FEATURES


def validate_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = data[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return validate_input_frame(pd.DataFrame([payload]))
