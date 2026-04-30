import pandas as pd
import pytest
from backend.src.ml.preprocessing import preprocess, validate_input_frame


# Real feature columns matching the actual model
REAL_FEATURES = {
    "ProductionVolume": 920,
    "ProductionCost": 18400,
    "SupplierQuality": 84.2,
    "DeliveryDelay": 4,
    "DefectRate": 4.3,
    "QualityScore": 69.5,
    "MaintenanceHours": 3,
    "DowntimePercentage": 4.1,
    "InventoryTurnover": 3.4,
    "StockoutRate": 8.2,
    "WorkerProductivity": 83.4,
    "SafetyIncidents": 5,
    "EnergyConsumption": 4300,
    "EnergyEfficiency": 0.18,
    "AdditiveProcessTime": 8.1,
    "AdditiveMaterialCost": 420,
}


def test_preprocess_returns_dataframe():
    df = pd.DataFrame([REAL_FEATURES])
    result = preprocess(df)
    assert isinstance(result, pd.DataFrame)
    assert result is not None


def test_preprocess_has_correct_columns():
    df = pd.DataFrame([REAL_FEATURES])
    result = preprocess(df)
    for col in REAL_FEATURES:
        assert col in result.columns, f"Missing column: {col}"


def test_preprocess_handles_missing_columns():
    """preprocess() should fill missing columns with 0, not crash."""
    df = pd.DataFrame({"ProductionVolume": [100, 200]})
    result = preprocess(df)
    assert result is not None
    assert "ProductionVolume" in result.columns


def test_preprocess_multiple_rows():
    rows = [REAL_FEATURES, REAL_FEATURES]
    df = pd.DataFrame(rows)
    result = preprocess(df)
    assert len(result) == 2


def test_validate_input_frame_raises_on_missing_columns():
    df = pd.DataFrame({"ProductionVolume": [100]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input_frame(df)


def test_validate_input_frame_passes_with_all_columns():
    df = pd.DataFrame([REAL_FEATURES])
    result = validate_input_frame(df)
    assert result is not None
    assert len(result) == 1
