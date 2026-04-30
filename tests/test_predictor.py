
import pandas as pd
import pytest

from src.ml.predictor import predict_one, predict_batch


# 🔹 Sample valid input (same as your model expects)
VALID_INPUT = {
    "ProductionVolume": 900,
    "ProductionCost": 18000,
    "SupplierQuality": 85,
    "DeliveryDelay": 3,
    "DefectRate": 4,
    "QualityScore": 70,
    "MaintenanceHours": 2,
    "DowntimePercentage": 3,
    "InventoryTurnover": 3,
    "StockoutRate": 7,
    "WorkerProductivity": 80,
    "SafetyIncidents": 2,
    "EnergyConsumption": 4000,
    "EnergyEfficiency": 0.2,
    "AdditiveProcessTime": 8,
    "AdditiveMaterialCost": 400
}


# -------------------------------
# ✅ TEST: predict_one (valid input)
# -------------------------------
def test_predict_one_valid():
    result = predict_one(VALID_INPUT)

    assert isinstance(result, dict)
    assert "risk_level" in result
    assert "risk_score" in result


# -------------------------------
# ❌ TEST: predict_one (invalid input)
# -------------------------------
def test_predict_one_invalid():
    invalid_input = {}

    with pytest.raises(Exception):
        predict_one(invalid_input)


# -------------------------------
# ✅ TEST: predict_batch (valid dataframe)
# -------------------------------
def test_predict_batch_valid():
    df = pd.DataFrame([VALID_INPUT, VALID_INPUT])

    result_df = predict_batch(df)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2


# -------------------------------
# ❌ TEST: predict_batch (empty dataframe)
# -------------------------------
def test_predict_batch_empty():
    df = pd.DataFrame()

    with pytest.raises(Exception):
        predict_batch(df)