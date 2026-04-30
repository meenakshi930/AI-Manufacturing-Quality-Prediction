
import pandas as pd
import pytest

from src.ml import predictor


# 🔹 Sample valid input
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
# ✅ MOCK predict_one
# -------------------------------
def test_predict_one_valid(monkeypatch):
    def fake_predict(data):
        return {"risk_level": "Low", "risk_score": 0.2}

    monkeypatch.setattr(predictor, "predict_one", fake_predict)

    result = predictor.predict_one(VALID_INPUT)

    assert isinstance(result, dict)
    assert "risk_level" in result
    assert "risk_score" in result


# -------------------------------
# ❌ INVALID INPUT
# -------------------------------
def test_predict_one_invalid():
    with pytest.raises(Exception):
        predictor.predict_one({})


# -------------------------------
# ✅ BATCH TEST
# -------------------------------
def test_predict_batch_valid(monkeypatch):
    def fake_batch(df):
        df["risk_score"] = 0.5
        return df

    monkeypatch.setattr(predictor, "predict_batch", fake_batch)

    df = pd.DataFrame([VALID_INPUT, VALID_INPUT])
    result_df = predictor.predict_batch(df)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2


# -------------------------------
# ❌ EMPTY DATAFRAME
# -------------------------------
def test_predict_batch_empty():
    with pytest.raises(Exception):
        predictor.predict_batch(pd.DataFrame())
