from backend.src.defect_prevention.recommender import prevention_recommendations, risk_level


def test_risk_level_thresholds():
    assert risk_level(0.2) == "Low"
    assert risk_level(0.5) == "Medium"
    assert risk_level(0.8) == "High"


def test_recommendations_for_high_risk_record():
    record = {
        "ProductionVolume": 900,
        "ProductionCost": 18000,
        "SupplierQuality": 84,
        "DeliveryDelay": 4,
        "DefectRate": 4.1,
        "QualityScore": 70,
        "MaintenanceHours": 3,
        "DowntimePercentage": 4.2,
        "InventoryTurnover": 3.2,
        "StockoutRate": 8.1,
        "WorkerProductivity": 83,
        "SafetyIncidents": 5,
        "EnergyConsumption": 4400,
        "EnergyEfficiency": 0.18,
        "AdditiveProcessTime": 8.4,
        "AdditiveMaterialCost": 420,
    }

    recommendations = prevention_recommendations(record, 0.82)

    assert len(recommendations) >= 5
    assert any("supplier" in item.lower() for item in recommendations)
    assert any("maintenance" in item.lower() for item in recommendations)
