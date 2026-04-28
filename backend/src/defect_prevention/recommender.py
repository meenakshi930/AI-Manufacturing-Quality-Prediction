from __future__ import annotations

from typing import Any


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def prevention_recommendations(record: dict[str, Any], probability: float) -> list[str]:
    recommendations: list[str] = []

    if record["SupplierQuality"] < 88:
        recommendations.append("Audit the supplier lot and tighten incoming material inspection.")
    if record["DeliveryDelay"] >= 3:
        recommendations.append("Review delayed supplier deliveries because schedule pressure can raise defect risk.")
    if record["DefectRate"] > 3.2:
        recommendations.append("Pause the line for first-article inspection and root-cause review.")
    if record["QualityScore"] < 75:
        recommendations.append("Increase sampling frequency and validate gauge calibration.")
    if record["MaintenanceHours"] < 6:
        recommendations.append("Schedule additional preventive maintenance before the next production cycle.")
    if record["DowntimePercentage"] > 3:
        recommendations.append("Investigate downtime drivers and stabilize machine availability.")
    if record["StockoutRate"] > 6:
        recommendations.append("Reduce stockout risk to avoid rushed substitutions and quality escapes.")
    if record["WorkerProductivity"] < 86:
        recommendations.append("Balance workload or add operator support for this production cell.")
    if record["SafetyIncidents"] >= 4:
        recommendations.append("Run a safety and process discipline review before continuing high-volume output.")
    if record["EnergyEfficiency"] < 0.22:
        recommendations.append("Check machine energy efficiency and calibrate high-consumption equipment.")
    if record["AdditiveProcessTime"] > 7.5:
        recommendations.append("Tune additive process parameters to reduce cycle-time variation.")
    if record["AdditiveMaterialCost"] > 390:
        recommendations.append("Inspect additive material handling and supplier certificate compliance.")

    if not recommendations and probability < 0.4:
        recommendations.append("Continue normal production and monitor standard SPC control charts.")
    elif not recommendations:
        recommendations.append("Review process settings and inspect the first five output units.")

    return recommendations
