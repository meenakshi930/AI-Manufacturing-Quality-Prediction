from __future__ import annotations

from typing import Any, Dict


# ✅ Centralized thresholds (easy to tune later)
THRESHOLDS: Dict[str, float] = {
    "SupplierQuality_low": 88,
    "DeliveryDelay_high": 3,
    "DefectRate_high": 3.2,
    "QualityScore_low": 75,
    "MaintenanceHours_low": 5.0,          # tuned (was 6)
    "DowntimePercentage_high": 3,
    "StockoutRate_high": 7.5,             # tuned (was 6)
    "WorkerProductivity_low": 86,
    "SafetyIncidents_high": 6,            # tuned (was 4)
    "EnergyEfficiency_low": 0.22,
    "AdditiveProcessTime_high": 7.5,
    "AdditiveMaterialCost_high": 410,     # tuned (was 390)
}


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def prevention_recommendations(record: dict[str, Any], probability: float) -> list[str]:
    recommendations: list[str] = []

    # ✅ Safe getter (prevents crash if key missing)
    def get(key: str, default: float = 0.0):
        return float(record.get(key, default))

    if get("SupplierQuality") < THRESHOLDS["SupplierQuality_low"]:
        recommendations.append("Audit the supplier lot and tighten incoming material inspection.")

    if get("DeliveryDelay") >= THRESHOLDS["DeliveryDelay_high"]:
        recommendations.append("Review delayed supplier deliveries because schedule pressure can raise defect risk.")

    if get("DefectRate") > THRESHOLDS["DefectRate_high"]:
        recommendations.append("Pause the line for first-article inspection and root-cause review.")

    if get("QualityScore") < THRESHOLDS["QualityScore_low"]:
        recommendations.append("Increase sampling frequency and validate gauge calibration.")

    if get("MaintenanceHours") < THRESHOLDS["MaintenanceHours_low"]:
        recommendations.append("Schedule additional preventive maintenance before the next production cycle.")

    if get("DowntimePercentage") > THRESHOLDS["DowntimePercentage_high"]:
        recommendations.append("Investigate downtime drivers and stabilize machine availability.")

    if get("StockoutRate") > THRESHOLDS["StockoutRate_high"]:
        recommendations.append("Reduce stockout risk to avoid rushed substitutions and quality escapes.")

    if get("WorkerProductivity") < THRESHOLDS["WorkerProductivity_low"]:
        recommendations.append("Balance workload or add operator support for this production cell.")

    if get("SafetyIncidents") >= THRESHOLDS["SafetyIncidents_high"]:
        recommendations.append("Run a safety and process discipline review before continuing high-volume output.")

    if get("EnergyEfficiency") < THRESHOLDS["EnergyEfficiency_low"]:
        recommendations.append("Check machine energy efficiency and calibrate high-consumption equipment.")

    if get("AdditiveProcessTime") > THRESHOLDS["AdditiveProcessTime_high"]:
        recommendations.append("Tune additive process parameters to reduce cycle-time variation.")

    if get("AdditiveMaterialCost") > THRESHOLDS["AdditiveMaterialCost_high"]:
        recommendations.append("Inspect additive material handling and supplier certificate compliance.")

    # ✅ Fallback logic (cleaned)
    if not recommendations:
        if probability < 0.4:
            recommendations.append("Continue normal production and monitor standard SPC control charts.")
        else:
            recommendations.append("Review process settings and inspect the first five output units.")

    return recommendations