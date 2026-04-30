# backend/src/utils/validation.py

from typing import Dict, Any

# Expected fields (must match your form + model features)
REQUIRED_FIELDS = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "DefectRate",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "InventoryTurnover",
    "StockoutRate",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
    "EnergyEfficiency",
    "AdditiveProcessTime",
    "AdditiveMaterialCost",
]

# Reasonable ranges (adjust if your dataset differs)
RANGES = {
    "ProductionVolume": (0, 10_000_000),
    "ProductionCost": (0, 1_000_000_000),
    "SupplierQuality": (0, 100),
    "DeliveryDelay": (0, 365),
    "DefectRate": (0, 100),
    "QualityScore": (0, 100),
    "MaintenanceHours": (0, 24 * 365),
    "DowntimePercentage": (0, 100),
    "InventoryTurnover": (0, 1000),
    "StockoutRate": (0, 100),
    "WorkerProductivity": (0, 100),
    "SafetyIncidents": (0, 10_000),
    "EnergyConsumption": (0, 10_000_000),
    "EnergyEfficiency": (0, 1),  # assuming ratio
    "AdditiveProcessTime": (0, 10_000),
    "AdditiveMaterialCost": (0, 10_000_000),
}


class ValidationError(ValueError):
    """Raised when input payload fails validation."""
    pass


def _to_number(value: Any, field: str) -> float:
    """Convert to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Field '{field}' must be a number.")


def validate_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Validate and sanitize incoming JSON for /predict.

    Returns:
        cleaned (Dict[str, float]): numeric, range-checked payload

    Raises:
        ValidationError: with clear message for client
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    cleaned: Dict[str, float] = {}

    # 1) Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValidationError(f"Missing required fields: {', '.join(missing)}")

    # 2) Type conversion + range checks
    for field in REQUIRED_FIELDS:
        value = payload.get(field)

        num = _to_number(value, field)

        # Optional: forbid NaN/inf
        if num != num or num in (float("inf"), float("-inf")):
            raise ValidationError(f"Field '{field}' contains invalid number.")

        # Range check if defined
        if field in RANGES:
            lo, hi = RANGES[field]
            if not (lo <= num <= hi):
                raise ValidationError(
                    f"Field '{field}' must be between {lo} and {hi}."
                )

        cleaned[field] = num

    # 3) (Optional) reject unknown fields to keep schema strict
    unknown = [k for k in payload.keys() if k not in REQUIRED_FIELDS]
    if unknown:
        # You can switch to "ignore" instead of error if you prefer
        raise ValidationError(f"Unknown fields: {', '.join(unknown)}")

    return cleaned