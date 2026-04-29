from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.config import SAMPLE_DATASET_PATH


def generate_manufacturing_defects_data(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    production_volume = rng.integers(100, 1001, rows)
    production_cost = rng.uniform(5000, 20000, rows)
    supplier_quality = rng.uniform(80, 100, rows)
    delivery_delay = rng.integers(0, 6, rows)
    defect_rate = rng.uniform(0.5, 5.0, rows)
    quality_score = rng.uniform(60, 100, rows)
    maintenance_hours = rng.integers(0, 25, rows)
    downtime_percentage = rng.uniform(0, 5, rows)
    inventory_turnover = rng.uniform(2, 10, rows)
    stockout_rate = rng.uniform(0, 10, rows)
    worker_productivity = rng.uniform(80, 100, rows)
    safety_incidents = rng.integers(0, 11, rows)
    energy_consumption = rng.uniform(1000, 5000, rows)
    energy_efficiency = rng.uniform(0.1, 0.5, rows)
    additive_process_time = rng.uniform(1, 10, rows)
    additive_material_cost = rng.uniform(100, 500, rows)

    risk_score = (
        0.002 * (production_volume - 600)
        + 0.00008 * (production_cost - 12000)
        - 0.11 * (supplier_quality - 90)
        + 0.34 * delivery_delay
        + 0.72 * (defect_rate - 2.4)
        - 0.09 * (quality_score - 78)
        - 0.045 * (maintenance_hours - 10)
        + 0.55 * (downtime_percentage - 2)
        - 0.05 * (inventory_turnover - 5)
        + 0.18 * (stockout_rate - 4)
        - 0.09 * (worker_productivity - 90)
        + 0.16 * safety_incidents
        + 0.00018 * (energy_consumption - 3000)
        - 2.0 * (energy_efficiency - 0.3)
        + 0.13 * (additive_process_time - 5)
        + 0.003 * (additive_material_cost - 280)
        + rng.normal(0, 0.75, rows)
    )

    probability = 1 / (1 + np.exp(-risk_score))
    defect_status = (probability > 0.5).astype(int)

    return pd.DataFrame(
        {
            "ProductionVolume": production_volume,
            "ProductionCost": production_cost.round(2),
            "SupplierQuality": supplier_quality.round(2),
            "DeliveryDelay": delivery_delay,
            "DefectRate": defect_rate.round(2),
            "QualityScore": quality_score.round(2),
            "MaintenanceHours": maintenance_hours,
            "DowntimePercentage": downtime_percentage.round(2),
            "InventoryTurnover": inventory_turnover.round(2),
            "StockoutRate": stockout_rate.round(2),
            "WorkerProductivity": worker_productivity.round(2),
            "SafetyIncidents": safety_incidents,
            "EnergyConsumption": energy_consumption.round(2),
            "EnergyEfficiency": energy_efficiency.round(3),
            "AdditiveProcessTime": additive_process_time.round(2),
            "AdditiveMaterialCost": additive_material_cost.round(2),
            "DefectStatus": defect_status,
        }
    )


def save_sample_dataset(rows: int = 1200) -> str:
    SAMPLE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_manufacturing_defects_data(rows=rows).to_csv(SAMPLE_DATASET_PATH, index=False)
    return str(SAMPLE_DATASET_PATH)


if __name__ == "__main__":
    path = save_sample_dataset()
    print(f"Sample dataset saved to {path}")
