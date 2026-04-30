from pathlib import Path

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# IMPORTANT: match your backend structure
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_PATH = MODEL_DIR / "quality_model.joblib"


# ─────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
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

# No categorical features for now
CATEGORICAL_FEATURES = []

# Final feature list used in model
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target column (must match dataset exactly)
TARGET_COLUMN = "DefectStatus"


# ─────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────
KAGGLE_DATASET = "rabieelkharoua/predicting-manufacturing-defects-dataset"

# Sample dataset path
SAMPLE_DATASET_PATH = RAW_DATA_DIR / "manufacturing_defects_sample.csv"
