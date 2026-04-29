# AI-Driven Manufacturing Quality Prediction & Defect Prevention Platform

This project predicts manufacturing quality risk before a product leaves the production line. It combines a Flask web app, machine-learning based defect prediction, preventive recommendations, batch CSV scoring, and a clean dashboard for plant supervisors.

## Features

- Predicts whether a production record is likely to have high defects.
- Estimates defect probability and quality risk level.
- Gives prevention recommendations based on supplier, production, quality, maintenance, inventory, workforce, energy, and additive manufacturing metrics.
- Supports single prediction and batch CSV prediction.
- Uses the Kaggle "Predicting Manufacturing Defects Dataset" schema.
- Includes a compatible fallback dataset so the project can run without Kaggle credentials.
- Provides tests for API health and prevention rules.

## Tech Stack

- Python 3.10+
- Flask
- scikit-learn
- pandas
- numpy
- joblib
- HTML, CSS, JavaScript

## Kaggle Dataset

Recommended dataset: [Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)

Configure your Kaggle API token, then download the dataset:

```powershell
.\scripts\download_kaggle_dataset.ps1
```

The training script automatically uses any CSV in `data/raw` that contains the required Kaggle columns and the `DefectStatus` target. If no matching CSV is present, it creates `data/raw/manufacturing_defects_sample.csv` with the same schema.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.ml.train_model
flask --app src.api.main run --debug
```

Open the dashboard:

```text
http://127.0.0.1:5000
```

## Project Modules

- `src/ml`: data generation, validation, model training, prediction service
- `src/api`: Flask app and API routes
- `src/defect_prevention`: rule-based prevention recommendations
- `frontend`: static dashboard served by Flask
- `data`: sample input and Kaggle/raw datasets
- `models`: trained model and metrics
- `tests`: automated test cases

## Input Parameters

| Field | Description | Example |
| --- | --- | --- |
| `ProductionVolume` | Units produced per day | `920` |
| `ProductionCost` | Daily production cost | `18400` |
| `SupplierQuality` | Supplier quality rating | `84.2` |
| `DeliveryDelay` | Average supplier delay in days | `4` |
| `DefectRate` | Defects per thousand units | `4.3` |
| `QualityScore` | Overall quality score | `69.5` |
| `MaintenanceHours` | Weekly maintenance hours | `3` |
| `DowntimePercentage` | Production downtime percentage | `4.1` |
| `InventoryTurnover` | Inventory turnover ratio | `3.4` |
| `StockoutRate` | Stockout rate percentage | `8.2` |
| `WorkerProductivity` | Workforce productivity percentage | `83.4` |
| `SafetyIncidents` | Monthly safety incidents | `5` |
| `EnergyConsumption` | Energy consumed in kWh | `4300` |
| `EnergyEfficiency` | Energy usage efficiency factor | `0.18` |
| `AdditiveProcessTime` | Additive manufacturing time | `8.1` |
| `AdditiveMaterialCost` | Additive material cost per unit | `420` |

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dashboard |
| `GET` | `/health` | API health check |
| `GET` | `/sample-data` | Download sample CSV |
| `POST` | `/predict` | Predict one manufacturing record |
| `POST` | `/predict/batch` | Predict a CSV batch |

## Example Prediction Request

```json
{
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
  "AdditiveMaterialCost": 420
}
```

## Testing

```powershell
pytest
```

## Notes

The included fallback dataset is synthetic and designed for academic/demo use. Use the Kaggle dataset or real plant data before production deployment.
