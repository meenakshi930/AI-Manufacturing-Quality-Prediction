# AI-Driven Manufacturing Quality Prediction & Defect Prevention Platform

This project predicts manufacturing defect risk **before products leave the production line**.
It combines a Flask backend, machine learning model, rule-based recommendations, batch processing, and a simple dashboard for plant supervisors.

---

## 🚀 Features

* Predicts whether a production record is likely defective
* Outputs defect probability and risk level (Low / Medium / High)
* Provides preventive recommendations based on process conditions
* Supports:

  * Single prediction (`/predict`)
  * Batch CSV prediction (`/predict/batch`)
* Works with Kaggle dataset schema
* Includes fallback synthetic dataset (no Kaggle required)
* Robust batch handling (row-level error recovery)
* Clean frontend dashboard

---

## 🧠 Tech Stack

* Python 3.10+
* Flask
* scikit-learn
* pandas, numpy
* joblib
* HTML, CSS, JavaScript

---

## 📁 Project Structure (Simplified)

```
project-root/
│
├── backend/
│   ├── src/
│   ├── models/
│   └── tests/
│
├── frontend/
├── data/
│   ├── raw/
│   └── processed/   # reserved for future use
│
├── scripts/
├── docs/
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

Recommended dataset:
https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset

### Download via script

```
.\scripts\download_kaggle_dataset.ps1
```

### Fallback behavior

If no dataset is found:

* A synthetic dataset is generated in `data/raw/`
* Schema matches Kaggle dataset

---

## ⚙️ Setup & Run

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🧪 Train Model

```
python -m src.ml.train_model
```

---

## ▶️ Run Backend

```
flask --app src.api.main run --debug
```

Open:

```
http://127.0.0.1:5000
```

---

## 🌐 API Endpoints

| Method | Endpoint         | Description          |
| ------ | ---------------- | -------------------- |
| GET    | `/`              | Dashboard            |
| GET    | `/health`        | Health check         |
| GET    | `/sample-data`   | Download sample CSV  |
| POST   | `/predict`       | Single prediction    |
| POST   | `/predict/batch` | Batch CSV prediction |

---

## 📥 Example Input

```
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

---

## 🧪 Testing

```
pytest
```

---

## ⚠️ Important Notes

* Synthetic dataset is for **demo/testing only**
* Use real manufacturing data for production
* Trained model is **not stored in Git**
* Train model before running predictions:

```
python -m src.ml.train_model
```

---

## 🚀 Deployment (Render)

* Root Directory: `backend`
* Build Command:

```
pip install -r requirements.txt
```

* Start Command:

```
python -m src.ml.train_model && python -m src.api.main
```

---

## 📌 Future Improvements

* Real-time data integration (PLC / MES / SCADA)
* Model monitoring & drift detection
* Database integration (PostgreSQL)
* Authentication & role-based access
* Cloud model storage

---

## 👩‍💻 Contributors

* Meenakshi Gupta — Backend, ML Model, API Development
* Anshika Garg — Frontend, UI Design, Integration
