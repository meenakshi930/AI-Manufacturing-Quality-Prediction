AI-Driven Manufacturing Quality Prediction & Defect Prevention Platform

This project predicts manufacturing defect risk before products leave the production line.
It combines a Flask backend, machine learning model, rule-based recommendations, batch processing, and a frontend dashboard for plant supervisors.

---

🚀 Features

- Predict defect risk (Low / Medium / High)
- Output probability score
- Preventive recommendations based on production conditions
- Supports:
  - Single prediction ("/predict")
  - Batch CSV prediction ("/predict/batch")
- Integrated frontend dashboard (connected to API)
- Input validation & error handling
- Automated testing with pytest
- CI/CD using GitHub Actions
- Docker containerization for deployment

---

🧠 Tech Stack

- Python 3.10
- Flask
- scikit-learn
- pandas, numpy
- joblib
- HTML, CSS, JavaScript
- Docker
- GitHub Actions (CI/CD)

---

📁 Project Structure

AI-Manufacturing-Quality-Prediction/
│
├── backend/
│   ├── src/
│   │   ├── api/
│   │   ├── ml/
│   │   ├── defect_prevention/
│   │   └── utils/
│   ├── models/
│   └── tests/
│
├── frontend/
├── data/
├── scripts/
├── docs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md

---

⚙️ Setup & Run (Local)

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

---

🧪 Train Model

python -m src.ml.train_model

---

▶️ Run Application

flask --app src.api.main run --debug

Open:

http://localhost:5000

---

🐳 Run with Docker (Recommended)

docker compose up --build

Open:

http://localhost:5000

---

🌐 API Endpoints

Method| Endpoint| Description
GET| "/"| Dashboard UI
GET| "/health"| Health check
GET| "/sample-data"| Download sample CSV
POST| "/predict"| Single prediction
POST| "/predict/batch"| Batch CSV prediction

---

📥 Example Input

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

---

🧪 Testing

pytest

---

⚙️ CI/CD

- Automated testing using GitHub Actions
- Runs tests on every push and pull request

---

📊 Dataset

Recommended dataset:
https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset

Fallback:

- Synthetic dataset generated automatically if not available

---

⚠️ Notes

- Train model before running predictions:

python -m src.ml.train_model

- Model file is not committed to Git
- Synthetic data is for testing only

---

🚀 Deployment

Supports deployment using:

- Docker
- Render / cloud platforms

---

📌 Future Improvements

- Authentication & authorization
- Rate limiting
- Database integration
- Model monitoring
- Real-time factory data integration

---

👩‍💻 Contributors

- Meenakshi Gupta — Backend, ML, API, Docker, CI/CD
- Anshika Garg — Frontend, UI Design, Integration

---
