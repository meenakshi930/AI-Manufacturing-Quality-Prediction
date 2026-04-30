# 🏭 AI-Driven Manufacturing Quality Prediction & Defect Prevention Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.1-orange?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-teal?style=flat-square)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-purple?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-87.08%25-success?style=flat-square)

> Predicts manufacturing defect risk **before products leave the production line** — combining a Flask backend, Random Forest ML model, rule-based recommendations, batch processing, and a supervisor dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Defect risk prediction | Low / Medium / High with probability score |
| 💡 Preventive recommendations | Rule-based actions for risky process conditions |
| 📂 Batch CSV prediction | Upload a file, get predictions for every row |
| 🖥️ Supervisor dashboard | Browser UI — no ML expertise required |
| ✅ Input validation | Descriptive HTTP 422 error responses |
| ⚙️ CI/CD pipeline | pytest runs on every push via GitHub Actions |

---

## 🧠 Tech Stack

`Python 3.10` · `Flask` · `scikit-learn` · `pandas` · `NumPy` · `joblib` · `Docker` · `GitHub Actions` · `pytest` · `HTML/CSS/JS`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Health check |
| `GET` | `/sample-data` | Download sample CSV |
| `POST` | `/predict` | Single-record prediction |
| `POST` | `/predict/batch` | Batch CSV prediction |

---

## ⚙️ Setup & Run

```bash
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python -m src.ml.train_model

# 4. Run the app
flask --app src.api.main run --debug
```

🐳 **Docker (recommended)**
```bash
docker compose up --build
```

Open **http://localhost:5000**

---

## 📌 Future Improvements

- Authentication & authorization
- Database integration (PostgreSQL)
- Model drift monitoring & scheduled retraining
- Real-time PLC/SCADA factory data integration
- Alerting for high-risk batches (Slack, email)

---

## 👩‍💻 Contributors

| | Name | Responsibilities |
|---|---|---|
| 🔵 | **Meenakshi Gupta** | Backend, ML, API, Docker, CI/CD |
| 🟢 | **Anshika Garg** | Frontend, UI Design, Integration |
