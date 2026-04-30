AI-Manufacturing-Quality-Prediction/
│
├── README.md                         # Project overview + setup + Docker usage
├── PROJECT_STRUCTURE.md              # Folder structure documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignore files for Git
├── pytest.ini                        # Pytest configuration (PYTHONPATH fix)
│
├── Dockerfile                        # Docker image configuration
├── docker-compose.yml                # Multi-container setup
│
├── .github/
│   └── workflows/
│       └── test.yml                  # CI/CD pipeline (GitHub Actions)
│
├── backend/                          # Backend (Flask + ML)
│   │
│   ├── src/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # Flask app (routes + endpoints)
│   │   │   └── schemas.py            # Request validation schemas
│   │   │
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── data_generator.py
│   │   │   ├── preprocessing.py
│   │   │   ├── train_model.py
│   │   │   └── predictor.py          # ML prediction logic
│   │   │
│   │   ├── defect_prevention/
│   │   │   ├── __init__.py
│   │   │   └── recommender.py        # Recommendation system
│   │   │
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── validation.py         # Input validation logic
│   │   │
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── quality_model.joblib      # Trained ML model
│   │   └── metrics.json              # Model evaluation metrics
│   │
│   └── tests/
│       ├── test_api.py
│       ├── test_predictor.py
│       └── test_recommender.py
│
├── frontend/                         # UI (connected to API)
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── data/
│   ├── raw/
│   │   ├── sample_input.csv
│   │   └── manufacturing_defects_sample.csv
│   │
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   ├── download_kaggle_dataset.ps1
│   ├── run_dev.ps1
│   └── run_tests.ps1
│
└── docs/
    └── architecture.md

## Run Order

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\scripts\download_kaggle_dataset.ps1
python -m src.ml.train_model
flask --app src.api.main run --debug
```

Dashboard URL:

```text
http://127.0.0.1:5000
```
