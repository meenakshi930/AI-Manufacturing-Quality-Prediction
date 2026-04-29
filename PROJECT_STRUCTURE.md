# Clean Folder Structure

```text
AI-Driven-Manufacturing-Quality-Prediction/
│
├── README.md                         # Project overview and instructions
├── PROJECT_STRUCTURE.md              # Folder structure documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files/folders to ignore in Git
│
├── backend/                          # Backend application (API + ML logic)
│   │
│   ├── src/                          # Source code
│   │   │
│   │   ├── api/                      # API layer (Flask/FastAPI routes)
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # Entry point for API (routes, endpoints)
│   │   │   └── schemas.py            # Request/response validation schemas
│   │   │
│   │   ├── ml/                       # Machine Learning logic
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # Model/config parameters
│   │   │   ├── data_generator.py     # Synthetic data generator (if used)
│   │   │   ├── preprocessing.py      # Data cleaning & feature engineering
│   │   │   ├── train_model.py        # Model training script
│   │   │   └── predictor.py          # Loads model and predicts output
│   │   │
│   │   ├── defect_prevention/        # Recommendation logic
│   │   │   ├── __init__.py
│   │   │   └── recommender.py        # Suggests preventive actions
│   │   │
│   │   └── __init__.py
│   │
│   ├── models/                       # Trained ML models
│   │   ├── quality_model.joblib      # Saved trained model
│   │   └── metrics.json              # Model performance metrics
│   │
│   └── tests/                        # Backend test cases
│       ├── test_api.py
│       └── test_recommender.py
│
├── frontend/                         # Frontend UI (static)
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── data/                             # Dataset (NOT used in deployment)
│   ├── raw/                          # Raw datasets
│   │   ├── sample_input.csv
│   │   └── manufacturing_defects_sample.csv
│   │
│   └── processed/                    # Cleaned/processed data
│       └── .gitkeep
│
├── scripts/                          # Utility & automation scripts
│   ├── download_kaggle_dataset.ps1   # Download dataset from Kaggle
│   ├── run_dev.ps1                   # Run development server
│   └── run_tests.ps1                 # Execute test cases
│
├── docs/                             # Documentation
│   └── architecture.md               # System architecture & flow

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
