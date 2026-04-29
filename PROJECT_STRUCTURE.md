# Clean Folder Structure

```text
AI-Driven-Manufacturing-Quality-Prediction/
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- raw/
|   |   |-- sample_input.csv
|   |   `-- manufacturing_defects_sample.csv
|   `-- processed/
|       `-- .gitkeep
|-- docs/
|   `-- architecture.md
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|-- models/
|   |-- quality_model.joblib
|   `-- metrics.json
|-- scripts/
|   |-- download_kaggle_dataset.ps1
|   |-- run_dev.ps1
|   `-- run_tests.ps1
|-- src/
|   |-- __init__.py
|   |-- api/
|   |   |-- __init__.py
|   |   |-- main.py
|   |   `-- schemas.py
|   |-- defect_prevention/
|   |   |-- __init__.py
|   |   `-- recommender.py
|   `-- ml/
|       |-- __init__.py
|       |-- config.py
|       |-- data_generator.py
|       |-- predictor.py
|       |-- preprocessing.py
|       `-- train_model.py
`-- tests/
    |-- test_api.py
    `-- test_recommender.py
```

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
