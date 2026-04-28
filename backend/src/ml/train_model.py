from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.config import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RAW_DATA_DIR,
    SAMPLE_DATASET_PATH,
    TARGET_COLUMN,
)
from src.ml.data_generator import save_sample_dataset
from src.ml.preprocessing import validate_input_frame


def load_training_data() -> pd.DataFrame:
    candidates = sorted(RAW_DATA_DIR.glob("*.csv"))
    for dataset_path in candidates:
        data = pd.read_csv(dataset_path)
        if TARGET_COLUMN in data.columns and all(column in data.columns for column in FEATURE_COLUMNS):
            return data

    if not SAMPLE_DATASET_PATH.exists():
        save_sample_dataset()
    return pd.read_csv(SAMPLE_DATASET_PATH)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train() -> dict[str, float | str]:
    data = load_training_data()
    features = validate_input_frame(data[FEATURE_COLUMNS])
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "report": classification_report(y_test, predictions),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
