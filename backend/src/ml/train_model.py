"""
Train and persist the quality prediction model.
Run: python -m backend.src.ml.train_model
"""

import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backend.src.ml.config import FEATURE_COLUMNS, TARGET_COLUMN

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]   # backend/
DATA_PATH = _BASE_DIR / "data" / "raw" / "manufacturing_defects_sample.csv"
MODEL_PATH = _BASE_DIR / "models" / "model.pkl"
METRICS_PATH = _BASE_DIR / "models" / "metrics.json"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Load real data ────────────────────────────────────────────────────────────
def _load_data():
    df = pd.read_csv(DATA_PATH)

    # Ensure required columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_save(force: bool = False) -> dict:
    if MODEL_PATH.exists() and not force:
        print(f"[train_model] Model already exists at {MODEL_PATH}")
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                return json.load(f)
        return {}

    print("[train_model] Loading real dataset...")
    X, y = _load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("[train_model] Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Model saved at: {MODEL_PATH}")
    print(f"✅ Metrics saved at: {METRICS_PATH}")
    print(f"📊 Metrics: {metrics}")

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_and_save(force=args.force)
