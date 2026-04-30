"""
Train and persist the quality prediction model.
Run directly:  python -m src.ml.train_model
Or imported:   from src.ml.train_model import train_and_save
"""

import json
import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
# Works whether you run from repo root or from backend/
_HERE = Path(__file__).resolve().parent          # backend/src/ml/
_MODELS_DIR = _HERE.parents[1] / "models"        # backend/models/
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = _MODELS_DIR / "quality_model.joblib"
METRICS_PATH = _MODELS_DIR / "metrics.json"


# ── Synthetic data (replace with your real data loader) ──────────────────────
def _load_data():
    """
    Replace this with your real dataset loading logic.
    Returns X (np.ndarray) and y (np.ndarray of 0/1).
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(loc=[50, 200, 0.5, 100], scale=[5, 20, 0.05, 10], size=(n, 4))
    # Simple rule: defect when any feature exceeds +1.5 std
    y = (np.abs(X - X.mean(axis=0)) > 1.5 * X.std(axis=0)).any(axis=1).astype(int)
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_save(force: bool = False) -> dict:
    """
    Train the model and save it to disk.

    Args:
        force: Re-train even if a model file already exists.

    Returns:
        dict of evaluation metrics.
    """
    if MODEL_PATH.exists() and not force:
        print(f"[train_model] Model already exists at {MODEL_PATH}. "
              "Pass force=True to retrain.")
        with open(METRICS_PATH) as f:
            return json.load(f)

    print("[train_model] Loading data …")
    X, y = _load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("[train_model] Training …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_model] Saved model  → {MODEL_PATH}")
    print(f"[train_model] Saved metrics→ {METRICS_PATH}")
    print(f"[train_model] Metrics: {metrics}")
    return metrics


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the quality prediction model.")
    parser.add_argument("--force", action="store_true",
                        help="Re-train even if model file already exists.")
    args = parser.parse_args()
    train_and_save(force=args.force)
