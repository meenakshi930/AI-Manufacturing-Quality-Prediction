"""
train_models.py
===============
Run this script to (re)train all models and save them to the models/ directory.

Usage:
    python train_models.py --data path/to/ai4i2020.csv
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Config ───────────────────────────────────────────────────────────────────

FEATURES = [
    "Type_enc",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_diff",
    "Power",
    "Tool_torque",
]

FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_TYPE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, le: LabelEncoder = None):
    df = df.copy()
    if le is None:
        le = LabelEncoder()
        df["Type_enc"] = le.fit_transform(df["Type"])
    else:
        df["Type_enc"] = le.transform(df["Type"])
    df["Temp_diff"]   = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power"]       = df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)
    df["Tool_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df, le


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, scaled=False, scaler=None):
    X_input = scaler.transform(X) if (scaled and scaler) else X
    y_pred  = model.predict(X_input)
    y_prob  = model.predict_proba(X_input)[:, 1]
    rep     = classification_report(y, y_pred, output_dict=True)
    cm      = confusion_matrix(y, y_pred).tolist()
    return {
        "roc_auc":   round(roc_auc_score(y, y_prob), 4),
        "accuracy":  round(rep["accuracy"], 4),
        "precision": round(rep["1"]["precision"], 4),
        "recall":    round(rep["1"]["recall"], 4),
        "f1":        round(rep["1"]["f1-score"], 4),
        "confusion_matrix": cm,
    }


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train(csv_path: str):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Failure rate: {df['Machine failure'].mean():.2%}")

    # Feature engineering
    df, le = engineer_features(df)
    X = df[FEATURES]
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\nTraining models ...")

    # ── Gradient Boosting (best model) ────────────────────────────────────────
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
    gb.fit(X_train, y_train)
    print("  [1/3] GradientBoosting trained")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  [2/3] RandomForest trained")

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    print("  [3/3] LogisticRegression trained")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = {
        "GradientBoosting":  evaluate(gb, X_test, y_test),
        "RandomForest":      evaluate(rf, X_test, y_test),
        "LogisticRegression": evaluate(lr, X_test_sc, y_test, scaled=True, scaler=scaler),
    }

    print("\nModel Metrics:")
    for name, m in metrics.items():
        print(f"  {name:20s}  ROC-AUC={m['roc_auc']}  F1={m['f1']}  Accuracy={m['accuracy']}")

    # ── Feature Importances ───────────────────────────────────────────────────
    feature_importances = {
        "GradientBoosting": dict(zip(FEATURES, gb.feature_importances_.tolist())),
        "RandomForest":     dict(zip(FEATURES, rf.feature_importances_.tolist())),
    }

    # ── Save Artifacts ────────────────────────────────────────────────────────
    def save_pkl(obj, name):
        path = os.path.join(MODELS_DIR, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {name}")

    save_pkl(gb,     "gb_model.pkl")
    save_pkl(rf,     "rf_model.pkl")
    save_pkl(lr,     "lr_model.pkl")
    save_pkl(scaler, "scaler.pkl")
    save_pkl(le,     "label_encoder.pkl")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "features":             FEATURES,
        "metrics":              metrics,
        "feature_importances":  feature_importances,
        "classes":              le.classes_.tolist(),
        "failure_types":        FAILURE_TYPES,
        "failure_type_names":   FAILURE_TYPE_NAMES,
        "dataset_stats": {
            "total_samples":        int(len(df)),
            "failure_rate":         round(float(df["Machine failure"].mean()), 4),
            "type_distribution":    df["Type"].value_counts().to_dict(),
            "failure_mode_counts":  {col: int(df[col].sum()) for col in FAILURE_TYPES},
        },
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")

    print("\nTraining complete! All files saved to ./models/")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train defect prediction models")
    parser.add_argument(
        "--data",
        default="data/ai4i2020.csv",
        help="Path to the AI4I 2020 CSV dataset",
    )
    args = parser.parse_args()
    train(args.data)
