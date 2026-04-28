import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE  # Fix: Handle imbalanced data
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/raw/ai4i2020.csv')
MODELS_DIR = os.path.join(BASE_DIR, '../models')
 
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df
 
def preprocess(df):
    features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    X = df[features]
    y = df['Machine failure']
    return X, y
 
def train():
    print("Loading data...")
    df = load_data()
    X, y = preprocess(df)
 
    print(f"Class distribution before SMOTE:\n{y.value_counts()}\n")
 
    # Split before SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    # Fix: Apply SMOTE only on training data to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}\n")
 
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
 
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # Extra safety for imbalance
    )
    model.fit(X_train_scaled, y_train_balanced)
 
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1        = round(f1_score(y_test, y_pred, zero_division=0), 4)
 
    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1 Score : {f1}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
 
    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    pickle.dump(model, open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb'))
 
    # Fix: Save metrics to JSON so app.py can load them dynamically
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "note": "Metrics computed on held-out test set (20% split)"
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
 
    print("\nModel, scaler, and metrics saved successfully.")
    return model, scaler, metrics
 
if __name__ == '__main__':
    train()