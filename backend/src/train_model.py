import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess_data


def train_model():
    file_path = "backend/data/raw/ai4i2020.csv"

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model & scaler
    joblib.dump(model, "backend/models/model.pkl")
    joblib.dump(scaler, "backend/models/scaler.pkl")

    print("\nModel and scaler saved successfully!")


if __name__ == "__main__":
    train_model()