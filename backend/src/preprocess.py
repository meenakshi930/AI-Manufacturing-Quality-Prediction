import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Select required features
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    target = "Machine failure"

    X = df[features]
    y = df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler



if __name__ == "__main__":
    file_path = "backend/data/raw/ai4i2020.csv"
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)