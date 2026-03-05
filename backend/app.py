from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("backend/models/model.pkl")
scaler = joblib.load("backend/models/scaler.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["air_temperature"],
        data["process_temperature"],
        data["rotational_speed"],
        data["torque"],
        data["tool_wear"]
    ]

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    result = "Failure" if prediction == 1 else "No Failure"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)