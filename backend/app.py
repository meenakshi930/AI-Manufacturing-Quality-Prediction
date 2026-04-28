from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import json
import os

# Template folder fix
app = Flask(__name__, template_folder="../templates")

CORS(app)

# Rate limiting
limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour"])

# Load model and scaler
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load metrics
METRICS_PATH = "models/metrics.json"
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        MODEL_METRICS = json.load(f)
else:
    MODEL_METRICS = {
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.96,
        "f1_score": 0.965,
        "note": "Default metrics"
    }

REQUIRED_FIELDS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@limiter.limit("30 per minute")
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        air_temperature = float(data["air_temperature"])
        process_temperature = float(data["process_temperature"])
        rotational_speed = float(data["rotational_speed"])
        torque = float(data["torque"])
        tool_wear = float(data["tool_wear"])
    except:
        return jsonify({"error": "All fields must be numeric"}), 400

    if not (250 <= air_temperature <= 400):
        return jsonify({"error": "air_temperature must be between 250-400"}), 400
    if rotational_speed <= 0:
        return jsonify({"error": "rotational_speed must be positive"}), 400
    if tool_wear < 0:
        return jsonify({"error": "tool_wear cannot be negative"}), 400

    features = np.array([[air_temperature, process_temperature, rotational_speed, torque, tool_wear]])
    features = np.array([[air_temperature, process_temperature, rotational_speed, torque, tool_wear]])
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    result = "Failure" if prediction == 1 else "No Failure"
    confidence = round(float(max(probability)) * 100, 2)

    return jsonify({
        "prediction": result,
        "confidence_percent": confidence,
        "model_metrics": MODEL_METRICS
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(MODEL_METRICS)

if __name__ == "__main__":
    app.run(debug=True)