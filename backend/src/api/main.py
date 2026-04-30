"""
Flask application — routes and endpoints.
Location: backend/src/api/main.py

Run from repo root:
    python -m backend.src.api.main
"""

import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Rate Limiting ─────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per minute"],
    storage_uri="memory://",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Imports (AFTER app init)
# ─────────────────────────────────────────────
from backend.src.ml.predictor import predict_one, predict_batch
from backend.src.utils.validation import validate_payload, ValidationError
from backend.src.defect_prevention.recommender import prevention_recommendations


@app.route("/")
def home():
    return jsonify({
        "message": "AI Manufacturing Quality Prediction API is running 🚀"
    })


# ── Health ───────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Prediction ───────────────────────────────
@app.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        validate_payload(data)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 422

    try:
        result = predict_one(data)
        return jsonify(result), 200

    except Exception:
        logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal server error."}), 500


# ── Batch Prediction ─────────────────────────
@app.route("/predict-batch", methods=["POST"])
@limiter.limit("20 per minute")
def predict_batch_api():
    import pandas as pd

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)
        results = predict_batch(df)
        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": str(e)}), 500


# ── Recommendations ──────────────────────────
@app.route("/recommendations", methods=["POST"])
def recommendations():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        recs = prevention_recommendations(data, 0.5)
        return jsonify({"recommendations": recs}), 200

    except Exception:
        logger.exception("Error generating recommendations")
        return jsonify({"error": "Internal server error."}), 500


# ── Metrics ──────────────────────────────────
@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_path = Path(__file__).resolve().parents[2] / "models" / "metrics.json"

    if not metrics_path.exists():
        return jsonify({"error": "Metrics not available yet."}), 404

    return jsonify(json.loads(metrics_path.read_text())), 200


# ── Run Server ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug)
