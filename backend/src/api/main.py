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

# ── All imports use the same absolute style: backend.src.* ───────────────────
from backend.src.utils.validation        import validate_input
from backend.src.ml.predictor            import predict_one
from backend.src.ml.train_model          import train_and_save
from backend.src.defect_prevention.recommender import get_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Tighten origins in production: CORS(app, origins=os.environ["CORS_ORIGINS"].split(","))
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=_cors_origins)

# ── Ensure the model is ready before accepting traffic ────────────────────────
train_and_save()   # no-op if quality_model.joblib already exists


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Prediction ────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = validate_input(data)
    if errors:
        return jsonify({"error": errors}), 422

    try:
        result = predict_one(data)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Internal server error."}), 500


# ── Recommendations ───────────────────────────────────────────────────────────
@app.route("/recommendations", methods=["POST"])
def recommendations():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        recs = get_recommendations(data)
        return jsonify({"recommendations": recs}), 200
    except Exception:
        logger.exception("Error generating recommendations")
        return jsonify({"error": "Internal server error."}), 500


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def metrics():
    # backend/models/metrics.json
    metrics_path = Path(__file__).resolve().parents[2] / "models" / "metrics.json"
    if not metrics_path.exists():
        return jsonify({"error": "Metrics not available yet."}), 404
    return jsonify(json.loads(metrics_path.read_text())), 200


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
