from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from src.ml.predictor import predict_batch, predict_one

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR),
        static_folder=str(FRONTEND_DIR),
        static_url_path="/static",
    )

    @app.get("/")
    def dashboard() -> str:
        return render_template("index.html")

    @app.get("/health")
    def health() -> tuple[Response, int]:
        return jsonify({"status": "ok", "service": "manufacturing-quality-platform"}), 200

    @app.get("/sample-data")
    def sample_data():
        return send_from_directory(PROJECT_ROOT / "data" / "raw", "sample_input.csv", as_attachment=True)

    @app.post("/predict")
    def predict() -> tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"detail": "Request body must be a JSON object."}), 400

        try:
            return jsonify(predict_one(payload)), 200
        except FileNotFoundError as exc:
            return jsonify({"detail": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

    @app.post("/predict/batch")
    def predict_batch_csv() -> tuple[Response, int] | Response:
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename.lower().endswith(".csv"):
            return jsonify({"detail": "Upload a CSV file."}), 400

        try:
            frame = pd.read_csv(uploaded)
            result = predict_batch(frame)
        except UnicodeDecodeError as exc:
            return jsonify({"detail": "CSV must be UTF-8 encoded."}), 400
        except FileNotFoundError as exc:
            return jsonify({"detail": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

        output = StringIO()
        result.to_csv(output, index=False)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=quality_predictions.csv"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
