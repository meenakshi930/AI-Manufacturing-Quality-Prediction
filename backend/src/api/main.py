from __future__ import annotations

from io import StringIO
from pathlib import Path
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory, Response
from flask_cors import CORS

# 🔥 ML logic
from src.ml.predictor import predict_one, predict_batch

# 🔥 validation
from src.utils.validation import validate_payload, ValidationError


# 📁 Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR),
        static_folder=str(FRONTEND_DIR),
        static_url_path="/static",
    )

    # ✅ Enable CORS
    CORS(app)

    # -------------------------------
    # 🔹 ROUTES
    # -------------------------------

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "manufacturing-quality-prediction"
        }), 200

    @app.get("/sample-data")
    def sample_data():
        file_path = DATA_DIR / "sample_input.csv"

        if not file_path.exists():
            return jsonify({"detail": "Sample data not found"}), 404

        return send_from_directory(file_path.parent, file_path.name, as_attachment=True)

    # -------------------------------
    # 🔥 SINGLE PREDICTION
    # -------------------------------
    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)

        # 🔥 FIX 1: handle empty JSON
        if not payload:
            return jsonify({"detail": "Empty request body"}), 400

        try:
            # 🔥 validation
            clean_data = validate_payload(payload)

            # 🔥 prediction
            result = predict_one(clean_data)

            return jsonify(result), 200

        except ValidationError as e:
            return jsonify({"detail": str(e)}), 400

        except FileNotFoundError as e:
            return jsonify({"detail": str(e)}), 503

        except ValueError as e:
            return jsonify({"detail": str(e)}), 400

        except Exception as e:
            return jsonify({"detail": f"Unexpected error: {str(e)}"}), 500

    # -------------------------------
    # 📊 BATCH PREDICTION
    # -------------------------------
    @app.post("/predict/batch")
    def predict_batch_csv():
        file = request.files.get("file")

        if not file or not file.filename.lower().endswith(".csv"):
            return jsonify({"detail": "Please upload a valid CSV file"}), 400

        try:
            # 🔥 FIX 2: safer CSV reading
            df = pd.read_csv(file, encoding="utf-8", errors="replace")

            result_df = predict_batch(df)

            output = StringIO()
            result_df.to_csv(output, index=False)

            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=predictions.csv"
                }
            )

        except FileNotFoundError as e:
            return jsonify({"detail": str(e)}), 503

        except ValueError as e:
            return jsonify({"detail": str(e)}), 400

        except Exception as e:
            return jsonify({"detail": f"Batch processing failed: {str(e)}"}), 500

    return app


# 🔥 Create app
app = create_app()


# -------------------------------
# 🚀 RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",   # ✅ required for Docker
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )