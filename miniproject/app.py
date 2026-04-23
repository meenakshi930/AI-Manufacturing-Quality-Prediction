"""
AutoQuality Predictor— AI Manufacturing Defect Prediction API
=====================================================
Flask REST API · AI4I 2020 Predictive Maintenance Dataset
Models: Gradient Boosting · Random Forest · Logistic Regression
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
app        = Flask(__name__, static_folder="static", static_url_path="/static")

def _load(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)

log.info("Loading models …")
GB_MODEL = _load("gb_model.pkl")
RF_MODEL = _load("rf_model.pkl")
LR_MODEL = _load("lr_model.pkl")
SCALER   = _load("scaler.pkl")
LE       = _load("label_encoder.pkl")
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    META = json.load(f)

MODELS   = {"GradientBoosting": GB_MODEL, "RandomForest": RF_MODEL, "LogisticRegression": LR_MODEL}
FEATURES = META["features"]
_req_count = {"total": 0, "failures_predicted": 0}
_start_time = time.time()
log.info("All models loaded OK")

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.before_request
def preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        return cors(make_response("", 204))

FEATURE_RANGES = {
    "type":                   ("categorical", ["L","M","H"]),
    "air_temperature_k":      ("range", (295.3, 304.5)),
    "process_temperature_k":  ("range", (305.7, 313.8)),
    "rotational_speed_rpm":   ("range", (1168, 2886)),
    "torque_nm":              ("range", (3.8, 76.6)),
    "tool_wear_min":          ("range", (0, 253)),
}

def validate(data):
    errors = []
    for field, (kind, bounds) in FEATURE_RANGES.items():
        if field not in data:
            errors.append(f"Missing field: '{field}'"); continue
        if kind == "categorical":
            if data[field] not in bounds:
                errors.append(f"'{field}' must be one of {bounds}")
        else:
            try: v = float(data[field])
            except: errors.append(f"'{field}' must be numeric"); continue
            lo, hi = bounds
            if not (lo <= v <= hi):
                errors.append(f"'{field}'={v} out of range [{lo},{hi}]")
    return errors

def engineer(data):
    type_enc  = int(LE.transform([data["type"]])[0])
    air       = float(data["air_temperature_k"])
    proc      = float(data["process_temperature_k"])
    rpm       = float(data["rotational_speed_rpm"])
    torque    = float(data["torque_nm"])
    wear      = float(data["tool_wear_min"])
    temp_diff = proc - air
    power     = rpm * torque * (2 * np.pi / 60)
    tt        = wear * torque
    return pd.DataFrame([[type_enc, air, proc, rpm, torque, wear, temp_diff, power, tt]], columns=FEATURES)

def infer(model_name, df):
    model = MODELS[model_name]
    X     = SCALER.transform(df) if model_name == "LogisticRegression" else df
    prob  = float(model.predict_proba(X)[0][1])
    pred  = int(model.predict(X)[0])
    risk  = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.3 else "LOW")
    return {"prediction": pred, "failure_predicted": bool(pred), "failure_probability": round(prob, 6), "risk_level": risk}

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api")
def api_info():
    return jsonify({
        "name":"DefectSense Prediction API","version":"2.0.0","status":"online",
        "uptime_seconds": round(time.time()-_start_time,1),
        "requests_served": _req_count["total"],
        "failures_predicted": _req_count["failures_predicted"],
        "models": list(MODELS.keys()), "best_model":"GradientBoosting"
    })

@app.route("/api/health")
def health():
    return jsonify({
        "status":"ok","timestamp":datetime.utcnow().isoformat()+"Z",
        "models_loaded":list(MODELS.keys()),
        "uptime_seconds":round(time.time()-_start_time,1),
        "requests_served":_req_count["total"]
    })

@app.route("/api/models")
def get_models():
    return jsonify({"models":list(MODELS.keys()),"best_model":"GradientBoosting","metrics":META["metrics"],"feature_importances":META["feature_importances"]})

@app.route("/api/dataset/stats")
def dataset_stats():
    return jsonify({"dataset_stats":META["dataset_stats"],"failure_type_names":META["failure_type_names"],"product_types":{"L":"Low quality — 60% of data","M":"Medium quality — 30% of data","H":"High quality — 10% of data"}})

@app.route("/api/features")
def features():
    return jsonify({
        "input_features":{
            "type":{"type":"string","values":["L","M","H"],"description":"Product quality variant"},
            "air_temperature_k":{"type":"float","range":[295.3,304.5],"unit":"Kelvin"},
            "process_temperature_k":{"type":"float","range":[305.7,313.8],"unit":"Kelvin"},
            "rotational_speed_rpm":{"type":"int","range":[1168,2886],"unit":"RPM"},
            "torque_nm":{"type":"float","range":[3.8,76.6],"unit":"Nm"},
            "tool_wear_min":{"type":"int","range":[0,253],"unit":"minutes"}
        },
        "engineered_features":{"Temp_diff":"Process temp − Air temp [K]","Power":"RPM × Torque × (2π/60) [W]","Tool_torque":"Tool wear × Torque [Nm·min]"},
        "all_model_features":FEATURES
    })

@app.route("/api/predict", methods=["POST"])
@app.route("/api/predict/<model_name>", methods=["POST"])
def predict(model_name="GradientBoosting"):
    if model_name not in MODELS:
        return jsonify({"error":f"Unknown model '{model_name}'","available":list(MODELS.keys())}), 404
    data = request.get_json(force=True, silent=True)
    if not data: return jsonify({"error":"Invalid JSON body"}), 400
    errs = validate(data)
    if errs: return jsonify({"error":"Validation failed","details":errs}), 422
    try:
        result = infer(model_name, engineer(data))
        _req_count["total"] += 1
        if result["failure_predicted"]: _req_count["failures_predicted"] += 1
        log.info("PREDICT model=%s prob=%.4f risk=%s", model_name, result["failure_probability"], result["risk_level"])
        return jsonify({"model":model_name,"input":data,**result})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(force=True, silent=True)
    if not body or "data" not in body: return jsonify({"error":"'data' key required"}), 400
    model_name = body.get("model","GradientBoosting")
    if model_name not in MODELS: return jsonify({"error":f"Unknown model '{model_name}'"}), 404
    records = body["data"]
    if not isinstance(records, list) or not records: return jsonify({"error":"'data' must be a non-empty list"}), 400
    if len(records) > 500: return jsonify({"error":"Batch limit is 500"}), 400
    results = []
    for i, rec in enumerate(records):
        errs = validate(rec)
        if errs: results.append({"index":i,"error":errs})
        else:
            try:
                res = infer(model_name, engineer(rec))
                results.append({"index":i,**res})
                _req_count["total"] += 1
                if res["failure_predicted"]: _req_count["failures_predicted"] += 1
            except Exception as e:
                results.append({"index":i,"error":str(e)})
    failures = [r for r in results if r.get("failure_predicted")]
    log.info("BATCH model=%s total=%d failures=%d", model_name, len(records), len(failures))
    return jsonify({"model":model_name,"total":len(records),"failures_detected":len(failures),"failure_rate":round(len(failures)/len(records),6),"results":results})

@app.route("/api/predict/compare", methods=["POST"])
def predict_compare():
    data = request.get_json(force=True, silent=True)
    if not data: return jsonify({"error":"Invalid JSON body"}), 400
    errs = validate(data)
    if errs: return jsonify({"error":"Validation failed","details":errs}), 422
    try:
        df      = engineer(data)
        results = {name: infer(name, df) for name in MODELS}
        avg_p   = round(float(np.mean([v["failure_probability"] for v in results.values()])), 6)
        risk    = "HIGH" if avg_p >= 0.6 else ("MEDIUM" if avg_p >= 0.3 else "LOW")
        _req_count["total"] += len(MODELS)
        return jsonify({"input":data,"model_predictions":results,"ensemble":{"avg_failure_probability":avg_p,"failure_predicted":bool(avg_p>=0.5),"risk_level":risk}})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.errorhandler(404)
def not_found(e): return jsonify({"error":"Not found"}), 404
@app.errorhandler(405)
def method_not_allowed(e): return jsonify({"error":"Method not allowed"}), 405

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🏭  DefectSense — Defect Prediction System")
    print("  🌐  http://127.0.0.1:5000  (serves frontend too)")
    print("  📡  API → http://127.0.0.1:5000/api")
    print("═"*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
