"""
Tests for src.ml.predictor

Run:  pytest backend/tests/test_predictor.py -v
"""

import pytest
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────
VALID_INPUT = {
    "temperature":     72.5,
    "pressure":       210.0,
    "humidity":         0.48,
    "vibration_level": 98.3,
}


# ── Unit test: mock the pipeline so no model file is needed ──────────────────
class _FakePipeline:
    """Minimal sklearn-compatible pipeline stub."""
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.85, 0.15]])


def test_predict_one_returns_expected_keys(monkeypatch):
    """predict_one should return prediction, confidence, and label."""
    import src.ml.predictor as predictor_module

    # Patch _get_pipeline so the real model file is never needed
    monkeypatch.setattr(predictor_module, "_get_pipeline", lambda: _FakePipeline())
    # Also reset the cached pipeline so our patch is used
    monkeypatch.setattr(predictor_module, "_pipeline", None)

    from src.ml.predictor import predict_one  # re-import AFTER patch
    result = predict_one(VALID_INPUT)

    assert "prediction"  in result
    assert "confidence"  in result
    assert "label"       in result


def test_predict_one_pass_label(monkeypatch):
    import src.ml.predictor as predictor_module
    monkeypatch.setattr(predictor_module, "_get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(predictor_module, "_pipeline", None)

    from src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)

    assert result["prediction"] == 0
    assert result["label"] == "Pass"
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_one_defect_label(monkeypatch):
    class _DefectPipeline:
        def predict(self, X):      return np.array([1])
        def predict_proba(self, X): return np.array([[0.1, 0.9]])

    import src.ml.predictor as predictor_module
    monkeypatch.setattr(predictor_module, "_get_pipeline", lambda: _DefectPipeline())
    monkeypatch.setattr(predictor_module, "_pipeline", None)

    from src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)

    assert result["prediction"] == 1
    assert result["label"] == "Defect"


def test_predict_one_missing_feature(monkeypatch):
    import src.ml.predictor as predictor_module
    monkeypatch.setattr(predictor_module, "_get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(predictor_module, "_pipeline", None)

    from src.ml.predictor import predict_one
    bad_input = {k: v for k, v in VALID_INPUT.items() if k != "pressure"}

    with pytest.raises(ValueError, match="pressure"):
        predict_one(bad_input)


# ── Integration smoke-test (skipped if model file absent) ─────────────────────
def test_predict_one_integration():
    """
    Full end-to-end test using the real model.
    Skipped automatically if the model file hasn't been trained yet.
    """
    from pathlib import Path
    model_path = (
        Path(__file__).resolve().parents[1] / "models" / "quality_model.joblib"
    )
    pytest.importorskip("joblib")   # always available, keeps import clean
    if not model_path.exists():
        pytest.skip("Model file not present — run `python -m src.ml.train_model` first.")

    # Reset cache so the real file is loaded fresh
    import src.ml.predictor as predictor_module
    predictor_module._pipeline = None

    from src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["label"] in ("Pass", "Defect")
    
