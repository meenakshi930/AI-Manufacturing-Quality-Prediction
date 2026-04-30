"""
Tests for backend/src/ml/predictor.py

Run from repo root:
    pytest backend/tests/test_predictor.py -v
"""

import numpy as np
import pytest


# ── Shared fixtures ───────────────────────────────────────────────────────────
VALID_INPUT = {
    "temperature":      72.5,
    "pressure":        210.0,
    "humidity":          0.48,
    "vibration_level":  98.3,
}


class _PassPipeline:
    """Stub that always predicts class 0 (Pass) with 85 % confidence."""
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.85, 0.15]])


class _DefectPipeline:
    """Stub that always predicts class 1 (Defect) with 90 % confidence."""
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.10, 0.90]])


# ── Helper: patch the module-level cache ─────────────────────────────────────
def _patch_pipeline(monkeypatch, pipeline):
    """
    Patch _get_pipeline() so no model file is loaded from disk.
    Also reset the module-level cache so the patch takes effect.
    """
    import backend.src.ml.predictor as mod
    monkeypatch.setattr(mod, "_pipeline", None)
    monkeypatch.setattr(mod, "_get_pipeline", lambda: pipeline)


# ── Unit tests ────────────────────────────────────────────────────────────────
def test_predict_one_returns_required_keys(monkeypatch):
    _patch_pipeline(monkeypatch, _PassPipeline())
    from backend.src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)
    assert {"prediction", "confidence", "label"} <= result.keys()


def test_predict_one_pass_label(monkeypatch):
    _patch_pipeline(monkeypatch, _PassPipeline())
    from backend.src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)
    assert result["prediction"] == 0
    assert result["label"] == "Pass"
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_one_defect_label(monkeypatch):
    _patch_pipeline(monkeypatch, _DefectPipeline())
    from backend.src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)
    assert result["prediction"] == 1
    assert result["label"] == "Defect"
    assert result["confidence"] == pytest.approx(0.90, abs=1e-4)


def test_predict_one_missing_feature_raises(monkeypatch):
    _patch_pipeline(monkeypatch, _PassPipeline())
    from backend.src.ml.predictor import predict_one
    bad_input = {k: v for k, v in VALID_INPUT.items() if k != "pressure"}
    with pytest.raises(ValueError, match="pressure"):
        predict_one(bad_input)


def test_predict_one_extra_keys_ignored(monkeypatch):
    """Extra keys in the payload should not cause an error."""
    _patch_pipeline(monkeypatch, _PassPipeline())
    from backend.src.ml.predictor import predict_one
    extra = {**VALID_INPUT, "unknown_field": 999}
    result = predict_one(extra)          # should not raise
    assert result["prediction"] in (0, 1)


def test_predict_one_confidence_is_rounded(monkeypatch):
    _patch_pipeline(monkeypatch, _PassPipeline())
    from backend.src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)
    # confidence should have at most 4 decimal places
    assert result["confidence"] == round(result["confidence"], 4)


# ── Integration smoke-test ────────────────────────────────────────────────────
def test_predict_one_integration():
    """
    End-to-end test using the real trained model.
    Auto-skipped if quality_model.joblib has not been generated yet.
    """
    from pathlib import Path
    model_path = (
        Path(__file__).resolve().parents[1] / "models" / "quality_model.joblib"
    )
    if not model_path.exists():
        pytest.skip(
            "Model file absent — run `python -m backend.src.ml.train_model` first."
        )

    # Reset module cache so the real file is loaded fresh
    import backend.src.ml.predictor as mod
    mod._pipeline = None

    from backend.src.ml.predictor import predict_one
    result = predict_one(VALID_INPUT)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["label"] in ("Pass", "Defect")
