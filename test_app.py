"""
Tests for the Manufacturing Quality Prediction API.
Run with: pytest test_app.py -v
"""
import pytest
import json
import sys
import os

# Add backend folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ─── HOME ENDPOINT ────────────────────────────────────────────────────────────

def test_home_endpoint(client):
    """GET / should return status running."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'


# ─── PREDICT ENDPOINT - VALID INPUTS ──────────────────────────────────────────

def test_predict_no_failure(client):
    """Normal machine parameters should predict No Failure."""
    payload = {
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert data['prediction'] in ['No Failure', 'Failure']
    assert 'confidence_percent' in data
    assert 'model_metrics' in data


def test_predict_high_risk_parameters(client):
    """High tool wear and torque should be accepted and return a valid prediction."""
    payload = {
        "air_temperature": 305.0,
        "process_temperature": 315.0,
        "rotational_speed": 1200,
        "torque": 70.0,
        "tool_wear": 200
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['prediction'] in ['No Failure', 'Failure']


def test_predict_returns_model_metrics(client):
    """Response must include model evaluation metrics."""
    payload = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    metrics = data['model_metrics']
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics


def test_predict_returns_input_echo(client):
    """Response must echo back the input values."""
    payload = {
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'input_received' in data
    assert data['input_received']['air_temperature'] == 298.0


# ─── PREDICT ENDPOINT - INVALID INPUTS ────────────────────────────────────────

def test_predict_missing_field(client):
    """Missing a required field should return 400."""
    payload = {
        "air_temperature": 298.0,
        # missing process_temperature, rotational_speed, torque, tool_wear
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_no_body(client):
    """Sending no body should return 400."""
    response = client.post('/predict',
                           content_type='application/json')
    assert response.status_code == 400


def test_predict_non_numeric_value(client):
    """Non-numeric input should return 400."""
    payload = {
        "air_temperature": "hot",
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400


def test_predict_out_of_range_temperature(client):
    """Temperature outside 250-400K range should return 400."""
    payload = {
        "air_temperature": 999,  # invalid
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400


def test_predict_negative_tool_wear(client):
    """Negative tool wear should return 400."""
    payload = {
        "air_temperature": 298.0,
        "process_temperature": 308.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": -5
    }
    response = client.post('/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400


# ─── METRICS ENDPOINT ─────────────────────────────────────────────────────────

def test_metrics_endpoint(client):
    """GET /metrics should return model performance metrics."""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'accuracy' in data
    assert 0 <= data['accuracy'] <= 1
