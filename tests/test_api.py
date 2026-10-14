"""
Integration Tests for FastAPI Endpoints.
Tests:
- GET /health
- GET /metrics
- POST /predict (Valid sample, Missing feature, Extra feature, Non-finite values)
- POST /predict-batch
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app, load_model_artifacts

@pytest.fixture(scope="module")
def client():
    load_model_artifacts()
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_records():
    samples_path = os.path.join("artifacts", "sample_records.json")
    if not os.path.exists(samples_path):
        pytest.skip("sample_records.json not available")
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_api_health(client):
    """Verify /health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data

def test_api_metrics(client):
    """Verify /metrics returns offline metrics with distinction from live operational queries."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "offline_test_metrics" in data
    assert "operational_live_requests_count" in data

def test_predict_valid_ddos_sample(client, sample_records):
    """Verify /predict correctly detects known DDoS sample."""
    ddos_feat = sample_records["ddos_sample"]
    response = client.post("/predict", json={"features": ddos_feat})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["label"] == "DDoS"
    assert data["probability"] >= 0.5
    assert "inference_latency_ms" in data

def test_predict_valid_benign_sample(client, sample_records):
    """Verify /predict correctly detects known Benign sample."""
    benign_feat = sample_records["benign_sample"]
    response = client.post("/predict", json={"features": benign_feat})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["label"] == "BENIGN"
    assert data["probability"] < 0.5
    assert "inference_latency_ms" in data

def test_predict_missing_feature(client, sample_records):
    """Verify /predict rejects request missing a required feature with 422."""
    incomplete_feat = dict(sample_records["benign_sample"])
    del incomplete_feat["Destination Port"]  # delete one required feature
    response = client.post("/predict", json={"features": incomplete_feat})
    assert response.status_code == 422

def test_predict_extra_feature(client, sample_records):
    """Verify /predict rejects unexpected extra fields with 422."""
    bad_feat = dict(sample_records["benign_sample"])
    bad_feat["malicious_extra_field"] = 999.0
    response = client.post("/predict", json={"features": bad_feat})
    assert response.status_code == 422

def test_predict_invalid_string_type(client, sample_records):
    """Verify /predict rejects non-numeric types with 422."""
    bad_feat = dict(sample_records["benign_sample"])
    bad_feat["Flow Bytes/s"] = "not_a_number"
    response = client.post("/predict", json={"features": bad_feat})
    assert response.status_code == 422

def test_predict_batch(client, sample_records):
    """Verify /predict-batch correctly handles multiple records."""
    batch_payload = {
        "records": [
            {"features": sample_records["benign_sample"]},
            {"features": sample_records["ddos_sample"]}
        ]
    }
    response = client.post("/predict-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction"] == 0
    assert data["predictions"][1]["prediction"] == 1
