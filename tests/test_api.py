"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from inference_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


VALID = {
    "stat_type":      "points",
    "line":            25.5,
    "rolling_5_avg":   26.0,
    "rolling_10_avg":  25.0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_returns_status(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_health_returns_version(self, client):
        data = client.get("/health").json()
        assert "version" in data and "model_version" in data

    def test_health_returns_timestamp(self, client):
        assert "timestamp" in client.get("/health").json()


class TestModelEndpoint:
    def test_model_returns_200(self, client):
        assert client.get("/model").status_code == 200

    def test_model_returns_version(self, client):
        data = client.get("/model").json()
        assert "model_version" in data and "api_version" in data

    def test_model_returns_inputs(self, client):
        inputs = client.get("/model").json()["inputs"]
        assert "stat_type" in inputs and "line" in inputs
        assert "rolling_5_avg" in inputs and "rolling_10_avg" in inputs

    def test_model_returns_engineered_features(self, client):
        feats = client.get("/model").json()["engineered_features"]
        assert "line_vs_avg" in feats and "avg_trend" in feats

    def test_model_returns_supported_sports(self, client):
        assert "NBA" in client.get("/model").json()["supported_sports"]


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        assert client.post("/predict", json=VALID).status_code == 200

    def test_predict_returns_probability(self, client):
        data = client.post("/predict", json=VALID).json()
        assert "probability" in data and 0.0 <= data["probability"] <= 1.0

    def test_predict_returns_confidence(self, client):
        assert client.post("/predict", json=VALID).json()["confidence"] in ["low", "medium", "high"]

    def test_predict_returns_recommendation(self, client):
        assert client.post("/predict", json=VALID).json()["recommendation"] in [
            "OVER", "UNDER", "NO_EDGE"
        ]

    def test_predict_returns_model_version(self, client):
        assert client.post("/predict", json=VALID).json()["model_version"]

    def test_predict_returns_source(self, client):
        assert client.post("/predict", json=VALID).json()["source"] in ["model", "heuristic"]

    def test_predict_requires_stat_type(self, client):
        assert client.post("/predict", json={k: v for k, v in VALID.items() if k != "stat_type"}).status_code == 422

    def test_predict_requires_line(self, client):
        assert client.post("/predict", json={k: v for k, v in VALID.items() if k != "line"}).status_code == 422

    def test_predict_requires_rolling_avgs(self, client):
        assert client.post("/predict", json={"stat_type": "points", "line": 25.5}).status_code == 422

    def test_valid_stat_types_accepted(self, client):
        for stat in ["points", "rebounds", "assists"]:
            assert client.post("/predict", json={**VALID, "stat_type": stat}).status_code == 200, stat

    def test_invalid_stat_type_rejected(self, client):
        assert client.post("/predict", json={**VALID, "stat_type": "goals"}).status_code == 422

    def test_line_must_be_positive(self, client):
        assert client.post("/predict", json={**VALID, "line": -1.0}).status_code == 422


class TestBatchEndpoint:
    def test_batch_returns_200(self, client):
        assert client.post("/predict/batch", json={"predictions": [VALID, VALID]}).status_code == 200

    def test_batch_returns_correct_count(self, client):
        data = client.post("/predict/batch", json={"predictions": [VALID] * 3}).json()
        assert data["count"] == 3 and len(data["results"]) == 3

    def test_batch_each_result_is_valid(self, client):
        results = client.post("/predict/batch", json={"predictions": [VALID, VALID]}).json()["results"]
        for r in results:
            assert 0.0 <= r["probability"] <= 1.0
            assert r["confidence"] in ["low", "medium", "high"]
            assert r["recommendation"] in ["OVER", "UNDER", "NO_EDGE"]

    def test_batch_empty_rejected(self, client):
        assert client.post("/predict/batch", json={"predictions": []}).status_code == 422

    def test_batch_over_limit_rejected(self, client):
        assert client.post("/predict/batch", json={"predictions": [VALID] * 51}).status_code == 422


class TestRequestHeaders:
    def test_response_includes_request_id(self, client):
        assert "X-Request-ID" in client.get("/health").headers

    def test_response_includes_latency(self, client):
        resp = client.get("/health")
        assert "X-Latency-MS" in resp.headers and float(resp.headers["X-Latency-MS"]) >= 0
