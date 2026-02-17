"""
API endpoint tests.

Tests for:
- Health check endpoint
- Model info endpoint
- Prediction endpoint
- Error handling
- Input validation
"""

import pytest
from fastapi.testclient import TestClient

from inference_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health response should include status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_version(self, client):
        """Health response should include version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert "model_version" in data

    def test_health_returns_timestamp(self, client):
        """Health response should include timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestModelEndpoint:
    """Tests for /model endpoint."""

    def test_model_returns_200(self, client):
        """Model endpoint should return 200."""
        response = client.get("/model")
        assert response.status_code == 200

    def test_model_returns_version(self, client):
        """Model response should include version."""
        response = client.get("/model")
        data = response.json()
        assert "model_version" in data
        assert "api_version" in data

    def test_model_returns_inputs(self, client):
        """Model response should include inputs list."""
        response = client.get("/model")
        data = response.json()
        assert "inputs" in data
        assert isinstance(data["inputs"], list)
        assert len(data["inputs"]) > 0
        # Check required inputs are listed
        assert "sport" in data["inputs"]
        assert "stat_type" in data["inputs"]
        assert "line" in data["inputs"]

    def test_model_returns_engineered_features(self, client):
        """Model response should include engineered features."""
        response = client.get("/model")
        data = response.json()
        assert "engineered_features" in data
        assert isinstance(data["engineered_features"], list)
        assert "line_vs_avg" in data["engineered_features"]
        assert "elo_diff" in data["engineered_features"]

    def test_model_returns_supported_sports(self, client):
        """Model response should include supported sports."""
        response = client.get("/model")
        data = response.json()
        assert "supported_sports" in data
        assert "NBA" in data["supported_sports"]

    def test_model_returns_notes(self, client):
        """Model response should include notes field."""
        response = client.get("/model")
        data = response.json()
        assert "notes" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_200(self, client):
        """Predict endpoint should return 200 for valid input."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
            "line": 25.5,
            "glicko_mu": 1600.0,
            "last_5_avg": 27.0,
            "last_10_avg": 26.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_probability(self, client):
        """Predict response should include probability."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_returns_confidence(self, client):
        """Predict response should include confidence level."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "confidence" in data
        assert data["confidence"] in ["low", "medium", "high"]

    def test_predict_returns_recommendation(self, client):
        """Predict response should include recommendation."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "recommendation" in data
        assert data["recommendation"] in ["OVER", "UNDER", "NO_EDGE"]

    def test_predict_returns_model_version(self, client):
        """Predict response should include model version."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "model_version" in data

    def test_predict_requires_sport(self, client):
        """Predict should require sport field."""
        payload = {
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_requires_line(self, client):
        """Predict should require line field."""
        payload = {
            "sport": "NBA",
            "stat_type": "points",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_all_sports(self, client):
        """Predict should accept all supported sports."""
        for sport in ["NBA", "NFL", "NHL", "MLB"]:
            payload = {
                "sport": sport,
                "stat_type": "points",
                "line": 25.5,
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for sport: {sport}"


class TestInputValidation:
    """Tests for input validation with enums."""

    def test_invalid_sport_rejected(self, client):
        """Invalid sport value should return 422."""
        payload = {
            "sport": "INVALID_SPORT",
            "stat_type": "points",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_invalid_stat_type_rejected(self, client):
        """Invalid stat_type value should return 422."""
        payload = {
            "sport": "NBA",
            "stat_type": "invalid_stat",
            "line": 25.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_valid_stat_types_accepted(self, client):
        """Valid stat types should be accepted."""
        valid_stats = ["points", "rebounds", "assists", "goals", "hits"]
        for stat in valid_stats:
            payload = {
                "sport": "NBA",
                "stat_type": stat,
                "line": 25.5,
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for stat_type: {stat}"


class TestRequestHeaders:
    """Tests for request tracking headers."""

    def test_response_includes_request_id(self, client):
        """Response should include X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_response_includes_latency(self, client):
        """Response should include X-Latency-MS header."""
        response = client.get("/health")
        assert "X-Latency-MS" in response.headers
        # Latency should be a valid number
        latency = float(response.headers["X-Latency-MS"])
        assert latency >= 0
