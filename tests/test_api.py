"""
Unit tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestAPI:
    """Test cases for API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "transformer_loaded" in data

    def test_predict_endpoint_valid(self):
        """Test prediction endpoint with valid data."""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "prediction_label" in data
        assert "confidence" in data
        assert "probabilities" in data

        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1, 2]
        assert isinstance(data["prediction_label"], str)
        assert 0 <= data["confidence"] <= 1
        assert len(data["probabilities"]) == 3

    def test_predict_endpoint_invalid_negative(self):
        """Test prediction endpoint with negative values."""
        payload = {
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self):
        """Test prediction endpoint with missing field."""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
            # Missing petal_width
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_type(self):
        """Test prediction endpoint with invalid data type."""
        payload = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_docs_endpoint(self):
        """Test that OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self):
        """Test that ReDoc docs are accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
