"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    sepal_length: float = Field(..., description="Sepal length in cm", gt=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", gt=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", gt=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", gt=0, le=10)

    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def validate_positive(cls, v):
        """Validate that all measurements are positive."""
        if v <= 0:
            raise ValueError('Measurement must be positive')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: int = Field(..., description="Predicted class (0, 1, or 2)")
    prediction_label: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Confidence score", ge=0, le=1)
    probabilities: List[float] = Field(..., description="Probability for each class")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": 0,
                    "prediction_label": "setosa",
                    "confidence": 0.95,
                    "probabilities": [0.95, 0.03, 0.02]
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    transformer_loaded: bool = Field(..., description="Whether transformer is loaded")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "transformer_loaded": True
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Detailed error information")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "Prediction failed",
                    "detail": "Model not loaded"
                }
            ]
        }
    }
