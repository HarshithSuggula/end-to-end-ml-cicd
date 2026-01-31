"""
FastAPI Application - ML Model Serving API

This module defines the FastAPI application for serving machine learning predictions.
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.prediction import get_prediction_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Pipeline API",
    description="REST API for Iris flower classification using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status.

    Returns:
        HealthResponse with service status
    """
    try:
        service = get_prediction_service()
        model_loaded, transformer_loaded = service.is_healthy()

        status = "healthy" if (model_loaded and transformer_loaded) else "unhealthy"

        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            transformer_loaded=transformer_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            transformer_loaded=False
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for Iris flower classification.

    Args:
        request: PredictionRequest with flower measurements

    Returns:
        PredictionResponse with prediction results

    Raises:
        HTTPException: If prediction fails
    """
    try:
        service = get_prediction_service()

        # Convert request to dict
        input_data = {
            'sepal_length': request.sepal_length,
            'sepal_width': request.sepal_width,
            'petal_length': request.petal_length,
            'petal_width': request.petal_width
        }

        # Make prediction
        prediction, label, confidence, probabilities = service.predict(input_data)

        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
