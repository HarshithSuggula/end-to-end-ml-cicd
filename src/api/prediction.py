"""
Prediction service for loading model and making predictions.
"""
import os
import logging
import joblib
import pandas as pd
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions with trained model."""

    # Class labels for Iris dataset
    CLASS_LABELS = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    def __init__(self, models_dir: str = "models"):
        """
        Initialize PredictionService.

        Args:
            models_dir: Directory containing model artifacts
        """
        self.models_dir = models_dir
        self.model = None
        self.transformer = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load model and transformer from disk."""
        try:
            model_path = os.path.join(self.models_dir, "model.joblib")
            transformer_path = os.path.join(self.models_dir, "feature_transformer.joblib")

            logger.info(f"Loading model from {model_path}...")
            self.model = joblib.load(model_path)

            logger.info(f"Loading transformer from {transformer_path}...")
            self.transformer = joblib.load(transformer_path)

            logger.info("Artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """
        Preprocess input data for prediction.

        Args:
            input_data: Dictionary with feature values

        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Create DataFrame with correct column names
        df = pd.DataFrame([{
            'sepal length (cm)': input_data['sepal_length'],
            'sepal width (cm)': input_data['sepal_width'],
            'petal length (cm)': input_data['petal_length'],
            'petal width (cm)': input_data['petal_width']
        }])

        # Transform using fitted transformer
        df_transformed = self.transformer.transform(df)

        return pd.DataFrame(
            df_transformed,
            columns=df.columns
        )

    def predict(self, input_data: dict) -> Tuple[int, str, float, list]:
        """
        Make prediction for input data.

        Args:
            input_data: Dictionary with feature values

        Returns:
            Tuple of (prediction, label, confidence, probabilities)
        """
        if self.model is None or self.transformer is None:
            raise ValueError("Model or transformer not loaded")

        # Preprocess input
        X = self.preprocess_input(input_data)

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0].tolist()

        # Get label and confidence
        label = self.CLASS_LABELS[prediction]
        confidence = max(probabilities)

        logger.info(f"Prediction: {prediction} ({label}), Confidence: {confidence:.4f}")

        return int(prediction), label, float(confidence), probabilities

    def is_healthy(self) -> Tuple[bool, bool]:
        """
        Check if service is healthy.

        Returns:
            Tuple of (model_loaded, transformer_loaded)
        """
        return self.model is not None, self.transformer is not None


# Global instance for reuse across requests
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """
    Get or create global prediction service instance.

    Returns:
        PredictionService instance
    """
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
