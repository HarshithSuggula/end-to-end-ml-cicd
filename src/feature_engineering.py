"""
Feature Engineering Module

This module handles feature transformations and preprocessing for the ML pipeline.
"""
import os
import logging
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Handle feature transformations and preprocessing."""

    def __init__(self, models_dir: str = "models"):
        """
        Initialize FeatureEngineering.

        Args:
            models_dir: Directory to save feature transformers
        """
        self.models_dir = models_dir
        self.transformer = None
        os.makedirs(self.models_dir, exist_ok=True)

    def create_transformer(self) -> Pipeline:
        """
        Create feature transformation pipeline.

        Returns:
            Scikit-learn Pipeline for feature transformation
        """
        logger.info("Creating feature transformation pipeline...")

        # For Iris dataset, we'll use standard scaling
        transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        logger.info("Feature transformer created")
        return transformer

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit transformer on training data and transform it.

        Args:
            X_train: Training features

        Returns:
            Transformed training features
        """
        logger.info("Fitting and transforming training data...")

        self.transformer = self.create_transformer()
        X_train_transformed = self.transformer.fit_transform(X_train)

        # Convert back to DataFrame
        X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=X_train.columns,
            index=X_train.index
        )

        logger.info(f"Training data transformed. Shape: {X_train_transformed.shape}")
        return X_train_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformer.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if self.transformer is None:
            raise ValueError("Transformer not fitted. Call fit_transform first.")

        logger.info("Transforming data...")
        X_transformed = self.transformer.transform(X)

        # Convert back to DataFrame
        X_transformed = pd.DataFrame(
            X_transformed,
            columns=X.columns,
            index=X.index
        )

        logger.info(f"Data transformed. Shape: {X_transformed.shape}")
        return X_transformed

    def save_transformer(self, filename: str = "feature_transformer.joblib") -> None:
        """
        Save fitted transformer to disk.

        Args:
            filename: Name of file to save transformer
        """
        if self.transformer is None:
            raise ValueError("No transformer to save. Fit transformer first.")

        filepath = os.path.join(self.models_dir, filename)
        logger.info(f"Saving transformer to {filepath}...")

        joblib.dump(self.transformer, filepath)
        logger.info("Transformer saved successfully")

    def load_transformer(self, filename: str = "feature_transformer.joblib") -> None:
        """
        Load transformer from disk.

        Args:
            filename: Name of file to load transformer from
        """
        filepath = os.path.join(self.models_dir, filename)
        logger.info(f"Loading transformer from {filepath}...")

        self.transformer = joblib.load(filepath)
        logger.info("Transformer loaded successfully")

    def run_pipeline(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete feature engineering pipeline.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_transformed, X_test_transformed)
        """
        logger.info("Starting feature engineering pipeline...")

        # Fit and transform training data
        X_train_transformed = self.fit_transform(X_train)

        # Transform test data
        X_test_transformed = self.transform(X_test)

        # Save transformer
        self.save_transformer()

        logger.info("Feature engineering pipeline completed successfully")
        return X_train_transformed, X_test_transformed


if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    # Run feature engineering
    fe = FeatureEngineering()
    X_train_transformed, X_test_transformed = fe.run_pipeline(X_train, X_test)

    print("\nFeature engineering completed!")
    print(f"Training features shape: {X_train_transformed.shape}")
    print(f"Test features shape: {X_test_transformed.shape}")
