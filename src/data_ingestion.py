"""
Data Ingestion Module

This module handles loading, validating, and splitting data for the ML pipeline.
"""
import os
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data loading, validation, and splitting."""

    def __init__(self, data_dir: str = "data", test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataIngestion.

        Args:
            data_dir: Directory to save processed data
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Iris dataset.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Loading Iris dataset...")
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name='target')

        logger.info(f"Loaded dataset with shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target classes: {np.unique(y)}")

        return X, y

    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Validate data quality.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating data...")

        # Check for missing values
        if X.isnull().any().any():
            raise ValueError("Features contain missing values")

        if y.isnull().any():
            raise ValueError("Target contains missing values")

        # Check dimensions match
        if len(X) != len(y):
            raise ValueError("Features and target have different lengths")

        # Check for minimum samples
        if len(X) < 10:
            raise ValueError("Dataset too small (minimum 10 samples required)")

        logger.info("Data validation passed")
        return True

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data with test_size={self.test_size}...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def save_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Save processed data to disk.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """
        logger.info(f"Saving data to {self.data_dir}...")

        X_train.to_csv(os.path.join(self.data_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.data_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.data_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.data_dir, "y_test.csv"), index=False)

        logger.info("Data saved successfully")

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete data ingestion pipeline.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data ingestion pipeline...")

        # Load data
        X, y = self.load_data()

        # Validate data
        self.validate_data(X, y)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Save data
        self.save_data(X_train, X_test, y_train, y_test)

        logger.info("Data ingestion pipeline completed successfully")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    ingestion = DataIngestion()
    X_train, X_test, y_train, y_test = ingestion.run_pipeline()
    print("\nPipeline completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
