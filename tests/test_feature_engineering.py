"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineering
from src.data_ingestion import DataIngestion


@pytest.fixture
def feature_eng():
    """Fixture for FeatureEngineering instance."""
    return FeatureEngineering(models_dir="test_models")


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    ingestion = DataIngestion()
    X, y = ingestion.load_data()
    X_train, X_test, y_train, y_test = ingestion.split_data(X, y)
    return X_train, X_test


class TestFeatureEngineering:
    """Test cases for FeatureEngineering class."""

    def test_create_transformer(self, feature_eng):
        """Test transformer creation."""
        transformer = feature_eng.create_transformer()
        assert transformer is not None
        assert hasattr(transformer, 'fit_transform')

    def test_fit_transform(self, feature_eng, sample_data):
        """Test fitting and transforming data."""
        X_train, _ = sample_data
        X_transformed = feature_eng.fit_transform(X_train)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X_train.shape
        assert list(X_transformed.columns) == list(X_train.columns)

        # Check that data is scaled (mean ~0, std ~1)
        assert np.abs(X_transformed.mean().mean()) < 0.1
        assert np.abs(X_transformed.std().mean() - 1.0) < 0.1

    def test_transform_without_fit(self, feature_eng, sample_data):
        """Test transform before fitting raises error."""
        _, X_test = sample_data

        with pytest.raises(ValueError, match="Transformer not fitted"):
            feature_eng.transform(X_test)

    def test_transform_after_fit(self, feature_eng, sample_data):
        """Test transform after fitting."""
        X_train, X_test = sample_data

        feature_eng.fit_transform(X_train)
        X_transformed = feature_eng.transform(X_test)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X_test.shape

    def test_run_pipeline(self, feature_eng, sample_data):
        """Test complete pipeline execution."""
        X_train, X_test = sample_data
        X_train_transformed, X_test_transformed = feature_eng.run_pipeline(X_train, X_test)

        assert X_train_transformed.shape == X_train.shape
        assert X_test_transformed.shape == X_test.shape
        assert feature_eng.transformer is not None
