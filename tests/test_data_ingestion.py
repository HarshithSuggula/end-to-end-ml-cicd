"""
Unit tests for data ingestion module.
"""
import pytest
import pandas as pd
from src.data_ingestion import DataIngestion


@pytest.fixture
def ingestion():
    """Fixture for DataIngestion instance."""
    return DataIngestion(data_dir="test_data")


class TestDataIngestion:
    """Test cases for DataIngestion class."""

    def test_load_data(self, ingestion):
        """Test data loading."""
        X, y = ingestion.load_data()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        assert X.shape[1] == 4  # Iris has 4 features

    def test_validate_data_valid(self, ingestion):
        """Test validation with valid data."""
        X, y = ingestion.load_data()
        result = ingestion.validate_data(X, y)
        assert result is True

    def test_validate_data_missing_values(self, ingestion):
        """Test validation with missing values."""
        X = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
        y = pd.Series([0, 1, 2])

        with pytest.raises(ValueError, match="Features contain missing values"):
            ingestion.validate_data(X, y)

    def test_validate_data_length_mismatch(self, ingestion):
        """Test validation with length mismatch."""
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y = pd.Series([0, 1])

        with pytest.raises(ValueError, match="different lengths"):
            ingestion.validate_data(X, y)

    def test_split_data(self, ingestion):
        """Test data splitting."""
        X, y = ingestion.load_data()
        X_train, X_test, y_train, y_test = ingestion.split_data(X, y)

        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        # Check sizes
        total_size = len(X_train) + len(X_test)
        assert total_size == len(X)
        assert len(X_train) > len(X_test)  # 80/20 split

        # Check no overlap
        assert len(set(X_train.index).intersection(set(X_test.index))) == 0

    def test_run_pipeline(self, ingestion):
        """Test complete pipeline execution."""
        X_train, X_test, y_train, y_test = ingestion.run_pipeline()

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
