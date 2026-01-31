"""
Unit tests for model training module.
"""
import pytest
import pandas as pd
from src.model_training import ModelTraining
from src.data_ingestion import DataIngestion


@pytest.fixture
def trainer():
    """Fixture for ModelTraining instance."""
    return ModelTraining(models_dir="test_models", n_estimators=10)


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    ingestion = DataIngestion()
    X, y = ingestion.load_data()
    X_train, _, y_train, _ = ingestion.split_data(X, y)
    return X_train, y_train


class TestModelTraining:
    """Test cases for ModelTraining class."""

    def test_create_model(self, trainer):
        """Test model creation."""
        model = trainer.create_model()
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_train_model(self, trainer, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        trainer.train_model(X_train, y_train)

        assert trainer.model is not None
        assert hasattr(trainer.model, 'predict')

        # Check that model can make predictions
        predictions = trainer.model.predict(X_train)
        assert len(predictions) == len(y_train)

    def test_cross_validate(self, trainer, sample_data):
        """Test cross-validation."""
        X_train, y_train = sample_data
        results = trainer.cross_validate(X_train, y_train, cv=3)

        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'scores' in results
        assert results['mean_score'] > 0
        assert results['mean_score'] <= 1.0
        assert len(results['scores']) == 3

    def test_get_feature_importance(self, trainer, sample_data):
        """Test feature importance extraction."""
        X_train, y_train = sample_data
        trainer.train_model(X_train, y_train)

        importance = trainer.get_feature_importance(list(X_train.columns))

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == X_train.shape[1]
        assert importance['importance'].sum() > 0

    def test_model_not_trained_error(self, trainer):
        """Test error when accessing untrained model."""
        with pytest.raises(ValueError, match="Model not trained"):
            trainer.get_feature_importance(['a', 'b'])

    def test_run_pipeline(self, trainer, sample_data):
        """Test complete pipeline execution."""
        X_train, y_train = sample_data
        results = trainer.run_pipeline(X_train, y_train)

        assert 'mean_score' in results
        assert trainer.model is not None
