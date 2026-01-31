"""
Model Training Module

This module handles training and saving machine learning models.
"""
import os
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTraining:
    """Handle model training and persistence."""

    def __init__(
        self,
        models_dir: str = "models",
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize ModelTraining.

        Args:
            models_dir: Directory to save trained models
            n_estimators: Number of trees in random forest
            random_state: Random seed for reproducibility
        """
        self.models_dir = models_dir
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        os.makedirs(self.models_dir, exist_ok=True)

    def create_model(self) -> RandomForestClassifier:
        """
        Create a Random Forest classifier.

        Returns:
            Untrained RandomForestClassifier instance
        """
        logger.info("Creating Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        logger.info(f"Model created with {self.n_estimators} estimators")
        return model

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training model...")

        self.model = self.create_model()
        self.model.fit(X_train, y_train)

        logger.info("Model training completed")

    def cross_validate(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
    ) -> dict:
        """
        Perform cross-validation on training data.

        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        if self.model is None:
            self.model = self.create_model()

        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')

        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }

        logger.info(f"Cross-validation mean accuracy: {results['mean_score']:.4f} "
                    f"(+/- {results['std_score']:.4f})")

        return results

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")

        logger.info("Extracting feature importance...")

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nFeature Importance:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance

    def save_model(self, filename: str = "model.joblib") -> None:
        """
        Save trained model to disk.

        Args:
            filename: Name of file to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")

        filepath = os.path.join(self.models_dir, filename)
        logger.info(f"Saving model to {filepath}...")

        joblib.dump(self.model, filepath)
        logger.info("Model saved successfully")

    def load_model(self, filename: str = "model.joblib") -> None:
        """
        Load model from disk.

        Args:
            filename: Name of file to load model from
        """
        filepath = os.path.join(self.models_dir, filename)
        logger.info(f"Loading model from {filepath}...")

        self.model = joblib.load(filepath)
        logger.info("Model loaded successfully")

    def run_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Run the complete model training pipeline.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Dictionary of cross-validation results
        """
        logger.info("Starting model training pipeline...")

        # Perform cross-validation
        cv_results = self.cross_validate(X_train, y_train)

        # Train final model
        self.train_model(X_train, y_train)

        # Get feature importance
        self.get_feature_importance(list(X_train.columns))

        # Save model
        self.save_model()

        logger.info("Model training pipeline completed successfully")
        return cv_results


if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    # Run model training
    trainer = ModelTraining()
    cv_results = trainer.run_pipeline(X_train, y_train)

    print(f"\nModel training completed!")
    print(f"Cross-validation accuracy: {cv_results['mean_score']:.4f} "
          f"(+/- {cv_results['std_score']:.4f})")
