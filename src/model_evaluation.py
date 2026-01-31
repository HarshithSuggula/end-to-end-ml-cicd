"""
Model Evaluation Module

This module handles model evaluation and performance metrics.
"""
import os
import logging
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluation:
    """Handle model evaluation and metrics calculation."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize ModelEvaluation.

        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics...")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        logger.info("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def get_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> list:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix as list of lists
        """
        logger.info("Calculating confusion matrix...")

        cm = confusion_matrix(y_true, y_pred)

        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")

        return cm.tolist()

    def get_classification_report(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> dict:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report as dictionary
        """
        logger.info("Generating classification report...")

        report = classification_report(y_true, y_pred, output_dict=True)

        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_true, y_pred)}")

        return report

    def save_results(
        self,
        metrics: dict,
        confusion_mat: list,
        classification_rep: dict,
        filename: str = "evaluation_results.json"
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            metrics: Dictionary of metrics
            confusion_mat: Confusion matrix
            classification_rep: Classification report
            filename: Output filename
        """
        filepath = os.path.join(self.results_dir, filename)
        logger.info(f"Saving evaluation results to {filepath}...")

        results = {
            'metrics': metrics,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation results saved successfully")

    def run_pipeline(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Run the complete evaluation pipeline.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of all evaluation results
        """
        logger.info("Starting model evaluation pipeline...")

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)

        # Get confusion matrix
        confusion_mat = self.get_confusion_matrix(y_true, y_pred)

        # Get classification report
        classification_rep = self.get_classification_report(y_true, y_pred)

        # Save results
        self.save_results(metrics, confusion_mat, classification_rep)

        results = {
            'metrics': metrics,
            'confusion_matrix': confusion_mat,
            'classification_report': classification_rep
        }

        logger.info("Model evaluation pipeline completed successfully")
        return results


if __name__ == "__main__":
    import joblib

    # Load test data
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    # Load model
    model = joblib.load("models/model.joblib")

    # Make predictions
    y_pred = model.predict(X_test)

    # Run evaluation
    evaluator = ModelEvaluation()
    results = evaluator.run_pipeline(y_test, y_pred)

    print("\nModel evaluation completed!")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
