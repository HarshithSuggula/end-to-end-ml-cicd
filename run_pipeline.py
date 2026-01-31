"""
Main Pipeline Runner

This script runs the complete end-to-end ML pipeline.
"""
import logging
import sys
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the complete ML pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting End-to-End ML Pipeline")
        logger.info("=" * 60)

        # Step 1: Data Ingestion
        logger.info("\n[Step 1/5] Data Ingestion")
        logger.info("-" * 60)
        ingestion = DataIngestion()
        X_train, X_test, y_train, y_test = ingestion.run_pipeline()

        # Step 2: Feature Engineering
        logger.info("\n[Step 2/5] Feature Engineering")
        logger.info("-" * 60)
        feature_eng = FeatureEngineering()
        X_train_transformed, X_test_transformed = feature_eng.run_pipeline(
            X_train, X_test
        )

        # Step 3: Model Training
        logger.info("\n[Step 3/5] Model Training")
        logger.info("-" * 60)
        trainer = ModelTraining()
        cv_results = trainer.run_pipeline(X_train_transformed, y_train)

        # Step 4: Model Evaluation
        logger.info("\n[Step 4/5] Model Evaluation")
        logger.info("-" * 60)
        y_pred = trainer.model.predict(X_test_transformed)
        evaluator = ModelEvaluation()
        eval_results = evaluator.run_pipeline(y_test, y_pred)

        # Step 5: Summary
        logger.info("\n[Step 5/5] Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Cross-validation accuracy: "
                    f"{cv_results['mean_score']:.4f} "
                    f"(+/- {cv_results['std_score']:.4f})")
        logger.info(f"Test accuracy: {eval_results['metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-score: {eval_results['metrics']['f1_score']:.4f}")
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        logger.info("\nGenerated artifacts:")
        logger.info("  - data/: Train and test datasets")
        logger.info("  - models/: Trained model and transformer")
        logger.info("  - results/: Evaluation metrics")
        logger.info("\nNext steps:")
        logger.info("  1. Start API: uvicorn src.api.main:app --reload")
        logger.info("  2. Run tests: pytest tests/ -v")
        logger.info("  3. Build Docker: docker build -t ml-pipeline .")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
