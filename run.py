import os
import argparse
import joblib
import pandas as pd
import numpy as np
import mlflow
from src.utils.logger import get_logger
from src.data.ingest import ingest_data
from src.data.validate import validate_dataset
from src.features.build_features import FeaturePipeline
from src.models.train import ModelTrainer
from src.models.evaluate import evaluate_predictions, optimize_threshold

logger = get_logger(__name__)

def run_pipeline(sample_size: int = 50000):
    logger.info("Initializing Fraud Intelligence Platform Training Pipeline...")
    
    # 1. Ingest Raw Datasets
    train_raw = ingest_data("fraudTrain.csv", sample_size=sample_size)
    test_raw = ingest_data("fraudTest.csv", sample_size=int(sample_size * 0.3))
    
    # 2. Data Contract & Quality Validation (Great Expectations)
    logger.info("Running Great Expectations data validation checks...")
    train_valid = validate_dataset(train_raw, "train_transactions")
    test_valid = validate_dataset(test_raw, "test_transactions")
    
    if not train_valid or not test_valid:
        logger.warning("Data quality validation failed rules! Proceeding with caution.")
    else:
        logger.info("Data quality validation successful.")
        
    # 3. Advanced Feature Engineering Pipeline
    logger.info("Applying feature engineering pipeline...")
    pipeline = FeaturePipeline()
    pipeline.fit(train_raw)
    
    train_feat = pipeline.transform(train_raw)
    test_feat = pipeline.transform(test_raw)
    
    X_train = train_feat
    y_train = train_raw['is_fraud']
    X_test = test_feat
    y_test = test_raw['is_fraud']
    
    logger.info(f"Training features dimensions: {X_train.shape}")
    logger.info(f"Testing features dimensions: {X_test.shape}")
    
    # 4. Supervised Model Training & Registry
    trainer = ModelTrainer()
    
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_test, y_test, n_trials=3)
    
    logger.info(f"Training final XGBoost model using optimized parameters: {best_params}")
    best_xgboost = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    logger.info("Training comparison baseline models...")
    _ = trainer.train_lightgbm(X_train, y_train, X_test, y_test)
    _ = trainer.train_catboost(X_train, y_train, X_test, y_test)
    _ = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Create target directory for models
    os.makedirs("models", exist_ok=True)
    
    # 5. Threshold Tuning & Business Evaluation
    test_probs = best_xgboost.predict_proba(X_test)[:, 1]
    best_threshold, best_cost_metrics = optimize_threshold(y_test.values, test_probs, test_raw['amt'].values)
    
    logger.info("Optimized Decision Threshold Summary:")
    logger.info(f"- Recommended Threshold: {best_threshold:.4f}")
    logger.info(f"- Cost Savings / Fraud Prevented: {best_cost_metrics['true_positives']} caught, {best_cost_metrics['false_positives']} false alarms.")
    
    # Save Pipeline and Model to Disk for API Serving
    joblib.dump(best_xgboost, "models/best_model.pkl")
    joblib.dump(pipeline, "models/feature_pipeline.pkl")
    logger.info("Model training artifacts saved to ./models/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=15000, help="Row sample limit to run train quickly")
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample_size)
