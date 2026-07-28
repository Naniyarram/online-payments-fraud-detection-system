import os
import joblib
import optuna
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger
from src.models.evaluate import evaluate_predictions

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name: str = "Fraud_Detection_Intelligence"):
        self.experiment_name = experiment_name
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_available = True
        except Exception as e:
            logger.warning(f"MLflow tracking disabled due to environment issues: {e}")
            self.mlflow_available = False

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
        logger.info("Starting XGBoost training...")
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42
        )
        model._estimator_type = "classifier"
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, probs)
        
        if self.mlflow_available:
            try:
                import mlflow
                with mlflow.start_run(run_name="XGBoost_Baseline"):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to log run to MLflow: {e}")
                
        logger.info(f"XGBoost Baseline trained successfully. Metrics: {metrics}")
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> LGBMClassifier:
        logger.info("Starting LightGBM training...")
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )
        model._estimator_type = "classifier"
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, probs)
        
        if self.mlflow_available:
            try:
                import mlflow
                with mlflow.start_run(run_name="LightGBM_Baseline"):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to log run to MLflow: {e}")
                
        logger.info(f"LightGBM Baseline trained. Metrics: {metrics}")
        return model

    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> CatBoostClassifier:
        logger.info("Starting CatBoost training...")
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=False
        )
        model._estimator_type = "classifier"
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, probs)
        
        if self.mlflow_available:
            try:
                import mlflow
                with mlflow.start_run(run_name="CatBoost_Baseline"):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to log run to MLflow: {e}")
                
        logger.info(f"CatBoost Baseline trained. Metrics: {metrics}")
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> RandomForestClassifier:
        logger.info("Starting Random Forest training...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            n_jobs=-1
        )
        model._estimator_type = "classifier"
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, probs)
        
        if self.mlflow_available:
            try:
                import mlflow
                with mlflow.start_run(run_name="RandomForest_Baseline"):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to log run to MLflow: {e}")
                
        logger.info(f"Random Forest Baseline trained. Metrics: {metrics}")
        return model

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 5) -> Dict[str, Any]:
        logger.info("Running Optuna Hyperparameter Optimization...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="aucpr")
            model._estimator_type = "classifier"
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            probs = model.predict_proba(X_val)[:, 1]
            metrics = evaluate_predictions(y_val, probs)
            return metrics["pr_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Optuna Optimization complete. Best trial params: {study.best_params}")
        return study.best_params
