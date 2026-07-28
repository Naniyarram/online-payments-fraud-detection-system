import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SHAPExplainer:
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initializes TreeExplainer for Tree-based models (XGBoost/LightGBM/RF)
        """
        self.model = model
        self.feature_names = feature_names
        # Using TreeExplainer for tree models
        self.explainer = shap.TreeExplainer(model)

    def explain_instance(self, instance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes SHAP values for a single transaction instance.
        Returns a dictionary of top features contributing to the decision.
        """
        # Calculate shap values
        # For XGBoost, TreeExplainer returns a matrix of shape (n_instances, n_features) or (n_instances, n_features, 2) if multiclass
        shap_values = self.explainer(instance_df)
        
        # Check dimensional shape
        values = shap_values.values
        if len(values.shape) == 3:
            # Multi-class output, extract values for positive class (1)
            values = values[:, :, 1]
            base_value = shap_values.base_values[0][1]
        else:
            base_value = shap_values.base_values[0]

        # Extract features and their attributions safely
        attributions = []
        for i, feat in enumerate(self.feature_names):
            if feat in instance_df.columns and i < values.shape[1]:
                attributions.append({
                    "feature": feat,
                    "value": float(instance_df[feat].iloc[0]),
                    "shap_value": float(values[0, i])
                })
            
        # Sort by absolute SHAP value to find most influential features
        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Split positive and negative contributions
        pushing_fraud = [a for a in attributions if a["shap_value"] > 0]
        pushing_legit = [a for a in attributions if a["shap_value"] < 0]
        
        return {
            "base_value": float(base_value),
            "prediction_value": float(instance_df.shape[0]), # placeholder or raw prediction logit
            "top_fraud_contributors": pushing_fraud[:5],
            "top_legit_contributors": pushing_legit[:5]
        }
