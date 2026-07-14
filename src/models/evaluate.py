import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.
    """
    # Use default 0.5 threshold for standard metrics
    y_pred = (y_prob >= 0.5).astype(int)
    
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc)
    }

def calculate_business_cost(y_true: np.ndarray, y_prob: np.ndarray, amt: np.ndarray, threshold: float) -> Tuple[float, Dict[str, Any]]:
    """
    Computes total business cost of decisions under a given threshold.
    Assumptions:
    - False Positive (legit transaction blocked): $10 (Customer support friction, manual review cost)
    - False Negative (fraud transaction missed): Full transaction amount (financial loss)
    - True Positive (fraud transaction caught): $2 (Small alert cost)
    - True Negative (legit transaction approved): $0
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Identify outcomes
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    tp = (y_pred == 1) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)
    
    # Financial costs
    fp_cost = fp.sum() * 10.0
    fn_cost = amt[fn].sum()
    tp_cost = tp.sum() * 2.0
    tn_cost = 0.0
    
    total_cost = fp_cost + fn_cost + tp_cost
    
    metrics = {
        "total_cost": float(total_cost),
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "true_positives": int(tp.sum()),
        "true_negatives": int(tn.sum()),
        "missed_fraud_amount": float(fn_cost),
        "blocked_legit_count": int(fp.sum())
    }
    
    return total_cost, metrics

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, amt: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the optimal decision threshold that minimizes business costs.
    """
    logger.info("Starting decision threshold optimization to minimize business costs...")
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    min_cost = float('inf')
    best_metrics = {}
    
    for th in thresholds:
        cost, metrics = calculate_business_cost(y_true, y_prob, amt, th)
        if cost < min_cost:
            min_cost = cost
            best_threshold = th
            best_metrics = metrics
            
    logger.info(f"Optimization complete. Best Threshold: {best_threshold:.4f} yielding Cost: ${min_cost:,.2f}")
    return float(best_threshold), best_metrics
