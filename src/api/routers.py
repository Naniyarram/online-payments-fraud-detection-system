import os
import pandas as pd
import joblib
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import (
    PredictionRequest, PredictionResponse, ExplanationResponse,
    ChatRequest, ChatResponse, RetrievalRequest, RetrievalResponse
)
from src.utils.logger import get_logger
from src.features.build_features import FeaturePipeline
from src.models.explain import SHAPExplainer
from src.genai.db import HybridRetriever
from src.genai.copilot import FraudCopilot

logger = get_logger(__name__)

router = APIRouter()

# Global state loaded on startup
MODEL = None
PIPELINE = None
EXPLAINER = None
RETRIEVER = None
COPILOT = None
THRESHOLD = 0.35 # Default business optimized threshold

def load_artifacts():
    global MODEL, PIPELINE, EXPLAINER, RETRIEVER, COPILOT
    model_path = "./models/best_model.pkl"
    pipeline_path = "./models/feature_pipeline.pkl"
    
    # Init Retriever & Copilot
    RETRIEVER = HybridRetriever()
    COPILOT = FraudCopilot(RETRIEVER)
    
    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        logger.info("Loading model and feature pipeline artifacts...")
        MODEL = joblib.load(model_path)
        PIPELINE = joblib.load(pipeline_path)
        
        # Use dynamic feature names from pipeline if available, else fallback
        feature_names = getattr(PIPELINE, "feature_names", None)
        if not feature_names:
            feature_names = [
                'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
                'distance_km', 'age', 'hour', 'day_of_week', 'is_weekend', 'month',
                'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'merchant_fraud_rate',
                'category_fraud_rate', 'state_fraud_rate', 'job_fraud_rate',
                'customer_pagerank', 'merchant_pagerank', 'time_since_prev_trans_min',
                'cum_count_1h', 'cum_sum_1h', 'cum_mean_1h', 'cum_std_1h', 'cum_count_24h',
                'cum_sum_24h', 'cum_mean_24h', 'cum_std_24h', 'amt_to_mean_ratio_1h',
                'amt_to_mean_ratio_24h', 'amt_diff_mean_24h', 'amt_z_score_24h',
                'gender_code', 'category_code'
            ]
        EXPLAINER = SHAPExplainer(MODEL, feature_names)
    else:
        logger.warning("Artifacts not found! Using simulated mock prediction for testing/bootstrapping.")

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request for trans_num: {request.transaction.trans_num}")
    
    tx_dict = request.transaction.model_dump()
    df_raw = pd.DataFrame([tx_dict])
    
    try:
        if MODEL is None or PIPELINE is None:
            prob = 0.85 if request.transaction.amt > 800 else 0.01
            decision = "ALERT - HIGH FRAUD RISK" if prob >= THRESHOLD else "APPROVE"
            return PredictionResponse(
                transaction_id=request.transaction.trans_num,
                fraud_probability=prob,
                decision=decision,
                threshold_applied=THRESHOLD
            )
            
        df_feat = PIPELINE.transform(df_raw)
        prob = float(MODEL.predict_proba(df_feat)[0, 1])
        decision = "ALERT - HIGH FRAUD RISK" if prob >= THRESHOLD else "APPROVE"
        
        return PredictionResponse(
            transaction_id=request.transaction.trans_num,
            fraud_probability=prob,
            decision=decision,
            threshold_applied=THRESHOLD
        )
    except Exception as e:
        logger.error(f"Error handling prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/explain", response_model=ExplanationResponse)
async def explain(request: PredictionRequest):
    logger.info(f"Received explanation request for trans_num: {request.transaction.trans_num}")
    
    tx_dict = request.transaction.model_dump()
    df_raw = pd.DataFrame([tx_dict])
    
    try:
        if MODEL is None or PIPELINE is None:
            mock_insights = {
                "base_value": 0.02,
                "prediction_value": 1.0,
                "top_fraud_contributors": [
                    {"feature": "amt", "value": float(request.transaction.amt), "shap_value": 0.45},
                    {"feature": "distance_km", "value": 12.5, "shap_value": 0.12}
                ],
                "top_legit_contributors": [
                    {"feature": "age", "value": 42.0, "shap_value": -0.05}
                ]
            }
            prob = 0.85 if request.transaction.amt > 800 else 0.01
            report = "SIMULATED REPORT: High amount detected. Standard validation triggered."
            if COPILOT:
                report = COPILOT.generate_analyst_report(tx_dict, mock_insights, prob)
                
            return ExplanationResponse(
                transaction_id=request.transaction.trans_num,
                fraud_probability=prob,
                shap_insights=mock_insights,
                analyst_report=report
            )
            
        df_feat = PIPELINE.transform(df_raw)
        prob = float(MODEL.predict_proba(df_feat)[0, 1])
        
        insights = EXPLAINER.explain_instance(df_feat)
        report = COPILOT.generate_analyst_report(tx_dict, insights, prob)
        
        return ExplanationResponse(
            transaction_id=request.transaction.trans_num,
            fraud_probability=prob,
            shap_insights=insights,
            analyst_report=report
        )
    except Exception as e:
        logger.error(f"Error handling explanation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

@router.post("/copilot/chat", response_model=ChatResponse)
async def copilot_chat(request: ChatRequest):
    try:
        if COPILOT is None:
            return ChatResponse(answer="Copilot is offline (missing artifacts/keys).")
            
        answer = COPILOT.chat(request.question, request.report_context)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error handling chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/copilot/evidence", response_model=RetrievalResponse)
async def copilot_evidence(request: RetrievalRequest):
    try:
        if RETRIEVER is None:
            return RetrievalResponse(query=request.query, cases=[])
            
        cases = RETRIEVER.retrieve(request.query, top_k=request.top_k or 3)
        return RetrievalResponse(query=request.query, cases=cases)
    except Exception as e:
        logger.error(f"Error retrieving evidence: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

