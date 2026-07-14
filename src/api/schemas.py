from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from src.data.schemas import TransactionSchema

class PredictionRequest(BaseModel):
    transaction: TransactionSchema

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    threshold_applied: float

class ExplanationResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    shap_insights: Dict[str, Any]
    analyst_report: str

class ChatRequest(BaseModel):
    question: str
    report_context: str

class ChatResponse(BaseModel):
    answer: str
