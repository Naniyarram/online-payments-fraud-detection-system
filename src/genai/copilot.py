import os
from groq import Groq
from typing import Dict, Any, List
from src.utils.logger import get_logger
from src.genai.db import HybridRetriever

logger = get_logger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class FraudCopilot:
    def __init__(self, retriever: HybridRetriever):
        logger.info("Initializing Fraud Investigation Copilot...")
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY environment variable is not set. Copilot completions will be unavailable.")
            self.client = None
        else:
            self.client = Groq(api_key=GROQ_API_KEY)
        self.retriever = retriever
        self.model_name = "llama-3.3-70b-versatile"

    def generate_analyst_report(self, transaction_data: Dict[str, Any], shap_insights: Dict[str, Any], fraud_prob: float) -> str:
        """
        Generates a highly concise, 150-250 word forensic analyst report based strictly on
        model decisions and SHAP attributions.
        """
        if not self.client:
            return "### Executive Summary\nGenAI Analyst Copilot is offline. Please configure GROQ_API_KEY in your environment.\n\n### Top Risk Drivers\n- Feature attribution weights are available under the Risk Attribution tab.\n\n### Similar Historical Cases\n- Context matches are available under the Evidence & Similar Cases tab.\n\n### Recommended Action\n- Review transaction metrics manually."

        logger.info("Generating concise natural language analyst report...")
        
        # Retrieve similar historical fraud cases
        query = f"Transaction for {transaction_data.get('category')} of {transaction_data.get('amt')} in state {transaction_data.get('state')}"
        past_cases = self.retriever.retrieve(query, top_k=2)
        past_cases_context = "\n".join([f"- Case {c['id']}: {c['text']}" for c in past_cases])

        # Prepare SHAP inputs
        top_fraud = ", ".join([f"{item['feature']} (weight: {item['shap_value']:.3f})" for item in shap_insights.get("top_fraud_contributors", [])])
        top_legit = ", ".join([f"{item['feature']} (weight: {item['shap_value']:.3f})" for item in shap_insights.get("top_legit_contributors", [])])

        prompt = f"""You are a Fraud Analyst Copilot. Draft a concise forensic summary of this credit card transaction.
Strictly adhere to the 150-250 word limit. Do not include introductory or concluding conversational filler.

DATA:
- Amount: ${transaction_data.get('amt')}
- Category: {transaction_data.get('category')}
- Customer State: {transaction_data.get('state')}
- Distance to Merchant: {transaction_data.get('distance_km', 'N/A')} km
- Fraud Probability: {fraud_prob * 100:.2f}%
- Decision: {'ALERT - RISK HIGH' if fraud_prob >= 0.5 else 'APPROVE - RISK LOW'}

SHAP ATTRIBUTIONS:
- Fraud Drivers: {top_fraud}
- Safety Signals: {top_legit}

HISTORICAL CASES CONTEXT:
{past_cases_context}

FORMAT:
### Executive Summary
[Brief risk status and recommendation]

### Top Risk Drivers
[Key triggers from SHAP weights]

### Similar Historical Cases
[Compare transaction with retrieved context cases]

### Recommended Action
[Actionable guidelines for analyst]
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a concise, factual fraud analytics assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return f"Error generating report: {str(e)}"

    def chat(self, user_question: str, report_context: str) -> str:
        if not self.client:
            return "Copilot chat services are currently offline. Please set the GROQ_API_KEY environment variable."
            
        logger.info(f"Answering analyst question: '{user_question}'")
        prompt = f"""You are a Fraud Analyst Copilot. Answer the analyst's question regarding an ongoing investigation.
Use the following report context to answer the question. If you cannot answer it using the context, state that clearly.

REPORT CONTEXT:
\"\"\"
{report_context}
\"\"\"

QUESTION: {user_question}
"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a concise, factual assistant specializing in credit card fraud analysis."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.2
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API for chat: {str(e)}")
            return f"Error: {str(e)}"
