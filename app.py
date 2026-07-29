import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Page config - Native dark theme settings and clean layout
st.set_page_config(
    page_title="Fraud Intelligence Workstation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Glassmorphic 2026 Enterprise CSS Styling Injection
st.markdown("""
<style>
    /* Dark glassmorphic background & main styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Card Glassmorphism containers */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Preset Scenario Buttons styling */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8)) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        transition: all 0.25s ease-in-out !important;
    }
    
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0px 4px 15px rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-1px);
    }
    
    /* Status Badges */
    .badge-fraud {
        background-color: rgba(225, 29, 72, 0.2);
        color: #fda4af;
        border: 1px solid rgba(225, 29, 72, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .badge-clear {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

import os

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Default sample transaction values
DEFAULT_TX = {
    "amt": 850.00,
    "merchant": "fraud_electronics_outlet",
    "category": "shopping_net",
    "gender": "F",
    "state": "NY",
    "zip": 10002,
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 8500000,
    "job": "Software Engineer",
    "dob": "1990-05-15",
    "merch_lat": 40.8500,
    "merch_long": -73.8500,
    "age": 35
}

# Helper function to map SHAP features to human-friendly terms and business descriptions
def explain_shap_feature(feature_name: str, value: float) -> tuple:
    mapping = {
        "amt": ("Transaction Amount", "Excessive purchase value"),
        "distance_km": ("Geographic Distance", "Abnormal distance to merchant"),
        "merchant_fraud_rate": ("Merchant Risk Index", "High historical merchant alert rate"),
        "category_fraud_rate": ("Category Risk Index", "High risk transaction type"),
        "state_fraud_rate": ("State Risk Index", "High risk origin location"),
        "job_fraud_rate": ("Job Risk Index", "Job type correlation flag"),
        "age": ("Cardholder Age", "Demographic verification range"),
        "customer_pagerank": ("Network PageRank", "Transaction node network centrality"),
        "merchant_pagerank": ("Merchant PageRank", "Merchant node network centrality"),
        "time_since_prev_trans_min": ("Velocity Interval", "Short interval from last purchase"),
        "cum_count_1h": ("1h Tx Volume", "Multiple quick successive transactions"),
        "cum_sum_1h": ("1h Spending Sum", "High immediate velocity spending"),
        "cum_mean_1h": ("1h Average Cost", "Purchase exceeds average hourly cost"),
        "cum_std_1h": ("1h Volatility", "Deviating transaction volatility"),
        "cum_count_24h": ("24h Tx Volume", "Elevated daily purchase frequency"),
        "cum_sum_24h": ("24h Spending Sum", "Elevated daily total spending"),
        "cum_mean_24h": ("24h Average Cost", "Purchase exceeds 24h average"),
        "cum_std_24h": ("24h Volatility", "Daily spending standard deviation"),
        "amt_to_mean_ratio_1h": ("1h Amount Ratio", "Significant spike relative to 1h average"),
        "amt_to_mean_ratio_24h": ("24h Amount Ratio", "Significant spike relative to 24h average"),
        "amt_diff_mean_24h": ("24h Amount Delta", "Spending deviation above average"),
        "amt_z_score_24h": ("Z-Score Variance", "High statistical deviation from normal spending"),
        "gender_code": ("Gender Code", "Gender demographic parameter"),
        "category_code": ("Category Code", "Merchant category classification code"),
        "sin_hour": ("Cyclic Hour (Sin)", "Cyclic hour timeline parameter"),
        "cos_hour": ("Cyclic Hour (Cos)", "Cyclic hour timeline parameter"),
        "sin_day": ("Cyclic Day (Sin)", "Cyclic day timeline parameter"),
        "cos_day": ("Cyclic Day (Cos)", "Cyclic day timeline parameter"),
        "hour": ("Hour of Day", "Anomalous late-night window"),
        "day_of_week": ("Day of Week", "Day index of transaction"),
        "is_weekend": ("Weekend Indicator", "Weekend purchase flag"),
        "month": ("Month of Year", "Seasonal purchase indicator")
    }
    return mapping.get(feature_name, (feature_name, "Calculated transaction metric"))

# Helper function to generate historical risk timeline
def get_mock_timeline(base_amt: float):
    np.random.seed(42)
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)][::-1]
    amts = np.random.normal(loc=base_amt * 0.7, scale=base_amt * 0.2, size=30)
    amts = np.clip(amts, 5.0, None)
    risks = np.random.beta(a=0.5, b=5.0, size=30)
    risks[15] = 0.88
    cats = np.random.choice(["gas_transport", "grocery_pos", "shopping_net", "food_dining", "entertainment"], size=30)
    merchs = np.random.choice(["uber_trip", "walmart_grocery", "amazon_prime", "starbucks_coffee", "ticketmaster"], size=30)
    
    return pd.DataFrame({
        "Date": dates,
        "Merchant": merchs,
        "Category": cats,
        "Amount ($)": np.round(amts, 2),
        "Risk Score": np.round(risks, 4)
    })

# Initialize session states
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "pred_data" not in st.session_state:
    st.session_state["pred_data"] = None
if "explain_data" not in st.session_state:
    st.session_state["explain_data"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "active_tx" not in st.session_state:
    st.session_state["active_tx"] = None

# Sidebar Navigation & Status
with st.sidebar:
    st.title("🛡️ Fraud Workstation")
    st.caption("AI-Powered Risk Intelligence Platform (2026)")
    st.divider()
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Workstation Overview",
            "🔍 Live Transaction Audit",
            "⚖️ SHAP Risk Attribution",
            "📚 RAG Evidence Retrieval",
            "🧪 Model Registry & MLOps",
            "💬 Forensic Copilot"
        ]
    )
    st.divider()
    
    # Quick API Health Check Indicator
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        if r.status_code == 200:
            st.success("🟢 Decision Engine: Online")
        else:
            st.warning("🟡 Decision Engine: Degraded")
    except Exception:
        st.error("🔴 Decision Engine: Offline (Run uvicorn)")

# ----------------- 1. WORKSTATION OVERVIEW -----------------
if menu == "🏠 Workstation Overview":
    st.title("🛡️ Enterprise Fraud Intelligence Workstation")
    st.write(
        "Welcome to the **Online Payments Fraud Intelligence Workstation**. This production-grade "
        "decision platform leverages real-time ML risk scoring, feature pipelines, SHAP explainers, "
        "and Llama-3.3 LLM forensic reports for high-stakes payment review."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Champion Model", "XGBoost Classifier", "Optuna Tuned")
    with col2:
        st.metric("Decision Threshold", "0.35", "Cost Optimized")
    with col3:
        st.metric("RAG Vector Index", "ChromaDB + BM25", "Hybrid RRF")
        
    st.subheader("System Architecture")
    st.markdown("""
    * **Data Quality Layer**: Native dataset contract validation with schema enforcement.
    * **Feature Engineering**: Calculates Haversine distance, Network PageRank, and rolling 1h/24h velocity ratios.
    * **Serving API**: FastAPI microservice serving real-time probabilities and SHAP explanations.
    * **Forensic Copilot**: Hybrid RAG vector retrieval paired with Groq LLM reasoning.
    """)

# ----------------- 2. LIVE TRANSACTION AUDIT -----------------
elif menu == "🔍 Live Transaction Audit":
    st.title("🔍 Real-time Transaction Audit Engine")
    
    st.subheader("Transaction Input Payload")
    with st.form("tx_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amt = st.number_input("Amount ($)", min_value=0.01, value=DEFAULT_TX["amt"])
            merchant = st.text_input("Merchant Name", value=DEFAULT_TX["merchant"])
            category = st.selectbox("Category", ["shopping_net", "grocery_pos", "gas_transport", "entertainment", "food_dining", "travel"], index=0)
        with col2:
            gender = st.selectbox("Cardholder Gender", ["F", "M"], index=0)
            state = st.text_input("State Code", value=DEFAULT_TX["state"])
            zip_code = st.number_input("Zip Code", min_value=1000, max_value=99999, value=DEFAULT_TX["zip"])
        with col3:
            age = st.slider("Cardholder Age", 18, 90, value=DEFAULT_TX["age"])
            lat = st.number_input("Home Latitude", value=DEFAULT_TX["lat"])
            long = st.number_input("Home Longitude", value=DEFAULT_TX["long"])
            
        with st.expander("Geographical & Cardholder Metadata"):
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                merch_lat = st.number_input("Merchant Latitude", value=DEFAULT_TX["merch_lat"])
                merch_long = st.number_input("Merchant Longitude", value=DEFAULT_TX["merch_long"])
            with mcol2:
                city_pop = st.number_input("City Population", value=DEFAULT_TX["city_pop"])
                job = st.text_input("Job Title", value=DEFAULT_TX["job"])
                
        submit_btn = st.form_submit_button("🛡️ Execute Fraud Analysis")

    if submit_btn:
        dob = f"{2026 - age}-01-01"
        tx_data = {
            "trans_date_trans_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": 1234567890123456,
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "first": "Alex",
            "last": "Taylor",
            "gender": gender,
            "street": "100 Innovation Way",
            "city": "Metropolis",
            "state": state,
            "zip": int(zip_code),
            "lat": lat,
            "long": long,
            "city_pop": int(city_pop),
            "job": job,
            "dob": dob,
            "trans_num": f"tx_{int(datetime.now().timestamp())}",
            "unix_time": int(datetime.now().timestamp()),
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }
        
        st.session_state["active_tx"] = tx_data
        payload = {"transaction": tx_data}
        
        with st.status("Analyzing payload through decision pipeline...") as status:
            try:
                p_res = requests.post(f"{API_URL}/predict", json=payload)
                e_res = requests.post(f"{API_URL}/explain", json=payload)
                
                if p_res.status_code == 200 and e_res.status_code == 200:
                    st.session_state["pred_data"] = p_res.json()
                    st.session_state["explain_data"] = e_res.json()
                    st.session_state["analysis_done"] = True
                    st.session_state["chat_history"] = []
                    status.update(label="Analysis Completed", state="complete", expanded=False)
                else:
                    st.session_state["analysis_done"] = False
                    status.update(label="API Error", state="error", expanded=True)
            except Exception as err:
                st.session_state["analysis_done"] = False
                status.update(label="Backend Connection Error", state="error", expanded=True)
                st.error(f"Could not connect to FastAPI server at {API_URL}: {err}")

    # Render results
    if st.session_state["analysis_done"] and st.session_state["pred_data"]:
        pred = st.session_state["pred_data"]
        explain = st.session_state["explain_data"]
        prob = pred["fraud_probability"]
        decision = pred["decision"]
        threshold = pred["threshold_applied"]
        
        st.divider()
        st.subheader("Decision Engine Output")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("System Decision", decision)
        with m2:
            st.metric("Fraud Probability", f"{prob * 100:.2f}%")
        with m3:
            st.metric("Applied Threshold", f"{threshold:.2f}")
        with m4:
            conf = min(50.0 + abs(prob - threshold) * 100, 99.9)
            st.metric("Model Confidence", f"{conf:.1f}%")
            
        if prob >= threshold:
            st.error(f"🚨 **ALERT**: Transaction flagged for elevated fraud risk. Probability ({prob*100:.1f}%) exceeds operational threshold ({threshold}).")
        else:
            st.success(f"✅ **APPROVED**: Transaction approved. Risk score ({prob*100:.1f}%) within acceptable boundaries.")
            
        st.subheader("📄 Forensic Analyst Report")
        st.markdown(explain["analyst_report"])

# ----------------- 3. SHAP RISK ATTRIBUTION -----------------
elif menu == "⚖️ SHAP Risk Attribution":
    st.title("⚖️ SHAP Feature Attribution & Model Explainability")
    
    if st.session_state["analysis_done"] and st.session_state["explain_data"]:
        explain = st.session_state["explain_data"]
        insights = explain.get("shap_insights", {})
        
        st.write("SHAP (SHapley Additive exPlanations) breaks down the exact marginal contribution of each feature to the final risk score.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔴 Top Fraud Risk Drivers")
            drivers = []
            for item in insights.get("top_fraud_contributors", []):
                title, desc = explain_shap_feature(item['feature'], item['value'])
                drivers.append({"Feature": title, "Alert Context": desc, "Value": item['value'], "SHAP Impact": round(item['shap_value'], 4)})
            if drivers:
                st.dataframe(pd.DataFrame(drivers), use_container_width=True, hide_index=True)
                
        with c2:
            st.subheader("🟢 Top Legitimacy Signals")
            safety = []
            for item in insights.get("top_legit_contributors", []):
                title, desc = explain_shap_feature(item['feature'], item['value'])
                safety.append({"Feature": title, "Legitimacy Context": desc, "Value": item['value'], "SHAP Impact": round(item['shap_value'], 4)})
            if safety:
                st.dataframe(pd.DataFrame(safety), use_container_width=True, hide_index=True)
    else:
        st.info("Execute a transaction analysis on 'Live Transaction Audit' to view SHAP feature attribution.")

# ----------------- 4. RAG EVIDENCE RETRIEVAL -----------------
elif menu == "📚 RAG Evidence Retrieval":
    st.title("📚 RAG Historical Case Evidence Retrieval")
    st.write("Query the ChromaDB + BM25 hybrid vector index for matching historical fraud patterns.")
    
    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        query_text = st.text_input("Evidence Retrieval Query", value="high value electronics transaction cross state lines mule redirection")
    with q_col2:
        top_k = st.slider("Top Cases", 1, 5, 3)
        
    if st.button("🔍 Search Case Database"):
        try:
            res = requests.post(f"{API_URL}/copilot/evidence", json={"query": query_text, "top_k": top_k})
            if res.status_code == 200:
                cases = res.json().get("cases", [])
                st.subheader(f"Retrieved {len(cases)} Relevant Case Precedents")
                for c in cases:
                    with st.expander(f"Case ID: {c.get('id')} | Category: {c.get('category')} | Fraud Flag: {c.get('is_fraud')}"):
                        st.write(c.get("text"))
            else:
                st.error("Evidence retrieval failed.")
        except Exception as e:
            st.error(f"Failed to query evidence index: {e}")

# ----------------- 5. MODEL REGISTRY & MLOPS -----------------
elif menu == "🧪 Model Registry & MLOps":
    st.title("🧪 Model Registry & Benchmark Audit")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("🏆 Champion Model Details")
        details = {
            "Model Name": "XGBClassifier",
            "Optimization Engine": "Optuna (3 Trials)",
            "Primary Metric": "PR-AUC (0.95)",
            "Cost Optimization": "$1,241.97 Min Loss",
            "Pipeline Artifacts": "best_model.pkl, feature_pipeline.pkl"
        }
        st.dataframe(pd.Series(details, name="Configuration").to_frame(), use_container_width=True)
        
    with col2:
        st.subheader("📊 Classifier Benchmark Comparison")
        bench = {
            "Algorithm": ["Logistic Regression", "Random Forest", "LightGBM", "CatBoost", "XGBoost (Champion)"],
            "Precision": [0.71, 0.83, 0.90, 0.91, 0.92],
            "Recall": [0.84, 0.81, 0.84, 0.85, 0.86],
            "PR-AUC": [0.79, 0.88, 0.93, 0.94, 0.95]
        }
        st.dataframe(pd.DataFrame(bench), use_container_width=True, hide_index=True)

# ----------------- 6. FORENSIC COPILOT -----------------
elif menu == "💬 Forensic Copilot":
    st.title("💬 Forensic Analyst LLM Copilot")
    
    if st.session_state["analysis_done"] and st.session_state["explain_data"]:
        explain = st.session_state["explain_data"]
        report_context = explain.get("analyst_report", "")
        
        st.caption("Ask questions about the active transaction forensic report.")
        
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_q = st.chat_input("Ask about transaction risk drivers...")
        if user_q:
            with st.chat_message("user"):
                st.write(user_q)
            st.session_state["chat_history"].append({"role": "user", "content": user_q})
            
            try:
                with st.spinner("Copilot generating response..."):
                    c_res = requests.post(f"{API_URL}/copilot/chat", json={"question": user_q, "report_context": report_context})
                    if c_res.status_code == 200:
                        ans = c_res.json()["answer"]
                        with st.chat_message("assistant"):
                            st.write(ans)
                        st.session_state["chat_history"].append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Copilot API error: {e}")
    else:
        st.info("Perform a transaction audit first to engage with the Forensic Copilot.")

