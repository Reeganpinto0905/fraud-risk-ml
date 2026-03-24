import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

# Add src to the path so we can import predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.predict import predict_transaction, FRAUD_THRESHOLD
except ImportError:
    pass

# ---- Configure Page ---- 
st.set_page_config(page_title="Fraud Risk ML Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2F;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #4ade80;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'creditcard.csv'))
    if os.path.exists(data_path):
        return pd.read_csv(data_path, nrows=50000)
    return None

def check_models_exist():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    if not os.path.exists(os.path.join(models_dir, 'fraud_model.pkl')):
        return False
    return True

def render_metric(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=60)
    st.sidebar.title("Fraud Guard AI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation Menu", ["📊 Dataset Overview & EDA", "🔮 Real-Time Predictor"])
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: **Online** 🟢")
    
    if page == "📊 Dataset Overview & EDA":
        st.title("Dataset Overview & EDA")
        st.markdown("Analyze the latest trends and statistics from our transaction database.")
        st.markdown("---")
        
        with st.spinner("Crunching numbers..."):
            df = load_data()
            
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric("Total Transactions", f"{len(df):,}")
            with col2:
                render_metric("Detected Fraud", f"{len(df[df['Class'] == 1]):,}")
            with col3:
                rate = len(df[df['Class'] == 1]) / len(df)
                render_metric("Current Fraud Rate", f"{rate:.3%}")
            
            st.markdown("### Transaction Data Distribution")
            
            # Layout for charts
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Class Imbalance")
                # Plotly is native to Streamlit but let's use seaborn
                fig, ax = plt.subplots(figsize=(5,5))
                sns.countplot(data=df, x='Class', ax=ax, palette=['#4ade80', '#ef4444'])
                ax.set_title("Valid (0) vs. Fraud (1)", color='white')
                ax.set_facecolor('#0E1117')
                fig.patch.set_facecolor('#0E1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                st.pyplot(fig)
            
            with c2:
                st.subheader("Recent Transactions Logs")
                st.dataframe(df[['Time', 'Amount', 'Class']].head(12), use_container_width=True)
                
        else:
            st.warning("Dataset not found at `data/raw/creditcard.csv`.")
            
    elif page == "🔮 Real-Time Predictor":
        st.title("Real-Time Fraud Predictor 🔮")
        st.markdown("Simulate a transaction and let the **Random Forest AI** assess the risk.")
        st.markdown("---")
        
        models_ready = check_models_exist()
        if not models_ready:
            st.error("🚨 AI Models not found. Please train the model by running `python src/train.py`.")
            return
            
        with st.container():
            st.markdown("### Input Transaction Features")
            col1, col2 = st.columns(2)
            
            with col1:
                description = st.text_input("📝 Description", value="Transfer to international merchant")
                amount = st.number_input("💵 Amount ($)", value=1500.0, step=50.0)
                time_val = st.number_input("⏱ Time Offset", value=12000)
                
            with col2:
                v1 = st.slider("🧬 PCA Feature V1", -5.0, 5.0, -1.2)
                v2 = st.slider("🧬 PCA Feature V2", -5.0, 5.0, 0.4)
                v3 = st.slider("🧬 PCA Feature V3", -5.0, 5.0, -2.1)
                
            st.info("💡 For demonstration, PCA features V4 through V28 will be auto-filled as 0.0.")
                
            if st.button("Scan Transaction Now", use_container_width=True, type="primary"):
                num_features = [0.0] * 30
                num_features[0] = time_val
                num_features[1] = v1
                num_features[2] = v2
                num_features[3] = v3
                num_features[-1] = amount 
                
                with st.spinner("Neural network analyzing patterns..."):
                    try:
                        risk_score = predict_transaction(num_features, description)
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Result")
                        
                        # Custom styled alert box based on risk
                        threshold = getattr(sys.modules.get('src.predict'), 'FRAUD_THRESHOLD', 0.3)
                        
                        if risk_score >= threshold:
                            st.error(f"### 🚨 HIGH RISK DETECTED: {risk_score:.1%}")
                            st.write("Pattern matches known fraudulent behavior. Transaction blocked.")
                            st.progress(risk_score)
                        else:
                            st.success(f"### ✅ TRANSACTION APPROVED: {risk_score:.1%}")
                            st.write("Standard activity profile recognized. Risk is within acceptable bounds.")
                            st.progress(risk_score)
                            
                    except Exception as e:
                        st.error(f"Prediction Engine Error: {e}")

if __name__ == "__main__":
    main()
