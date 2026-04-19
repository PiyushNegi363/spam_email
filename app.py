import streamlit as st
import pandas as pd
import os
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spam Email Classifier Pro",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #dee2e6;
        padding: 15px;
        font-size: 16px;
    }
    
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .result-container {
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    
    .example-btn {
        margin-bottom: 0.5rem;
        display: block;
        width: 100%;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
@st.cache_resource
def get_pipeline():
    try:
        return PredictionPipeline(load_models=True)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        return None

def load_metrics():
    # Helper to load metrics from the hardcoded output path for now
    metrics_path = "outputs/2025-12-25_14-02-05/observations/best_model_info.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-header">📧 Classifier Settings</p>', unsafe_allow_html=True)
    st.info("System: Production Ready\n\nModel: SVM (Linear)")
    
    st.divider()
    st.markdown("**Quick Examples**")
    st.caption("Click to load sample text")
    
    examples = {
        "Lottery Scam": "Congratulations! You have won a cash prize of $1,000,000. To claim your prize, click the link below and confirm your details. Act fast!",
        "Account Security": "Hi Alex, we noticed a login to your account from a new device in London, UK. If this was not you, please secure your account here.",
        "Meeting Request": "Hi Team, can we schedule a meeting tomorrow at 10 AM to discuss the project status review? Thanks, John.",
        "Phishing Link": "Dear customer, your bank account is temporarily locked. Please login to https://secure-bank-login.com to verify your identity."
    }
    
    for name, text in examples.items():
        if st.button(name, key=f"ex_{name}", use_container_width=True):
            st.session_state.email_input = text

    st.divider()
    st.markdown("**About this Project**")
    st.write("This tool uses an SVM model trained on the SMS/Email Spam Collection dataset to classify messages with high precision.")
    st.caption("v1.2.0 | Reliability Audit Completed")

# --- MAIN APP ---
def main():
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""

    st.title("📧 Spam Email Classifier Pro")
    st.markdown("Enter an email body below to evaluate its safety profile using our advanced machine learning engine.")

    tab1, tab2, tab3 = st.tabs(["🔍 Classifier", "📊 Model Insights", "📖 How it Works"])

    with tab1:
        email_text = st.text_area(
            "Email Content:",
            value=st.session_state.email_input,
            height=250,
            placeholder="Paste the email content here or choose an example from the sidebar...",
            help="Maximum 5000 characters for optimal analysis.",
            key="main_text_area" # Added key to avoid duplicate widget error
        )
        
        col1, col2, col3 = st.columns([1.5, 1, 3])
        with col1:
            classify_btn = st.button("🚀 Analyze Email", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.email_input = ""
                st.rerun()

        if classify_btn:
            if not email_text.strip():
                st.warning("⚠️ Please provide some content to analyze.")
            elif len(email_text) > 5000:
                st.error("❌ Input exceeds the 5000 character limit.")
            else:
                with st.spinner("Decoding language patterns and evaluating risk factors..."):
                    pipeline = get_pipeline()
                    if pipeline:
                        try:
                            result = pipeline.predict_single_email(email_text)
                            prediction = result["prediction"]
                            confidence = result.get("confidence", 0)

                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            
                            res_col1, res_col2 = st.columns([2, 1])
                            
                            with res_col1:
                                if prediction == "Spam":
                                    st.subheader("🚨 Risk Assessment: **HIGH (Spam Detected)**")
                                    st.markdown("This message exhibits patterns commonly found in phishing or unsolicited promotional content.")
                                else:
                                    st.subheader("✅ Risk Assessment: **LOW (Safe / Ham)**")
                                    st.markdown("This message appears to be a legitimate communication.")
                            
                            with res_col2:
                                if confidence > 0:
                                    st.metric("Confidence Score", f"{confidence:.1f}%")
                                    st.progress(confidence / 100)
                                else:
                                    st.metric("Classification", prediction)
                                    st.caption("Linear SVM Confidence: Fixed")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if prediction == "Spam":
                                st.toast("High Risk Detected!", icon="🚨")
                            else:
                                st.toast("Email appears safe.", icon="✅")
                                
                        except Exception as e:
                            st.error("An error occurred during classification. Please check the logs.")
                            logger.error(f"Prediction error: {str(e)}")
                    else:
                        st.error("System Error: Could not load the classification engine.")

    with tab2:
        st.header("Model Performance Dashboard")
        st.markdown("Evaluation metrics for the currently deployed **Support Vector Machine** model.")
        
        metrics_df = load_metrics()
        if metrics_df is not None:
            # Transpose or pivot for better display
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            # Find values by attribute name
            def get_val(attr):
                try:
                    val = metrics_df[metrics_df['Attribute'] == attr]['Value'].values[0]
                    return f"{float(val)*100:.1f}%"
                except:
                    # Return raw value if not a float
                    return metrics_df[metrics_df['Attribute'] == attr]['Value'].values[0]

            m_col1.metric("Accuracy", get_val("Accuracy"))
            m_col2.metric("F1-Score", get_val("F1-Score"))
            m_col3.metric("Precision", get_val("Precision"))
            m_col4.metric("CV Score", get_val("CV Score"))
            
            st.divider()
            st.subheader("Training Observations")
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        else:
            st.warning("Model metrics data not found. Please run the training pipeline.")

    with tab3:
        st.header("Behind the Scenes")
        st.markdown("""
        ### Our Approach
        1. **Text Normalization**: Every input is cleaned, normalized, and stripped of HTML/special characters to focus on semantic content.
        2. **TF-IDF Vectorization**: We convert words into numerical importance scores, capturing the essence of 'spammy' vs 'normal' vocabulary.
        3. **SVM Classification**: A Support Vector Machine with a linear kernel finds the optimal boundary between Spam and Ham in a high-dimensional word space.
        
        ### Security & Privacy
        - All processing is done in-memory.
        - No email content is stored or transmitted outside of this session.
        - Built-in protection against formula injection and extremely long inputs.
        """)
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_001.png", caption="Visualizing Linear SVM Boundaries", width=600)

if __name__ == "__main__":
    main()
