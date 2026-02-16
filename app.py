import streamlit as st
from src.pipeline.prediction_pipeline import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

st.title("📧 Spam Email Classifier")
st.markdown("Classify emails as **Spam** or **Ham** (Clean) using Machine Learning.")

st.header("Check a Single Email")
email_text = st.text_area(
    "Paste the email content here:",
    height=200,
    placeholder="Dear friend, I have a business proposal..."
)

if st.button("Classify Email", type="primary"):
    if email_text.strip():
        with st.spinner("Analyzing..."):
            try:
                result = pipeline.predict_single_email(email_text)
                prediction = result["prediction"]
                confidence = result.get("confidence", 0)

                if prediction == "Spam":
                    st.error("🚨 This email is **SPAM**")
                else:
                    st.success("✅ This email is **HAM** (Safe)")

                if confidence:
                    st.info(f"Confidence Score: {confidence:.1f}%")
            except Exception as e:
                st.error(f"Error analyzing email: {str(e)}")
    else:
        st.warning("Please enter some text to classify.")
