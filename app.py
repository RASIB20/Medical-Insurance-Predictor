import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Medical Insurance Risk Predictor",
    layout="wide"
)

st.title("🏥 Medical Insurance High-Risk Predictor")
st.write("Predict whether a patient is **High Risk** based on medical and claims data.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# ----------------------------
# Input Form
# ----------------------------
st.subheader("🔍 Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        systolic_bp = st.number_input("Systolic BP", value=120.0)
        diastolic_bp = st.number_input("Diastolic BP", value=80.0)
        total_claims_paid = st.number_input("Total Claims Paid", value=0.0)
        risk_score = st.number_input("Risk Score", value=0.5)

    with col2:
        arthritis = st.selectbox("Arthritis", [0, 1])
        mental_health = st.selectbox("Mental Health Issue", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])
        smoker = st.selectbox("Smoker", [0, 1])
        chronic_count = st.number_input("Chronic Conditions Count", min_value=0, value=1)

    submit_btn = st.form_submit_button("Predict Risk")

# ----------------------------
# Prediction
# ----------------------------
if submit_btn:
    input_df = pd.DataFrame([{
        "arthritis": arthritis,
        "mental_health": mental_health,
        "diastolic_bp": diastolic_bp,
        "hypertension": hypertension,
        "total_claims_paid": total_claims_paid,
        "smoker": smoker,
        "systolic_bp": systolic_bp,
        "chronic_count": chronic_count,
        "age": age,
        "risk_score": risk_score
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk Patient (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk Patient (Confidence: {1 - probability:.2%})")

# ----------------------------
# Feedback Section
# ----------------------------
st.subheader("📝 User Feedback")

if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(
        columns=["Name", "Usability", "Accuracy", "Suggestions"]
    )

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    usability = st.selectbox("Ease of Use", ["Excellent", "Good", "Average", "Poor"])
    accuracy = st.selectbox("Prediction Accuracy", ["Very Accurate", "Accurate", "Not Accurate"])
    suggestion = st.text_area("Suggestions for Improvement")

    feedback_submit = st.form_submit_button("Submit Feedback")

if feedback_submit:
    new_feedback = {
        "Name": name,
        "Usability": usability,
        "Accuracy": accuracy,
        "Suggestions": suggestion
    }

    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, pd.DataFrame([new_feedback])],
        ignore_index=True
    )

    st.success("Thank you for your feedback!")

# ----------------------------
# Display Feedback Table
# ----------------------------
st.subheader("📊 Collected Feedback (Live Session)")
st.dataframe(st.session_state.feedback_data)
