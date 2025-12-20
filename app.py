import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
st.set_page_config(page_title="Medical Risk Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Ensure 'random_forest_model.joblib' and 'scaler.joblib' exist.")

smoker_map = {'No': 1, 'Yes': 2, 'Occasional': 0}
st.title("🏥 Health Risk Prediction System")
st.write("Predict whether a patient is **High Risk** based on medical data.")

with st.form("prediction_form"):
    st.subheader("🔍 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 40)
        bmi = st.number_input("BMI", value=25.0)
        income = st.number_input("Annual Income", value=50000.0)
        risk_score = st.number_input("Internal Risk Score", value=0.5)
        smoker = st.selectbox("Smoker Status", list(smoker_map.keys()))
        chronic_count = st.number_input("Chronic Conditions Count", 0, 10, 1)

    with col2:
        systolic_bp = st.number_input("Systolic BP", value=120.0)
        diastolic_bp = st.number_input("Diastolic BP", value=80.0)
        total_claims_paid = st.number_input("Total Claims Paid", value=500.0)
        arthritis = st.checkbox("Arthritis")
        mental_health = st.checkbox("Mental Health Condition")
        hypertension = st.checkbox("Hypertension")

    submit_btn = st.form_submit_button("Predict Risk")
    
if submit_btn:
    # Feature Engineering (BMI Category) as done in training
    bmi_cat = 0
    if bmi < 18.5: bmi_cat = 0
    elif bmi < 24.9: bmi_cat = 1
    elif bmi < 29.9: bmi_cat = 2
    else: bmi_cat = 3
    input_dict = {
        "arthritis": int(arthritis),
        "mental_health": int(mental_health),
        "diastolic_bp": diastolic_bp,
        "hypertension": int(hypertension),
        "total_claims_paid": total_claims_paid,
        "smoker": smoker_map[smoker],
        "systolic_bp": systolic_bp,
        "chronic_count": chronic_count,
        "age": age,
        "risk_score": risk_score,
        "bmi": bmi,
        "income": income,
        "bmi_category": bmi_cat
    }
    
    input_df = pd.DataFrame([input_dict])

    # Scaling numerical columns
    scale_cols = ['age', 'bmi', 'income', 'risk_score']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])
    expected_cols = ["arthritis", "mental_health", "diastolic_bp", "hypertension", 
                     "total_claims_paid", "smoker", "systolic_bp", "chronic_count", 
                     "age", "risk_score"]
    
    final_input = input_df[expected_cols]

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk Patient (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk Patient (Confidence: {1 - probability:.2%})")

st.divider()
st.subheader("📝 User Feedback")

FEEDBACK_FILE = "collected_feedback.csv"

# Load previous feedback from CSV or initialize session state
if "feedback_data" not in st.session_state:
    if os.path.exists(FEEDBACK_FILE):
        st.session_state.feedback_data = pd.read_csv(FEEDBACK_FILE)
    else:
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

    # Update state and SAVE to file
    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, pd.DataFrame([new_feedback])],
        ignore_index=True
    )
    st.session_state.feedback_data.to_csv(FEEDBACK_FILE, index=False)
    st.success("Feedback saved successfully!")
    
if not st.session_state.feedback_data.empty:
    st.subheader("📊 Collected Feedback History")
    st.dataframe(st.session_state.feedback_data, use_container_width=True)
