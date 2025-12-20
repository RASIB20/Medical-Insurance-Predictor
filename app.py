import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------
# 1. App Configuration
# ----------------------------
st.set_page_config(page_title="Medical Insurance Risk Predictor", layout="wide")

# ----------------------------
# 2. Load Model & Scaler
# ----------------------------
@st.cache_resource
def load_assets():
    # Ensure these files are in the same folder as app.py
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Missing required files: {e}. Please upload 'random_forest_model.joblib' and 'scaler.joblib'.")

# ----------------------------
# 3. Input Form
# ----------------------------
st.title("🏥 Medical Insurance High-Risk Predictor")
st.write("Predict if a patient is **High Risk** based on medical history and claims data.")

st.subheader("🔍 Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 40)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        income = st.number_input("Annual Income ($)", value=50000.0)
        risk_score = st.number_input("Internal Risk Score (0-1)", value=0.5)
        smoker = st.selectbox("Smoker Status", ["No", "Yes", "Occasional"])
    
    with col2:
        systolic_bp = st.number_input("Systolic BP", value=120.0)
        diastolic_bp = st.number_input("Diastolic BP", value=80.0)
        total_claims_paid = st.number_input("Total Claims Paid ($)", value=500.0)
        chronic_count = st.number_input("Chronic Conditions Count", 0, 10, 1)
        # Binary flags
        arthritis = st.checkbox("Arthritis")
        mental_health = st.checkbox("Mental Health Issues")
        hypertension = st.checkbox("Hypertension")

    submit_btn = st.form_submit_button("Predict Risk")

# ----------------------------
# 4. Prediction Logic
# ----------------------------
if submit_btn:
    # Preprocessing: Match the training logic
    smoker_map = {'No': 1, 'Yes': 2, 'Occasional': 0}
    
    # Feature Engineering (BMI Category)
    bmi_cat = 0
    if bmi < 18.5: bmi_cat = 0
    elif bmi < 24.9: bmi_cat = 1
    elif bmi < 29.9: bmi_cat = 2
    else: bmi_cat = 3

    # Construct feature vector in the EXACT order the model was trained on
    input_data = {
        "arthritis": int(arthritis),
        "mental_health": int(mental_health),
        "diastolic_bp": diastolic_bp,
        "hypertension": int(hypertension),
        "total_claims_paid": total_claims_paid,
        "smoker": smoker_map[smoker],
        "systolic_bp": systolic_bp,
        "chronic_count": chronic_count,
        "age": age,
        "risk_score": risk_score
    }
    
    input_df = pd.DataFrame([input_data])

    # Scaling (Numerical features used for scaling in the notebook)
    # The notebook scaler was fit on: ['age', 'bmi', 'income', 'risk_score', 'annual_premium']
    # Note: We create a temporary DF to scale then extract the values needed
    scale_temp = pd.DataFrame([[age, bmi, income, risk_score, 0]], 
                              columns=['age', 'bmi', 'income', 'risk_score', 'annual_premium'])
    scaled_values = scaler.transform(scale_temp)[0]
    
    # Update scaled values back into our prediction DF
    input_df["age"] = scaled_values[0]
    input_df["risk_score"] = scaled_values[3]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk Patient (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk Patient (Confidence: {1 - probability:.2%})")

# ----------------------------
# 5. Persistent Feedback Section
# ----------------------------
st.divider()
st.subheader("📝 User Feedback")

FEEDBACK_FILE = "collected_feedback.csv"

# Load previous feedback from disk if it exists
if "feedback_data" not in st.session_state:
    if os.path.exists(FEEDBACK_FILE):
        try:
            st.session_state.feedback_data = pd.read_csv(FEEDBACK_FILE)
        except pd.errors.EmptyDataError:
            st.session_state.feedback_data = pd.DataFrame(columns=["Name", "Usability", "Accuracy", "Suggestions"])
    else:
        st.session_state.feedback_data = pd.DataFrame(columns=["Name", "Usability", "Accuracy", "Suggestions"])

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    usability = st.selectbox("Ease of Use", ["Excellent", "Good", "Average", "Poor"])
    accuracy = st.selectbox("Prediction Accuracy", ["Very Accurate", "Accurate", "Not Accurate"])
    suggestion = st.text_area("Suggestions for Improvement")
    feedback_submit = st.form_submit_button("Submit Feedback")

if feedback_submit:
    new_fb = pd.DataFrame([{"Name": name, "Usability": usability, "Accuracy": accuracy, "Suggestions": suggestion}])
    
    # Update Session State
    st.session_state.feedback_data = pd.concat([st.session_state.feedback_data, new_fb], ignore_index=True)
    
    # Write to local CSV immediately
    st.session_state.feedback_data.to_csv(FEEDBACK_FILE, index=False)
    st.success("Thank you! Feedback saved.")

# ----------------------------
# 6. Display Feedback Table
# ----------------------------
if not st.session_state.feedback_data.empty:
    st.subheader("📊 Feedback History (Persistent)")
    st.dataframe(st.session_state.feedback_data, use_container_width=True)
