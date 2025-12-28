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
# 2. Load Assets (Model & Scaler)
# ----------------------------
@st.cache_resource
def load_assets():
    # Ensure these files are in the same directory as app.py
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Make sure 'random_forest_model.joblib' and 'scaler.joblib' are uploaded.")

# ----------------------------
# 3. Categorical Mappings (From Notebook Encoders)
# ----------------------------
smoker_map = {'No': 1, 'Yes': 2, 'Occasional': 0}

# ----------------------------
# 4. Input Form
# ----------------------------
st.title("🏥 Medical Insurance High-Risk Predictor")
st.write("Predict whether a patient is **High Risk** based on medical and policy data.")

with st.form("prediction_form"):
    st.subheader("🔍 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 40)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        income = st.number_input("Annual Income ($)", value=50000.0)
        risk_score = st.number_input("Internal Risk Score", value=0.5)
        smoker = st.selectbox("Smoker Status", list(smoker_map.keys()))
        chronic_count = st.number_input("Chronic Conditions Count", 0, 10, 1)

    with col2:
        systolic_bp = st.number_input("Systolic BP", value=120.0)
        diastolic_bp = st.number_input("Diastolic BP", value=80.0)
        total_claims_paid = st.number_input("Total Claims Paid ($)", value=0.0)
        arthritis = st.checkbox("Arthritis")
        mental_health = st.checkbox("Mental Health Issue")
        hypertension = st.checkbox("Hypertension")

    submit_btn = st.form_submit_button("Predict Risk")

# ----------------------------
# 5. Prediction Logic
# ----------------------------
if submit_btn:
    # A. Feature Engineering (BMI Category)
    bmi_cat = 0
    if bmi < 18.5: bmi_cat = 0
    elif bmi < 24.9: bmi_cat = 1
    elif bmi < 29.9: bmi_cat = 2
    else: bmi_cat = 3

    # B. Construct Dataframe (Must contain all 10 features used in notebook X_train)
    # Order: arthritis, mental_health, diastolic_bp, hypertension, total_claims_paid, smoker, systolic_bp, chronic_count, age, risk_score
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

    # C. Scaling: The scaler was fit on 5 columns. We must recreate that structure to transform.
    # Scaler cols: ['age', 'bmi', 'income', 'risk_score', 'annual_premium']
    scale_temp = pd.DataFrame([[age, bmi, income, risk_score, 0]], 
                              columns=['age', 'bmi', 'income', 'risk_score', 'annual_premium'])
    scaled_values = scaler.transform(scale_temp)[0]
    
    # Update scaled values back into our prediction DF (index 0 is age, index 3 is risk_score)
    input_df["age"] = scaled_values[0]
    input_df["risk_score"] = scaled_values[3]

    # D. Run Prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk Patient (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ Low Risk Patient (Confidence: {1 - probability:.2%})")
    except ValueError as e:
        st.error(f"Model Feature Mismatch: {e}")

# ----------------------------
# 6. Persistent Feedback Section (Google Sheets Version)
# ----------------------------
from streamlit_gsheets import GSheetsConnection

st.divider()
st.subheader("📝 User Feedback")

# 1. Establish Connection
conn = st.connection("gsheets", type=GSheetsConnection)

# 2. Fetch Existing Data
try:
    # ttl=5 ensures it refreshes data from the sheet every 5 seconds so you see others' feedback
    feedback_df = conn.read(worksheet="Sheet1", usecols=[0, 1, 2, 3], ttl=5)
    
    # Handle case where sheet is empty (only headers) or None
    if feedback_df is None:
         feedback_df = pd.DataFrame(columns=["Name", "Usability", "Accuracy", "Suggestions"])
except Exception as e:
    st.error("Could not load feedback data. Check your Google Sheets connection.")
    feedback_df = pd.DataFrame(columns=["Name", "Usability", "Accuracy", "Suggestions"])

# 3. Form Input
with st.form("feedback_form"):
    name = st.text_input("Your Name")
    usability = st.selectbox("Ease of Use", ["Excellent", "Good", "Average", "Poor"])
    accuracy = st.selectbox("Prediction Accuracy", ["Very Accurate", "Accurate", "Not Accurate"])
    suggestion = st.text_area("Suggestions for Improvement")

    feedback_submit = st.form_submit_button("Submit Feedback")

# 4. Handle Submission
if feedback_submit:
    if name and suggestion:
        # Create new row
        new_feedback = pd.DataFrame([
            {"Name": name, "Usability": usability, "Accuracy": accuracy, "Suggestions": suggestion}
        ])
        
        # Append to existing data
        updated_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        
        # Update Google Sheet
        try:
            conn.update(worksheet="Sheet1", data=updated_df)
            st.success("Thank you! Feedback saved forever.")
            
            # Rerun to show the new data immediately
            st.rerun()
        except Exception as e:
            st.error(f"Error saving to Google Sheets: {e}")
    else:
        st.warning("Please fill in your name and suggestions.")

# 5. Display Feedback History
if not feedback_df.empty:
    st.subheader("📊 Collected Feedback History")
    # Show last 5 entries first
    st.dataframe(feedback_df.tail(5).iloc[::-1], use_container_width=True)

