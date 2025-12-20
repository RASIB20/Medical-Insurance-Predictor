import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Load trained model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# Exact training feature list
FEATURES = [
    "arthritis",
    "mental_health",
    "diastolic_bp",
    "hypertension",
    "total_claims_paid",
    "smoker",
    "systolic_bp",
    "chronic_count",
    "age",
    "risk_score"
]

st.title("🩺 Medical Insurance Risk Predictor")
st.write("Predict whether a person is **High Risk** or **Low Risk**.")

# ---------------------------------
# Session state for feedback
# ---------------------------------
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(
        columns=["Name", "Usability", "Accuracy", "Suggestions"]
    )

# ---------------------------------
# Prediction Form
# ---------------------------------
with st.form("prediction_form"):
    st.subheader("📥 Enter Patient Information")

    arthritis = st.selectbox("Arthritis (0 = No, 1 = Yes)", [0, 1])
    mental_health = st.selectbox("Mental Health Issue (0 = No, 1 = Yes)", [0, 1])
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    smoker = st.selectbox("Smoker (0 = No, 1 = Yes)", [0, 1])

    diastolic_bp = st.number_input("Diastolic BP", value=80.0)
    systolic_bp = st.number_input("Systolic BP", value=120.0)
    total_claims_paid = st.number_input("Total Claims Paid", value=0.0)
    chronic_count = st.number_input("Chronic Conditions Count", min_value=0, value=1)
    age = st.number_input("Age", min_value=0.0, value=40.0)
    risk_score = st.number_input("Risk Score", value=0.5)

    submit = st.form_submit_button("🔍 Predict Risk")

# ---------------------------------
# Prediction Logic
# ---------------------------------
if submit:
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

    # 🔥 FORCE correct feature order
    input_df = input_df[FEATURES]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: HIGH RISK")
    else:
        st.success("✅ Prediction: LOW RISK")

# ---------------------------------
# Feedback Section
# ---------------------------------
st.divider()
st.subheader("📝 User Feedback")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    usability = st.selectbox("Usability", ["Excellent", "Good", "Average", "Poor"])
    accuracy = st.selectbox(
        "Prediction Accuracy",
        ["Very Accurate", "Accurate", "Not Sure", "Inaccurate"]
    )
    suggestions = st.text_area("Suggestions for Improvement")

    fb_submit = st.form_submit_button("Submit Feedback")

if fb_submit:
    new_row = pd.DataFrame([[name, usability, accuracy, suggestions]],
                           columns=st.session_state.feedback_data.columns)
    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, new_row], ignore_index=True
    )
    st.success("Thank you for your feedback!")

st.subheader("📊 Collected Feedback (Live Data)")
st.dataframe(st.session_state.feedback_data)
