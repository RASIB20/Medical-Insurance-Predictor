import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

st.title("Health Risk Prediction System")
st.write("Predict whether a user is **High Risk** based on health & insurance data.")

if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(
        columns=["Name", "Usability", "Accuracy", "Suggestions"]
    )

with st.form("prediction_form"):
    st.subheader("📥 Enter Patient & Policy Details")

    age = st.number_input("Age", 0, 120, 30)
    income = st.number_input("Income", value=50000.0)
    household_size = st.number_input("Household Size", 1, 10, 1)
    dependents = st.number_input("Dependents", 0, 10, 0)
    bmi = st.number_input("BMI", value=25.0)
    visits_last_year = st.number_input("Visits Last Year", 0, 20, 0)
    hospitalizations_last_3yrs = st.number_input("Hospitalizations (3 yrs)", 0, 10, 0)
    days_hospitalized_last_3yrs = st.number_input("Days Hospitalized (3 yrs)", 0, 100, 0)
    medication_count = st.number_input("Medication Count", 0, 20, 0)
    systolic_bp = st.number_input("Systolic BP", value=120.0)
    diastolic_bp = st.number_input("Diastolic BP", value=80.0)
    ldl = st.number_input("LDL", value=100.0)
    hba1c = st.number_input("HbA1c", value=5.5)
    deductible = st.number_input("Deductible", value=500)
    copay = st.number_input("Copay", value=20)
    policy_term_years = st.number_input("Policy Term (Years)", 1, 10, 1)
    policy_changes_last_2yrs = st.number_input("Policy Changes (2 yrs)", 0, 5, 0)
    provider_quality = st.number_input("Provider Quality", value=0.9)
    risk_score = st.number_input("Risk Score", value=0.3)
    annual_premium = st.number_input("Annual Premium", value=3000.0)
    monthly_premium = st.number_input("Monthly Premium", value=250.0)
    claims_count = st.number_input("Claims Count", 0, 10, 0)
    avg_claim_amount = st.number_input("Avg Claim Amount", value=500.0)
    total_claims_paid = st.number_input("Total Claims Paid", value=0.0)
    chronic_count = st.number_input("Chronic Conditions", 0, 10, 0)

    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    asthma = st.selectbox("Asthma", [0, 1])
    copd = st.selectbox("COPD", [0, 1])
    cardiovascular_disease = st.selectbox("Cardiovascular Disease", [0, 1])
    cancer_history = st.selectbox("Cancer History", [0, 1])
    kidney_disease = st.selectbox("Kidney Disease", [0, 1])
    liver_disease = st.selectbox("Liver Disease", [0, 1])
    arthritis = st.selectbox("Arthritis", [0, 1])
    mental_health = st.selectbox("Mental Health Issues", [0, 1])
    had_major_procedure = st.selectbox("Major Procedure History", [0, 1])

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "income": income,
        "household_size": household_size,
        "dependents": dependents,
        "bmi": bmi,
        "visits_last_year": visits_last_year,
        "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
        "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
        "medication_count": medication_count,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "ldl": ldl,
        "hba1c": hba1c,
        "deductible": deductible,
        "copay": copay,
        "policy_term_years": policy_term_years,
        "policy_changes_last_2yrs": policy_changes_last_2yrs,
        "provider_quality": provider_quality,
        "risk_score": risk_score,
        "annual_premium": annual_premium,
        "monthly_premium": monthly_premium,
        "claims_count": claims_count,
        "avg_claim_amount": avg_claim_amount,
        "total_claims_paid": total_claims_paid,
        "chronic_count": chronic_count,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "asthma": asthma,
        "copd": copd,
        "cardiovascular_disease": cardiovascular_disease,
        "cancer_history": cancer_history,
        "kidney_disease": kidney_disease,
        "liver_disease": liver_disease,
        "arthritis": arthritis,
        "mental_health": mental_health,
        "had_major_procedure": had_major_procedure
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk Patient")
    else:
        st.success("✅ Low Risk Patient")

st.divider()
st.subheader("📝 User Feedback Form")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    usability = st.selectbox("Usability", ["Excellent", "Good", "Average", "Poor"])
    accuracy = st.selectbox("Prediction Accuracy", ["Very Accurate", "Accurate", "Not Sure", "Inaccurate"])
    suggestions = st.text_area("Suggestions for Improvement")

    fb_submit = st.form_submit_button("Submit Feedback")

if fb_submit:
    new_row = pd.DataFrame([[name, usability, accuracy, suggestions]],
                           columns=st.session_state.feedback_data.columns)
    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, new_row], ignore_index=True
    )
    st.success("Thank you for your feedback!")

st.subheader("📊 Collected Feedback (Live Dataset)")
st.dataframe(st.session_state.feedback_data)
