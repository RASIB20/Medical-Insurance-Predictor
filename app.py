import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load trained model and preprocessing tools
@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_assets()

# Manual mapping for LabelEncoder (based on your notebook logic)
sex_map = {'Female': 0, 'Male': 1}
region_map = {'North': 1, 'South': 2, 'East': 0, 'West': 3}
urban_rural_map = {'Urban': 2, 'Suburban': 1, 'Rural': 0}
education_map = {'No HS': 4, 'HS': 2, 'Some College': 5, 'Bachelors': 0, 'Masters': 3, 'Doctorate': 1}
marital_status_map = {'Single': 2, 'Married': 1, 'Divorced': 0}
employment_status_map = {'Employed': 0, 'Self-employed': 2, 'Unemployed': 3, 'Retired': 1}
smoker_map = {'No': 1, 'Yes': 2, 'Occasional': 0}
plan_type_map = {'Bronze': 0, 'Silver': 2, 'Gold': 1, 'Platinum': 3}
network_tier_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}

st.title("Health Risk Prediction System")
st.write("Predict whether a user is **High Risk** based on health & insurance data.")

with st.form("prediction_form"):
    st.subheader("📥 Enter Patient & Policy Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 30)
        sex = st.selectbox("Sex", list(sex_map.keys()))
        region = st.selectbox("Region", list(region_map.keys()))
        urban_rural = st.selectbox("Urban/Rural", list(urban_rural_map.keys()))
        income = st.number_input("Income", value=50000.0)
        education = st.selectbox("Education", list(education_map.keys()))
        marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
        employment_status = st.selectbox("Employment Status", list(employment_status_map.keys()))
        household_size = st.number_input("Household Size", 1, 10, 1)
        dependents = st.number_input("Dependents", 0, 10, 0)
        bmi = st.number_input("BMI", value=25.0)
        smoker = st.selectbox("Smoker Status", list(smoker_map.keys()))
        alcohol_freq = st.selectbox("Alcohol Frequency", ["None", "Rarely", "Occasionally", "Frequently"]) # Imputed as Mode in NB
    
    with col2:
        visits_last_year = st.number_input("Visits Last Year", 0, 20, 0)
        hospitalizations_last_3yrs = st.number_input("Hospitalizations (3 yrs)", 0, 10, 0)
        days_hospitalized_last_3yrs = st.number_input("Days Hospitalized (3 yrs)", 0, 100, 0)
        medication_count = st.number_input("Medication Count", 0, 20, 0)
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 50, 120, 80)
        ldl = st.number_input("LDL Cholesterol", 50, 250, 100)
        hba1c = st.number_input("HbA1c", 4.0, 15.0, 5.5)
        plan_type = st.selectbox("Plan Type", list(plan_type_map.keys()))
        network_tier = st.selectbox("Network Tier", list(network_tier_map.keys()))
        risk_score = st.slider("Internal Risk Score", 0.0, 100.0, 50.0)
        annual_premium = st.number_input("Annual Premium", value=1200.0)
        total_claims_paid = st.number_input("Total Claims Paid", value=0.0)

    st.subheader("🏥 Health History")
    # Mapping "Yes/No" to 1/0 for binary flags
    chronic_count = st.number_input("Total Chronic Conditions", 0, 10, 0)
    hypertension = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabetes")
    asthma = st.checkbox("Asthma")
    copd = st.checkbox("COPD")
    cardiovascular = st.checkbox("Cardiovascular Disease")
    cancer = st.checkbox("Cancer History")
    kidney = st.checkbox("Kidney Disease")
    liver = st.checkbox("Liver Disease")
    arthritis = st.checkbox("Arthritis")
    mental_health = st.checkbox("Mental Health Condition")
    major_procedure = st.checkbox("Had Major Procedure")

    submit = st.form_submit_button("Predict Risk")

if submit:
    # 2. Preprocessing: Feature Engineering (BMI Category)
    bmi_cat = 0
    if bmi < 18.5: bmi_cat = 0
    elif bmi < 24.9: bmi_cat = 1
    elif bmi < 29.9: bmi_cat = 2
    else: bmi_cat = 3

    # 3. Construct Dataframe (Must match training columns exactly)
    input_dict = {
        "age": age, "sex": sex_map[sex], "region": region_map[region], 
        "urban_rural": urban_rural_map[urban_rural], "income": income, 
        "education": education_map[education], "marital_status": marital_status_map[marital_status],
        "employment_status": employment_status_map[employment_status], "household_size": household_size,
        "dependents": dependents, "bmi": bmi, "smoker": smoker_map[smoker],
        "alcohol_freq": 0, # Placeholder for encoding
        "visits_last_year": visits_last_year, "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
        "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs, "medication_count": medication_count,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp, "ldl": ldl, "hba1c": hba1c,
        "plan_type": plan_type_map[plan_type], "network_tier": network_tier_map[network_tier],
        "deductible": 500, "copay": 20, "policy_term_years": 1, "policy_changes_last_2yrs": 0,
        "provider_quality": 3.0, "risk_score": risk_score, "annual_premium": annual_premium,
        "monthly_premium": annual_premium/12, "claims_count": 0, "avg_claim_amount": 0.0,
        "total_claims_paid": total_claims_paid, "chronic_count": chronic_count,
        "hypertension": int(hypertension), "diabetes": int(diabetes), "asthma": int(asthma),
        "copd": int(copd), "cardiovascular_disease": int(cardiovascular), "cancer_history": int(cancer),
        "kidney_disease": int(kidney), "liver_disease": int(liver), "arthritis": int(arthritis),
        "mental_health": int(mental_health), "proc_imaging_count": 0, "proc_surgery_count": 0,
        "proc_physio_count": 0, "proc_consult_count": 0, "proc_lab_count": 0,
        "had_major_procedure": int(major_procedure), "bmi_category": bmi_cat
    }
    
    input_df = pd.DataFrame([input_dict])

    # 4. Feature Scaling (Using the loaded scaler from the notebook)
    # The scaler was fit on ['age', 'bmi', 'income', 'risk_score', 'annual_premium']
    scale_cols = ['age', 'bmi', 'income', 'risk_score', 'annual_premium']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # 5. Prediction
    try:
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("⚠️ High Risk Patient Detected")
        else:
            st.success("✅ Low Risk Patient")
    except ValueError as e:
        st.error(f"Feature Mismatch: {e}")
