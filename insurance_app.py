import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessing objects
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Title and description
st.title("🏥 Insurance Charges Prediction")
st.markdown("Predict medical insurance charges based on personal information.")

# Create input widgets
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Calculate BMI category
if bmi < 18.5:
    bmi_category = 'Underweight'
elif bmi < 24.9:
    bmi_category = 'Normal'
elif bmi < 29.9:
    bmi_category = 'Overweight'
else:
    bmi_category = 'Obese'

# Prediction button
if st.button("Predict Charges", type="primary"):
    # Create input dictionary
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'is_female': 1 if sex == 'female' else 0,
        'is_smoker': 1 if smoker == 'yes' else 0,
        'region_northeast': 1 if region == 'northeast' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        'bmi_category_Normal': 1 if bmi_category == 'Normal' else 0,
        'bmi_category_Overweight': 1 if bmi_category == 'Overweight' else 0,
        'bmi_category_Obese': 1 if bmi_category == 'Obese' else 0
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[expected_columns]

    # Scale the numerical columns
    numerical_cols = ['age', 'bmi', 'children']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"💰 Predicted Annual Insurance Charges: **${prediction:,.2f}**")

    # Additional information
    st.info(f"Your BMI category: **{bmi_category}** (BMI: {bmi:.1f})")

    # Risk factors
    risk_factors = []
    if smoker == 'yes':
        risk_factors.append("Smoker")
    if bmi >= 30:
        risk_factors.append("Obese")
    if age >= 50:
        risk_factors.append("Age 50+")

    if risk_factors:
        st.warning(f"⚠️ Risk factors identified: {', '.join(risk_factors)}")
    else:
        st.info("✅ No major risk factors identified")