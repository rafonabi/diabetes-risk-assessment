import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# UI
st.title("🩺 AI Diabetes Risk Assessment")
st.write("Enter your health information:")

# Inputs
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict
if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of class 1

    st.write(f"Model Confidence (High Risk): **{probability:.2f}**")

    if prediction == 1:
        st.error("⚠️ High risk of Diabetes.")
    else:
        st.success("✅ Low risk of Diabetes.")

