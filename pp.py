import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Streamlit app layout
st.title("🩺 AI Diabetes Risk Assessment")

st.write("Enter your health details below:")

# Input fields
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, bmi, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High risk of Diabetes.")
    else:
        st.success("✅ Low risk of Diabetes.")
