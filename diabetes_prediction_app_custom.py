
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Assessment", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction", "ℹ️ About"])

if page == "🏠 Home":
    st.title("Welcome to Diabetes Risk Assessment App")
    st.markdown("This app uses a machine learning model to predict the risk of diabetes based on health metrics.")
    st.image("https://cdn.pixabay.com/photo/2017/01/31/13/14/diabetes-2028253_960_720.png", use_column_width=True)

elif page == "📊 Prediction":
    st.title("Diabetes Risk Prediction")
    st.markdown("Please enter your health information below:")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        st.markdown(f"**Prediction Probability:** {probability:.2%}")
        st.progress(int(probability * 100))

elif page == "ℹ️ About":
    st.title("About This App")
    st.markdown("""
    This app is built using **Streamlit** and a **Random Forest** model trained on the Pima Indians Diabetes dataset.
    
    **Features used for prediction:**
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age

    Developed as part of an AI project for diabetes risk assessment.
    """)
