import streamlit as st
import numpy as np
import pickle


st.set_page_config(
    page_title="CardioRisk",
    layout="centered"
)


# Title & disclaimer
#----------------------------------------
st.title("CardioRisk - Heart Disease Risk Prediction")
st.caption("Machine Learning - based clinical risk assessment")


#Load Pkl
with open("model/heart_model.pkl", "rb") as f:
    model = pickle.load(f)


# User Inputs (Readable Medical Names)
#--------------------------------------------
age = st.slider("Age (years)", 18, 90)

sex_label = st.selectbox("Sex", ["Male", "Female"])

cp_label = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

trestbps = st.slider(
    "Resting Blood Pressure (mm Hg)",
    80, 200
)

chol = st.slider(
    "Serum Cholesterol (mg/dl)",
    100, 400
)

fbs_label = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

restecg_label = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

thalch = st.slider(
    "Maximum Heart Rate Achieved",
    60, 220
)

exang_label = st.selectbox(
    "Exercise-Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.slider(
    "ST Depression (Exercise vs Rest)",
    0.0, 6.0
)

# Encoding (MUST match training preprocessing)
#-----------------------------------------------

sex = 1 if sex_label == "Male" else 0

cp = [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
].index(cp_label)

fbs = 1 if fbs_label == "Yes" else 0

restecg = [
    "Normal",
    "ST-T Wave Abnormality",
    "Left Ventricular Hypertrophy"
].index(restecg_label)

exang = 1 if exang_label == "Yes" else 0

# input  array
#------------------------------
features = np.array([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalch,
    exang,
    oldpeak
]])

# Prediction
#--------------------------------------------
if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\n"
            f"Confidence: {probability * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Confidence: {(1 - probability) * 100:.2f}%"
        )


# Footer
#---------------------
st.markdown("---")
st.caption("Developed with Random forest Classifier")
