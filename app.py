import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Stroke Prediction App 🚑")

# --- Input dari user ---
age = st.number_input("Age", min_value=0, max_value=120, value=45)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
avg_glucose_level = st.number_input("Average Glucose Level", value=100.0, step=0.1)
bmi = st.number_input("BMI", value=25.0, step=0.1)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Buat dictionary input sesuai feature model ---
# Mulai dengan dummy kolom
input_dict = {
    "id": 0,
    "gender": 0,
    "Residence_type": 0,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": 1 if ever_married == "Yes" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    # default one-hot untuk kategori
    "work_type_Private": 0,
    "work_type_Self-employed": 0,
    "work_type_Govt_job": 0,
    "work_type_children": 0,
    "work_type_Never_worked": 0,
    "smoking_status_never smoked": 0,
    "smoking_status_formerly smoked": 0,
    "smoking_status_smokes": 0,
    "smoking_status_Unknown": 0
}

# Aktifkan kolom one-hot sesuai pilihan user
if f"work_type_{work_type}" in input_dict:
    input_dict[f"work_type_{work_type}"] = 1
if f"smoking_status_{smoking_status}" in input_dict:
    input_dict[f"smoking_status_{smoking_status}"] = 1

# --- Pastikan urutan kolom sama persis dengan model ---
expected_cols = list(model.feature_names_in_)
X_new = pd.DataFrame([input_dict])[expected_cols]

st.write("Input siap dipakai model:")
st.dataframe(X_new)

# --- Prediksi ---
if st.button("Predict"):
    try:
        pred = model.predict(X_new)[0]
        if pred == 1:
            st.error("⚠️ Predicted: YES (risk of stroke detected)")
        else:
            st.success("✅ Predicted: NO (no stroke detected)")
    except Exception as e:
        st.error("Terjadi error saat prediksi:")
        st.exception(e)
