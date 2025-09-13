import streamlit as st
import joblib
import pandas as pd

# --- Load model ---
model = joblib.load("model2_LogReg.pkl")  # ganti sesuai nama file model kamu

# --- Page Config ---
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>🧠 Stroke Possibility Prediction</h1>
    <p style='text-align: center; color: gray;'>✨ Input your details below & let's see the magic ✨</p>
    """,
    unsafe_allow_html=True,
)

st.write("👇 Yuk isi datanya dulu, jangan malu-malu:")

# --- Input fields ---
age = st.number_input("🌱 Age", min_value=0.0, max_value=120.0, step=1.0)

hypertension = st.radio("💓 Hypertension Record", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

heart_disease = st.radio("❤️ Heart Disease Record", ["No", "Yes"])
heart_disease = 1 if heart_disease == "Yes" else 0

ever_married = st.radio("💍 Ever Married", ["No", "Yes"])
ever_married = 1 if ever_married == "Yes" else 0  # encode jadi 0/1

work_type = st.selectbox("💼 Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
glucose = st.number_input("🍭 Average Glucose Level", min_value=0.0, max_value=400.0, step=0.1)
bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=80.0, step=0.1)

smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Build input dictionary (dummy cols for id, gender, residence) ---
input_dict = {
    "id": 0,
    "gender": 0,
    "Residence_type": 0,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "avg_glucose_level": glucose,
    "bmi": bmi,
    # default one-hot cols
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

# aktifkan sesuai input
if f"work_type_{work_type}" in input_dict:
    input_dict[f"work_type_{work_type}"] = 1
if f"smoking_status_{smoking_status}" in input_dict:
    input_dict[f"smoking_status_{smoking_status}"] = 1

# --- Susun DataFrame sesuai urutan feature di model ---
expected_cols = list(model.feature_names_in_)
X_new = pd.DataFrame([input_dict])[expected_cols]

# --- Prediction button ---
if st.button("✨ Predict Now ✨"):
    try:
        pred = model.predict(X_new)[0]

        if pred == 1:
            st.error("⚠️ Predicted: **YES, risk of stroke detected!** 🚨 Take care and consult your doctor!")
            st.markdown("<p style='color: red; font-weight: bold;'>Category: WARNING ⚡</p>", unsafe_allow_html=True)
        else:
            st.success("✅ Predicted: **NO, you're safe!** 🎉 Stay healthy and keep shining 🌟")
            st.markdown("<p style='color: green; font-weight: bold;'>Category: NO WARNING 🌿</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Terjadi error saat prediksi:")
        st.exception(e)
