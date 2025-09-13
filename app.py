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
ever_married = 1 if ever_married == "Yes" else 0

work_type = st.selectbox("💼 Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

glucose = st.number_input("🍭 Average Glucose Level", min_value=0.0, max_value=400.0, step=0.1)
bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=80.0, step=0.1)

smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Ambil daftar kolom dari model ---
expected_cols = list(model.feature_names_in_)

# --- Build input dict dengan default 0 ---
input_dict = {col: 0 for col in expected_cols}

# isi numeric / binary langsung (cek kalau ada di model)
if "age" in input_dict: input_dict["age"] = age
if "hypertension" in input_dict: input_dict["hypertension"] = hypertension
if "heart_disease" in input_dict: input_dict["heart_disease"] = heart_disease
if "ever_married" in input_dict: input_dict["ever_married"] = ever_married
if "avg_glucose_level" in input_dict: input_dict["avg_glucose_level"] = glucose
if "bmi" in input_dict: input_dict["bmi"] = bmi

# one-hot work_type (kalau ada di model)
wt_col = f"work_type_{work_type}"
if wt_col in input_dict:
    input_dict[wt_col] = 1

# one-hot smoking_status (kalau ada di model)
sm_col = f"smoking_status_{smoking_status}"
if sm_col in input_dict:
    input_dict[sm_col] = 1

# --- Jadi DataFrame sesuai urutan model ---
X_new = pd.DataFrame([input_dict])[expected_cols]

# --- Predict button ---
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
