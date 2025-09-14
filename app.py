import streamlit as st
import joblib
import pandas as pd

# --- Load model ---
model = joblib.load("model2_LogReg.pkl")  # ganti sesuai nama file model kamu

# --- Page Config ---
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>🧠 Stroke Risk Prediction</h1>
    <p style='text-align: center; color: gray;'>✨ Input your details below & let's see your risk! ✨</p>
    """,
    unsafe_allow_html=True,
)

st.write("👇 Yuk isi datanya dulu, jangan malu-malu:")

# --- Input tambahan untuk tampilan ---
name = st.text_input("📝 Your Name")
gender = st.radio("🚻 Gender", ["Male", "Female", "Other"])

if name:
    st.write(f"Hello {name}! You are identified as {gender}.")

# --- Input fields untuk model ---
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
for col, val in {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "avg_glucose_level": glucose,
    "bmi": bmi
}.items():
    if col in input_dict:
        input_dict[col] = val

# one-hot work_type
wt_col = f"work_type_{work_type}"
if wt_col in input_dict:
    input_dict[wt_col] = 1
else:
    st.warning(f"⚠️ Warning: Column `{wt_col}` not found in model features. This value is ignored.")

# one-hot smoking_status
sm_col = f"smoking_status_{smoking_status}"
if sm_col in input_dict:
    input_dict[sm_col] = 1
else:
    st.warning(f"⚠️ Warning: Column `{sm_col}` not found in model features. This value is ignored.")

# --- Jadi DataFrame sesuai urutan model ---
X_new = pd.DataFrame([input_dict])[expected_cols]

# --- Tampilkan input sebelum prediksi ---
st.subheader("📊 Input yang dikirim ke model:")
st.dataframe(X_new)

# --- Predict button dengan probabilitas ---
if st.button("✨ Predict Now ✨"):
    try:
        # Prediksi probabilitas
        prob = model.predict_proba(X_new)[0][1]  # probabilitas stroke
        pred = model.predict(X_new)[0]  # 0/1

        st.subheader("📈 Prediction Result:")

        if pred == 1:
            st.error(f"⚠️ Predicted: **YES, risk of stroke detected for {name if name else 'user'}!** 🚨")
        else:
            st.success(f"✅ Predicted: **NO, you're safe, {name if name else 'user'}!** 🎉")

        # Tampilkan probabilitas
        st.info(f"📊 Estimated Stroke Risk: **{prob*100:.2f}%**")

        # Kategori risiko
        if prob >= 0.7:
            st.markdown("<p style='color: red; font-weight: bold;'>Category: HIGH RISK ⚡</p>", unsafe_allow_html=True)
        elif prob >= 0.4:
            st.markdown("<p style='color: orange; font-weight: bold;'>Category: MEDIUM RISK ⚠️</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: green; font-weight: bold;'>Category: LOW RISK 🌿</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Terjadi error saat prediksi:")
        st.exception(e)
