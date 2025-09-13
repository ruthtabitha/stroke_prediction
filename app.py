import streamlit as st
import joblib
import numpy as np

# Load model yang sudah kamu simpan
model = joblib.load("model2_LogReg.pkl")  # ganti sesuai nama file model kamu

# Judul & CTA
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>🧠 Stroke Possibility Prediction</h1>
    <p style='text-align: center; color: gray;'>✨ Input complete information here & let's see the magic ✨</p>
    """,
    unsafe_allow_html=True,
)

st.write("👇 Yuk isi datanya dulu, jangan malu-malu:")

# Input fields
age = st.number_input("🌱 Age", min_value=0.0, max_value=120.0, step=1.0)

hypertension = st.radio("💓 Hypertension Record", ["No", "Yes"])
hypertension = 1 if hypertension == "Yes" else 0

heart_disease = st.radio("❤️ Heart Disease Record", ["No", "Yes"])
heart_disease = 1 if heart_disease == "Yes" else 0

ever_married = st.radio("💍 Ever Married", ["No", "Yes"])
ever_married = 1 if ever_married == "Yes" else 0  # encode jadi 0/1

work_type = st.selectbox("💼 Work Type", ["Private", "Self-employed", "Govt_job", "children"])
residence_type = st.radio("🏡 Residence Type", ["Urban", "Rural"])

glucose = st.number_input("🍭 Average Glucose Level", min_value=0.0, max_value=400.0, step=0.1)
bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=80.0, step=0.1)

smoking_status = st.selectbox("🚬 Smoking Status", ["formerly smoked", "smokes", "Unknown", "never smoked"])

# Encode categorical features
work_type_dict = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3}
residence_type_dict = {"Urban": 1, "Rural": 0}
smoking_status_dict = {"formerly smoked": 0, "smokes": 1, "Unknown": 2, "never smoked": 3}

work_type = work_type_dict[work_type]
residence_type = residence_type_dict[residence_type]
smoking_status = smoking_status_dict[smoking_status]

# Prediction button
if st.button("✨ Predict Now ✨"):
    X_new = np.array([[age, hypertension, heart_disease, ever_married,
                       work_type, residence_type, glucose, bmi, smoking_status]])
    
    pred = model.predict(X_new)[0]
    
    if pred == 1:
        st.error("⚠️ Predicted: **YES, risk of stroke detected!** 🚨 Take care and consult your doctor!")
        st.markdown("<p style='color: red;'>Category: WARNING ⚡</p>", unsafe_allow_html=True)
    else:
        st.success("✅ Predicted: **NO, you're safe!** 🎉 Stay healthy and keep shining 🌟")
        st.markdown("<p style='color: green;'>Category: NO WARNING 🌿</p>", unsafe_allow_html=True)
