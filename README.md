# 🧠 Brain Stroke Prediction with Machine Learning

Stroke is one of the leading causes of death and disability worldwide.  
This project uses **machine learning** to analyze health and demographic data, predict the likelihood of stroke, and provide insights to support **preventive healthcare**.

---

## 📌 Business Understanding
The main objectives are:
- Identify the key factors contributing to stroke occurrence  
- Build a predictive model to classify individuals into stroke risk levels  
- Compare model performance against baseline predictions  
- Generate actionable insights to support public health strategies  

---

## ❓ Key Questions
- Which factors contribute the most to stroke occurrence?  
- Can a machine learning model accurately classify individuals into different stroke risk levels?  
- How well can the model perform compared to baseline predictions?  
- What insights can be drawn to inform preventive strategies?  

---

## 📊 Data Understanding

**Source:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
*with many manipulation since the real data had severe imbalance

| Column            | Description |
|-------------------|-------------|
| `gender`          | Male, Female, or Other |
| `age`             | Age of the patient |
| `hypertension`    | 0 = no hypertension, 1 = hypertension |
| `heart_disease`   | 0 = no heart disease, 1 = heart disease |
| `ever_married`    | Yes / No |
| `work_type`       | Children, Govt_job, Private, Self-employed, Never_worked |
| `Residence_type`  | Rural / Urban |
| `avg_glucose_level` | Average glucose level in blood |
| `bmi`             | Body mass index |
| `smoking_status`  | Formerly smoked, Never smoked, Smokes, Unknown |
| `stroke`          | 1 = stroke, 0 = no stroke |

---

## ⚙️ Data Preprocessing
- Handle missing values in **BMI**  
- Encode categorical variables  
- Normalize continuous features (`age`, `bmi`, `avg_glucose_level`)  
- Split dataset into **train (80%)** and **test (20%)**

---

## 🤖 Models
- Logistic Regression  
- Random Forest  
- etc 

---

## 📈 Results
- **Dataset Size:** 560 patients  
- **Positive Stroke Cases:** 288 (51.4%)  
- **Best Model Accuracy:** ~84.8% (random forest), ~80.4% (logistic regression)
- **Strongest Predictors:** Age  

---

## 💻 Dummy App
A **Streamlit app** was developed where users can input factors (age, gender, hypertension, glucose level, etc.) and instantly get a stroke prediction.

🔗 [Try the App](#) *(https://stroke-predictions-ruth.streamlit.app/)*

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
