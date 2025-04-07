import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import pickle
import matplotlib.pyplot as plt

# Load model
with open("model/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
feature_names = X.columns.tolist()

# UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction with Explainability")

st.write("### Enter patient details:")

# Inputs
age = st.slider("Age", 20, 90, 45)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of the ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Predict
if st.button("🧠 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ The person **may have heart disease**. (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ The person **is unlikely to have heart disease**. (Confidence: {prob:.2f})")

    # SHAP
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(input_data)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0][:, 1], show=False)
    st.pyplot(fig)

    # LIME
    st.subheader("🧪 LIME Explanation")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=feature_names,
        class_names=["No Disease", "Heart Disease"],
        mode="classification"
    )
    exp = lime_explainer.explain_instance(
        input_data[0],
        model.predict_proba,
        num_features=10
    )
    fig_lime = exp.as_pyplot_figure()
    st.pyplot(fig_lime)

    # Bullet Point Risk Analysis
    st.subheader("📌 Risk Analysis")
    bullet_points = []
    if chol > 240:
        bullet_points.append("🔴 High Cholesterol (>240 mg/dl) → Consider dietary changes and cholesterol management.")
    if trestbps > 140:
        bullet_points.append("🔴 High Resting BP (>140 mmHg) → May indicate hypertension. Monitor regularly.")
    if thalach < 120:
        bullet_points.append("🟠 Low Max Heart Rate → May suggest reduced heart function.")
    if ca >= 2:
        bullet_points.append("🔴 Multiple major vessels blocked → Clinical assessment advised.")
    if oldpeak > 2:
        bullet_points.append("🟠 ST depression > 2 → Possible myocardial ischemia.")
    if not bullet_points:
        bullet_points.append("✅ No major risk indicators based on current values.")

    for point in bullet_points:
        st.markdown(f"- {point}")

    # AI Advisor (Human-style Insight)
    st.subheader("🤖 AI Health Advisor")

    advisor_message = "Based on the inputs and prediction:\n\n"

    if prediction == 1:
        advisor_message += "- The model suspects potential heart disease. It's strongly advised to consult a cardiologist.\n"
    else:
        advisor_message += "- The model does not currently detect major signs of heart disease.\n"

    if chol > 240 or trestbps > 140:
        advisor_message += "- Your blood pressure or cholesterol level is above normal — these are key contributors to heart issues.\n"

    if oldpeak > 2:
        advisor_message += "- Exercise-induced ST depression is present, which could mean reduced oxygen supply to the heart during activity.\n"

    if thalach < 120:
        advisor_message += "- A low maximum heart rate might indicate poor heart efficiency.\n"

    advisor_message += "\n🧠 *Stay proactive: get regular checkups, maintain a healthy diet, exercise regularly, and reduce stress.*"

    st.info(advisor_message)
