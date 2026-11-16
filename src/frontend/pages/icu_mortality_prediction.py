import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.frontend.config.settings import settings

API_URL = settings.api_url

st.set_page_config(page_title="ICU Mortality Risk Prediction", page_icon="🏥", layout="wide")
st.title("🏥 ICU Patient Mortality Risk Prediction")
st.write("Enter the patient first-24-hour clinical summary values and click **Predict**.")
st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=65.0)
    BUN_max = st.number_input("Maximum BUN", min_value=0.0, value=30.0)
    sao2_std = st.number_input("SaO2 Standard Deviation", min_value=0.0, value=2.5)
    heartrate_mean = st.number_input("Mean Heart Rate", min_value=0.0, value=90.0)
    sao2_min = st.number_input("Minimum SaO2", min_value=0.0, max_value=100.0, value=90.0)
    respiration_std = st.number_input("Respiration Standard Deviation", min_value=0.0, value=4.0)
    respiration_mean = st.number_input("Mean Respiration Rate", min_value=0.0, value=20.0)

with col2:
    RDW_min = st.number_input("Minimum RDW", min_value=0.0, value=14.0)
    heartrate_min = st.number_input("Minimum Heart Rate", min_value=0.0, value=65.0)
    hospitaladmitoffset = st.number_input("Hospital Admit Offset", value=-120.0)
    heartrate_max = st.number_input("Maximum Heart Rate", min_value=0.0, value=120.0)
    heartrate_std = st.number_input("Heart Rate Standard Deviation", min_value=0.0, value=10.0)
    calcium_mean = st.number_input("Mean Calcium", min_value=0.0, value=8.5)
    admissionweight = st.number_input("Admission Weight", min_value=0.0, value=70.0)

if st.button("🔍 Predict Mortality Risk", use_container_width=True):
    payload = {
        "age": age,
        "BUN_max": BUN_max,
        "sao2_std": sao2_std,
        "heartrate_mean": heartrate_mean,
        "sao2_min": sao2_min,
        "respiration_std": respiration_std,
        "respiration_mean": respiration_mean,
        "RDW_min": RDW_min,
        "heartrate_min": heartrate_min,
        "hospitaladmitoffset": hospitaladmitoffset,
        "heartrate_max": heartrate_max,
        "heartrate_std": heartrate_std,
        "calcium_mean": calcium_mean,
        "admissionweight": admissionweight,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error("Unable to reach prediction service.")
        st.caption(str(e))
        st.stop()

    result = response.json()
    prediction = result["prediction"]
    probability = float(result["mortality_probability"])
    threshold = float(result["threshold"])

    
    st.divider()
    st.subheader("Prediction Results")

    st.metric(
        label="Estimated Mortality Probability",
        value=f"{probability:.2%}"
    )

    st.write(f"**Classification Threshold:** {threshold:.2%}")

    if prediction == "Expired":
        st.error("Model Classification: Expired")
    else:
        st.success("Model Classification: Alive")

    st.info(
        "This is an experimental machine learning prediction "
        "based on the supplied first-24-hour clinical features. "
        "It is not a clinical diagnosis or a guarantee of patient outcome."
    )

    