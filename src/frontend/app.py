import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

st.set_page_config(page_title="ICU Mortality Risk Prediction", page_icon="🏥", layout="wide")

st.title("🏥 ICU Patient Mortality Risk Prediction")

st.write(
    """
    This application predicts hospital mortality risk
    using patient information, vital signs and laboratory
    measurements collected during the first 24 hours
    of ICU stay.
    """
)
st.info("Open the ICU Mortality Prediction page from the sidebar.")
st.warning(
    "This application is for ML demonstration purposes only and is not intended for clinical diagnosis."
)
