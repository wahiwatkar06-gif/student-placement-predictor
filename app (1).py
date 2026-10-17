import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌦️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "weather_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🌦️ Weather Prediction System")
st.write(
    "Enter measured weather conditions to predict "
    "the likely weather category."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.info(
        "Make sure weather_model.joblib is uploaded "
        "beside app.py in your GitHub repository."
    )
    st.stop()

with st.form("weather_form"):
    st.subheader("Enter weather conditions")

    precipitation = st.number_input(
        "Precipitation",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Use the same units as the training dataset."
    )

    temp_max = st.number_input(
        "Maximum temperature (°C)",
        min_value=-30.0,
        max_value=60.0,
        value=22.0,
        step=0.5
    )

    temp_min = st.number_input(
        "Minimum temperature (°C)",
        min_value=-40.0,
        max_value=50.0,
        value=12.0,
        step=0.5
    )

    wind = st.number_input(
        "Wind speed",
        min_value=0.0,
        value=2.0,
        step=0.1,
        help="Use the same units as the training dataset."
    )

    submitted = st.form_submit_button(
        "Predict Weather",
        type="primary",
        use_container_width=True
    )

if submitted:
    if temp_min > temp_max:
        st.error(
            "Minimum temperature cannot exceed "
            "maximum temperature."
        )
    else:
        inputs = pd.DataFrame([{
            "precipitation": precipitation,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "wind": wind
        }])

        prediction = model.predict(inputs)[0]

        st.success(
            f"Predicted weather category: {str(prediction).title()}"
        )
        st.caption(
            "This is a machine-learning classification result "
            "based on the supplied measurements, not a live forecast."
        )
