import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Load trained pipeline
# ----------------------------
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="Daily Internet Usage Predictor", page_icon="💻")
st.title("Prediction of Daily Internet Data Usage Among University Students")
st.info("This model was trained using student data. Predictions for other users may be less accurate.")

st.write("Enter your daily usage details:")

# ----------------------------
# Inputs
# ----------------------------
screen_time_hours = st.slider("Screen Time (hours/day)", 1.0, 12.0, 5.0, 0.1)
video_hours = st.slider("Video Streaming (hours/day)", 0.0, 6.0, 2.0, 0.1)
social_hours = st.slider("Social Media (hours/day)", 0.0, 5.0, 1.0, 0.1)
online_classes_hours = st.slider("Online Classes (hours/day)", 0.0, 4.0, 1.0, 0.1)
downloads = st.number_input("Number of Downloads", min_value=0, max_value=20, value=1)

device_type = st.selectbox("Device Type", ["phone", "laptop"])
internet_type = st.selectbox("Internet Type", ["4G", "5G", "WiFi"])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Daily Data Usage"):
    input_data = pd.DataFrame([{
        "screen_time_hours": screen_time_hours,
        "video_hours": video_hours,
        "social_hours": social_hours,
        "online_classes_hours": online_classes_hours,
        "downloads": downloads,
        "device_type": device_type,
        "internet_type": internet_type
    }])

    prediction = model.predict(input_data)

    prediction_mb = prediction[0]
    prediction_gb = prediction_mb / 1024

    st.success(f"Predicted Usage: {prediction_mb:.2f} MB (~ {prediction_gb:.2f} GB)")
