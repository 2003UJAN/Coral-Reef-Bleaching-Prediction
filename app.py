import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load models and preprocessing objects from root directory
lstm_model = tf.keras.models.load_model("lstm_model.h5")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Coral Reef Health Prediction", layout="wide")

st.title("🌊 Coral Reef Health Prediction App")
st.markdown("""
This app uses **LSTM** and **XGBoost** to predict coral reef health status from environmental and sensor data.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload Coral Reef Data (.csv)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(data.head())

    # Preprocessing
    try:
        scaled_data = scaler.transform(data)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()
    
    # LSTM input needs 3D: (samples, timesteps, features)
    try:
        lstm_input = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))
        lstm_prediction = lstm_model.predict(lstm_input)
        lstm_prediction = (lstm_prediction > 0.5).astype(int)
    except Exception as e:
        st.error(f"LSTM prediction error: {e}")
        lstm_prediction = None

    # XGBoost prediction
    try:
        xgb_prediction = xgb_model.predict(scaled_data)
        xgb_proba = xgb_model.predict_proba(scaled_data)
    except Exception as e:
        st.error(f"XGBoost prediction error: {e}")
        xgb_prediction = None

    st.subheader("🧠 Model Predictions")
    if lstm_prediction is not None:
        st.write("**LSTM Predictions** (0 = Healthy, 1 = At Risk)")
        st.write(lstm_prediction.flatten())
    if xgb_prediction is not None:
        st.write("**XGBoost Predictions** (0 = Healthy, 1 = At Risk)")
        st.write(xgb_prediction)

    # SHAP Explanation for XGBoost
    st.subheader("📈 XGBoost Model Explanation (SHAP)")
    try:
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(scaled_data)

        # Visualize SHAP summary
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.title("SHAP Summary Plot")
        shap.summary_plot(shap_values, features=data, plot_type="bar")
        st.pyplot(bbox_inches='tight')

        # Force plot for the first prediction
        st.subheader("🧪 SHAP Force Plot for First Prediction")
        shap.initjs()
        st.components.v1.html(
            shap.plots.force(explainer.expected_value, shap_values[0], data.iloc[0]).html(),
            height=300,
        )
    except Exception as e:
        st.error(f"SHAP explanation error: {e}")

else:
    st.info("Please upload a CSV file to get predictions.")

st.markdown("---")
st.caption("© 2025 Coral Reef AI Project | Built with Streamlit, LSTM, XGBoost & SHAP")
