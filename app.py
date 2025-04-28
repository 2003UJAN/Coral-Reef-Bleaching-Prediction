import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# For LSTM
from tensorflow.keras.models import load_model

# For XGBoost
from xgboost import XGBRegressor

# --- CONFIGURATION ---
RF_REPO_ID = "ujan2003/rf_model.pkl"
RF_FILENAME = "rf_model.pkl"
XGB_MODEL_PATH = "xgb_model.pkl"  # Place your XGBoost model here
LSTM_MODEL_PATH = "lstm_model.h5" # Place your LSTM model here
SCALER_PATH = "scaler.pkl"        # Place your StandardScaler here

# --- LOAD MODELS ---
@st.cache_resource
def load_rf_model():
    model_path = hf_hub_download(repo_id=RF_REPO_ID, filename=RF_FILENAME, revision="main")
    return joblib.load(model_path)

@st.cache_resource
def load_xgb_model():
    if os.path.exists(XGB_MODEL_PATH):
        return joblib.load(XGB_MODEL_PATH)
    else:
        return None

@st.cache_resource
def load_lstm_model():
    if os.path.exists(LSTM_MODEL_PATH):
        return load_model(LSTM_MODEL_PATH)
    else:
        return None

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    else:
        return None

# --- MAIN APP ---
def main():
    st.title("Coral Reef Bleaching Prediction Web App")
    st.markdown("Predict coral bleaching using environmental factors and ML models (Random Forest, XGBoost, ARIMA).")

    # --- DATA UPLOAD ---
    uploaded_file = st.file_uploader("Upload your coral reef dataset (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("coralreef_dataset.csv")

    # --- FEATURE SELECTION ---
    features = [
        'Cyclone_Frequency', 'Depth_m', 'ClimSST', 'Turbidity',
        'Temperature_Maximum', 'SSTA', 'TSA', 'Temperature_Mean'
    ]
    target = 'Percent_Bleaching'

    # --- MODEL SELECTION ---
    model_type = st.selectbox("Select Model", ["Random Forest (HuggingFace)", "XGBoost", "ARIMA"])

    # --- PREPROCESSING ---
    df = df.copy()
    for col in features:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in data!")
            return
    df = df.dropna(subset=features)
    X = df[features]
    y = df[target] if target in df.columns else None

    # --- PREDICTION ---
    if model_type == "Random Forest (HuggingFace)":
        rf_model = load_rf_model()
        scaler = load_scaler()
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        preds = rf_model.predict(X_scaled)
        df['Predicted_Bleaching'] = preds
        st.write("Random Forest Predictions (last 10 rows):")
        st.dataframe(df[features + ['Predicted_Bleaching']].tail(10))
        st.line_chart(df['Predicted_Bleaching'])

    elif model_type == "XGBoost":
        xgb_model = load_xgb_model()
        scaler = load_scaler()
        if xgb_model is None:
            st.warning("XGBoost model not found. Please train and save as 'xgb_model.pkl'.")
        else:
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            preds = xgb_model.predict(X_scaled)
            df['Predicted_Bleaching'] = preds
            st.write("XGBoost Predictions (last 10 rows):")
            st.dataframe(df[features + ['Predicted_Bleaching']].tail(10))
            st.line_chart(df['Predicted_Bleaching'])

    elif model_type == "LSTM":
        lstm_model = load_lstm_model()
        scaler = load_scaler()
        if lstm_model is None:
            st.warning("LSTM model not found. Please train and save as 'lstm_model.h5'.")
        else:
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            preds = lstm_model.predict(X_lstm).flatten()
            df['Predicted_Bleaching'] = preds
            st.write("LSTM Predictions (last 10 rows):")
            st.dataframe(df[features + ['Predicted_Bleaching']].tail(10))
            st.line_chart(df['Predicted_Bleaching'])

    elif model_type == "ARIMA":
        if 'Date_Year' not in df.columns or target not in df.columns:
            st.warning("ARIMA requires 'Date_Year' and 'Percent_Bleaching' columns.")
        else:
            df['Date'] = pd.to_datetime(df['Date_Year'], format='%Y', errors='coerce')
            ts = df.groupby('Date')[target].mean().dropna()
            st.write("Time Series of Bleaching (Yearly Mean):")
            st.line_chart(ts)
            steps = st.slider("Forecast years into the future", 1, 10, 5)
            model = ARIMA(ts, order=(2,1,2))
            results = model.fit()
            forecast = results.get_forecast(steps=steps)
            pred_index = pd.date_range(ts.index[-1], periods=steps+1, freq='Y')[1:]
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int()
            # Plot
            fig, ax = plt.subplots()
            ts.plot(ax=ax, label='Observed')
            ax.plot(pred_index, pred_mean, label='Forecast', color='orange')
            ax.fill_between(pred_index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='orange', alpha=0.3)
            ax.legend()
            st.pyplot(fig)

    # --- INDIVIDUAL PREDICTION ---
    st.sidebar.header("Predict for Custom Input")
    input_data = {}
    for col in features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    input_df = pd.DataFrame([input_data])

    if st.sidebar.button("Predict Bleaching for Input"):
        scaler = load_scaler()
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        if model_type == "Random Forest (HuggingFace)":
            rf_model = load_rf_model()
            pred = rf_model.predict(input_scaled)[0]
        elif model_type == "XGBoost":
            xgb_model = load_xgb_model()
            if xgb_model is None:
                st.sidebar.warning("XGBoost model not found.")
                return
            pred = xgb_model.predict(input_scaled)[0]
        elif model_type == "LSTM":
            lstm_model = load_lstm_model()
            if lstm_model is None:
                st.sidebar.warning("LSTM model not found.")
                return
            input_lstm = input_scaled.reshape((1, len(features), 1))
            pred = lstm_model.predict(input_lstm)[0][0]
        else:
            st.sidebar.warning("Custom input prediction not supported for ARIMA.")
            return
        st.sidebar.success(f"Predicted Bleaching: {pred:.2f}%")

if __name__ == "__main__":
    main()                      
