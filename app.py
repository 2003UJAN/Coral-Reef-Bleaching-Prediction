# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load saved models and scaler
scaler = joblib.load('scaler.pkl')
xgb = joblib.load('xgb_model.pkl')
rf = joblib.load('rf_model.pkl')
lstm_model = load_model('lstm_model.h5')

def main():
    st.title("Coral Bleaching Prediction System")
    
    # Data upload
    uploaded_file = st.file_uploader("Upload coral reef data (CSV)", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv('coralreef_dataset.csv')
    
    # Model selection
    model_type = st.selectbox("Select Model", ["XGBoost", "Random Forest", "LSTM", "ARIMA"])
    
    # Preprocessing
    features = ['Cyclone_Frequency', 'Depth_m', 'ClimSST', 'Turbidity',
               'Temperature_Maximum', 'SSTA', 'TSA', 'Temperature_Mean']
    
    if model_type in ["XGBoost", "Random Forest", "LSTM"]:
        X = df[features].fillna(df[features].mean())
        X_scaled = scaler.transform(X)
        
        if model_type == "XGBoost":
            preds = xgb.predict(X_scaled)
        elif model_type == "Random Forest":
            preds = rf.predict(X_scaled)
        elif model_type == "LSTM":
            X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            preds = lstm_model.predict(X_lstm).flatten()
            
        df['Predicted_Bleaching'] = preds
        
        # Show results
        st.write("Prediction Results:")
        st.dataframe(df[['Date_Year'] + features + ['Predicted_Bleaching']].tail())
        
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(df['Date_Year'], df['Predicted_Bleaching'], label='Predictions')
        plt.xlabel('Year')
        plt.ylabel('Bleaching Percentage')
        plt.title('Bleaching Predictions Over Time')
        st.pyplot(plt)
    
    # ARIMA Forecasting
    if model_type == "ARIMA":
        df['Date'] = pd.to_datetime(df['Date_Year'], format='%Y')
        time_series = df.groupby('Date')['Percent_Bleaching'].mean()
        
        # Fit ARIMA
        model = ARIMA(time_series, order=(5,1,0))
        results = model.fit()
        
        # Forecast
        steps = st.slider("Forecast Period (years)", 1, 10, 5)
        forecast = results.get_forecast(steps=steps)
        
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(time_series.index, time_series, label='Historical')
        plt.plot(forecast.predicted_index, forecast.predicted_mean, label='Forecast', color='orange')
        plt.fill_between(forecast.predicted_index,
                         forecast.conf_int()['lower Percent_Bleaching'],
                         forecast.conf_int()['upper Percent_Bleaching'],
                         color='orange', alpha=0.3)
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
