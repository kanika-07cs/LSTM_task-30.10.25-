import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


st.set_page_config(page_title="LSTM Temperature Forecast", layout="wide")


@st.cache_resource
def load_lstm_model():
    model = load_model("lstm_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl") if "scaler.pkl" in os.listdir() else None


model = load_lstm_model()
scaler = load_scaler()

st.title("🌡️ LSTM Temperature Forecast Dashboard")
st.markdown("Visualize model predictions and **forecast future temperature trends** interactively.")

uploaded_file = st.file_uploader("📁 Upload your test CSV file (with 'date' and 'T' columns)", type=["csv"])

if uploaded_file:
   
    df = pd.read_csv(uploaded_file)

    if 'date' not in df.columns:
        st.error("The file must include a 'date' column!")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    st.write("### Uploaded Data (first 10 rows)")
    st.dataframe(df.head(10))

    LOOKBACK = 72
    target_col = 'T'

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df.drop(columns=['date']))
    else:
        scaled_data = scaler.transform(df.drop(columns=['date']))

    target_idx = list(df.drop(columns=['date']).columns).index(target_col)

    def create_sequences(data, lookback, target_col_index):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            y.append(data[i, target_col_index])
        return np.array(X), np.array(y)

    X, y_true = create_sequences(scaled_data, LOOKBACK, target_idx)

    y_pred = model.predict(X)

    inv_pred = np.zeros((len(y_pred), scaled_data.shape[1]))
    inv_pred[:, target_idx] = y_pred[:, 0]
    y_pred_actual = scaler.inverse_transform(inv_pred)[:, target_idx]

    inv_true = np.zeros((len(y_true), scaled_data.shape[1]))
    inv_true[:, target_idx] = y_true
    y_true_actual = scaler.inverse_transform(inv_true)[:, target_idx]


    st.subheader("📊 Model Predictions on Uploaded Data")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'][LOOKBACK:], y_true_actual, label='Actual Temperature', linewidth=2)
    ax.plot(df['date'][LOOKBACK:], y_pred_actual, label='Predicted Temperature', linestyle='--')
    ax.set_title("Actual vs Predicted Temperature")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature")
    ax.legend()
    st.pyplot(fig)

  
    st.subheader("🔮 Forecast Future Temperature")

    future_steps = st.number_input("Enter number of future time steps to predict:",
                                   min_value=1, max_value=200, value=50)

    if st.button("Predict Future Temperature"):
        last_seq = scaled_data[-LOOKBACK:]
        future_preds, current_seq = [], last_seq.copy()

        for _ in range(future_steps):
            pred = model.predict(current_seq[np.newaxis, :, :])[0][0]
            future_preds.append(pred)

            next_step = current_seq[-1].copy()
            next_step[target_idx] = pred
            current_seq = np.vstack([current_seq[1:], next_step])

        inv_future = np.zeros((len(future_preds), scaled_data.shape[1]))
        inv_future[:, target_idx] = future_preds
        future_actual = scaler.inverse_transform(inv_future)[:, target_idx]

        future_dates = pd.date_range(df['date'].iloc[-1], periods=future_steps + 1, freq='h')[1:]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Temperature': future_actual})

        st.write("### 🔥 Future Temperature Predictions")
        st.dataframe(forecast_df.head(20))


else:
    st.info("⬆️ Please upload a CSV file to begin forecasting.")
