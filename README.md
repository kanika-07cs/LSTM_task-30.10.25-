# 🌡️ LSTM Temperature Forecasting Project

## 📘 Project Overview

This project builds a **Long Short-Term Memory (LSTM)** based deep learning model to predict **future temperature values** using historical weather data.
The LSTM network captures long-term temporal dependencies, making it ideal for **time-series forecasting tasks** such as temperature trends, energy load forecasting, or stock price predictions.

**Streamlit Link** : https://lstm-weather.streamlit.app/

## 🧾 Dataset Description

The dataset contains **hourly temperature readings** along with timestamps.
Typical structure:

| date             | T    | humidity | windspeed | pressure |
| ---------------- | ---- | -------- | --------- | -------- |
| 2024-01-01 00:00 | 23.1 | 74       | 3.2       | 1012     |
| 2024-01-01 01:00 | 22.8 | 73       | 3.1       | 1011     |
| 2024-01-01 02:00 | 22.6 | 75       | 2.9       | 1010     |

> * **`date`** → Time column (hourly data)
> * **`T`** → Temperature (target variable)
> * Other columns (optional) → additional environmental features

## ⚙️ Data Preprocessing Steps

1. **Datetime Parsing:**
   Converted the `date` column to datetime format using pandas.

2. **Sorting by Time:**
   Ensured chronological order of records.

3. **Scaling Features:**
   Used **MinMaxScaler** from `sklearn` to normalize features between 0 and 1 for better LSTM performance.

4. **Sequence Creation:**
   Used a **lookback window** (e.g., 72 hours) to create input sequences for LSTM.

5. **Train-Test Split:**
   The dataset was split into training (80%) and testing (20%) sets to evaluate generalization.


## 📈 Results & Insights

* The LSTM model successfully captured **hourly temperature trends** with low prediction error.
* **RMSE and MAE values** indicated good generalization.
* The model accurately followed the **seasonal and daily variation patterns** in temperature.
* Future predictions show realistic continuation of past trends.

## 🖥️ How to Run

1. Clone the Repository
git clone <repo>
cd <repo>

2. Run the Streamlit App - streamlit run app.py

## 🏁 Conclusion

* **LSTM models** are powerful for **time series forecasting** due to their ability to remember long-term dependencies.
* This project demonstrates a complete pipeline: data preparation → model training → evaluation → deployment.
* The Streamlit dashboard provides an **interactive interface** for exploring real-time forecasts.
* The model can be extended for **humidity, wind speed, or energy consumption forecasting** with minimal modification.

Would you like me to include the **exact `requirements.txt`** (all Python packages used in your app)?
I can generate it next so your project runs perfectly on any system.
