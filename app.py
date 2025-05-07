import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet

# Page config
st.set_page_config(page_title="📈 Stock Analyzer & Forecaster", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Select Feature", ["📊 Time Series Analysis", "🔮 Forecasting (LSTM + Prophet)"])

# CSV Upload
file = st.file_uploader("📤 Upload CSV with 'Date' and 'Close' columns", type=["csv"])

if not file:
    st.info("👆 Upload a CSV to continue.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[['Date', 'Close']].dropna().sort_values('Date').reset_index(drop=True)
except Exception as e:
    st.error(f"❌ Failed to read file: {e}")
    st.stop()

if 'Date' not in df.columns or 'Close' not in df.columns:
    st.error("❌ CSV must contain 'Date' and 'Close' columns.")
    st.stop()

# Analysis Page
if page == "📊 Time Series Analysis":
    df_ts = df.copy()
    df_ts.set_index('Date', inplace=True)

    st.title("📊 Historical Stock Time Series Analysis")
    st.write(df_ts.head())

    # Summary
    st.subheader("📋 Summary Statistics")
    st.dataframe(df_ts.describe())

    # Time Series
    st.subheader("📈 Raw Time Series")
    st.line_chart(df_ts['Close'])

    # Decomposition
    st.subheader("🪄 Seasonal Decomposition")
    period = st.slider("Select decomposition period (e.g. 30 for monthly)", min_value=2, max_value=90, value=30)
    try:
        decomposition = seasonal_decompose(df_ts['Close'], model='additive', period=period)
        fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        decomposition.observed.plot(ax=ax[0], title='Observed')
        decomposition.trend.plot(ax=ax[1], title='Trend')
        decomposition.seasonal.plot(ax=ax[2], title='Seasonal')
        decomposition.resid.plot(ax=ax[3], title='Residual')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Decomposition Error: {e}")
        decomposition = None

    # Anomaly Detection
    if decomposition is not None:
        st.subheader("⚠️ Anomaly Detection (Residual > 2σ)")
        residual = decomposition.resid.dropna()
        threshold = 2 * np.std(residual)
        anomalies = residual[np.abs(residual) > threshold]
        if anomalies.empty:
            st.success("✅ No significant anomalies detected.")
        else:
            st.dataframe(anomalies.reset_index().rename(columns={"resid": "Residual"}))

    # ADF Test
    st.subheader("📏 Augmented Dickey-Fuller Test")
    result = adfuller(df_ts['Close'].dropna())
    st.write(f"**ADF Statistic**: {result[0]:.4f}")
    st.write(f"**p-value**: {result[1]:.4f}")
    if result[1] < 0.05:
        st.success("✅ The series is stationary.")
    else:
        st.warning("⚠️ The series is NOT stationary.")

    # Transformations
    st.subheader("🔄 Transformation for Stationarity")
    option = st.selectbox("Choose transformation", ["Log Transform", "First Difference", "Log + Difference"])
    transformed = None
    if option == "Log Transform":
        transformed = np.log(df_ts['Close'])
    elif option == "First Difference":
        transformed = df_ts['Close'].diff().dropna()
    else:
        transformed = np.log(df_ts['Close']).diff().dropna()

    st.line_chart(transformed)

    result_trans = adfuller(transformed.dropna())
    st.write(f"**Transformed ADF Statistic**: {result_trans[0]:.4f}")
    st.write(f"**p-value**: {result_trans[1]:.4f}")
    if result_trans[1] < 0.05:
        st.success("✅ Transformed series is stationary.")
    else:
        st.warning("⚠️ Transformed series is still NOT stationary.")

# Forecasting Page
else:
    # st.title("📈 Stock Forecasting with LSTM or Prophet")
    # st.write("Upload a CSV with 'Date' and 'Close' columns.")

    # # File Upload
    # file = st.file_uploader("Upload CSV", type=["csv"])

    # if file is not None:
    #     df = pd.read_csv(file)

        # Check for required columns
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("CSV must contain 'Date' and 'Close' columns.")
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
            df = df[['Date', 'Close']].dropna().reset_index(drop=True)

            st.subheader("📊 Raw Data")
            st.write(df.tail())

            # Model selection
            model_choice = st.radio("Choose Forecasting Model:", ["LSTM", "Prophet"])

            if model_choice == "LSTM":
                # Scaling
                scaler = MinMaxScaler()
                df['Close_scaled'] = scaler.fit_transform(df[['Close']])

                # Sequence generation
                def create_sequences(data, seq_len=60):
                    X, y = [], []
                    for i in range(seq_len, len(data)):
                        X.append(data[i-seq_len:i])
                        y.append(data[i])
                    return np.array(X), np.array(y)

                seq_len = 60
                data = df['Close_scaled'].values
                X, y = create_sequences(data, seq_len)

                if len(X) == 0:
                    st.error("Not enough data to create sequences. Please upload a longer dataset.")
                    st.stop()

                # Train-test split
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                st.subheader("🚀 Training LSTM Model...")
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                with st.spinner("Training LSTM..."):
                    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

                # Prediction
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled)
                y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Metrics
                st.subheader("✅ Evaluation Metrics (LSTM)")
                st.write(f"**MAE**: {mean_absolute_error(y_actual, y_pred):.2f}")
                st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_actual, y_pred)):.2f}")
                st.write(f"**R² Score**: {r2_score(y_actual, y_pred):.4f}")

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(y_actual, label="Actual")
                ax.plot(y_pred, label="Predicted")
                ax.set_title("LSTM Forecast")
                ax.legend()
                st.pyplot(fig)

                # Future prediction
                def predict_future(model, last_sequence, n_days):
                    predictions = []
                    current_input = last_sequence
                    for _ in range(n_days):
                        pred_scaled = model.predict(current_input.reshape(1, seq_len, 1))
                        pred = scaler.inverse_transform(pred_scaled)
                        predictions.append(pred[0, 0])
                        current_input = np.append(current_input[1:], pred_scaled, axis=0)
                    return predictions

                n_days = 7
                last_sequence = X_test[-1]
                future_predictions = predict_future(model, last_sequence, n_days)

                st.subheader(f"📅 Future Predictions ({n_days} Days Ahead)")
                st.write(future_predictions)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(np.arange(len(y_actual)), y_actual, label="Actual")
                ax.plot(np.arange(len(y_actual), len(y_actual) + n_days), future_predictions, label="Future (LSTM)", linestyle="--")
                ax.legend()
                st.pyplot(fig)

            elif model_choice == "Prophet":
                st.subheader("🔮 Prophet Forecasting")
                df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})

                m = Prophet()
                with st.spinner("Training Prophet..."):
                    m.fit(df_prophet)

                future = m.make_future_dataframe(periods=7)
                forecast = m.predict(future)

                # Metrics
                actual = df_prophet['y'][-7:].values
                predicted = forecast['yhat'][-7:].values
                st.subheader("✅ Evaluation Metrics (Prophet)")
                st.write(f"**MAE**: {mean_absolute_error(actual, predicted):.2f}")
                st.write(f"**RMSE**: {np.sqrt(mean_squared_error(actual, predicted)):.2f}")
                st.write(f"**R² Score**: {r2_score(actual, predicted):.4f}")

                # Plot
                st.subheader("📉 Forecast Plot (Prophet)")
                fig1 = m.plot(forecast)
                st.pyplot(fig1)

                # Future predictions
                st.subheader("📅 Next 7 Days Forecast")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))
