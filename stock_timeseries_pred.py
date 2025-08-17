# Complete Time Series ML Models Examples
# Real-world implementations for stock price, sales, and temperature prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Generate sample datasets for demonstration
np.random.seed(42)

# 1. Stock price data (with trend and volatility)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 150, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365)
noise = np.random.normal(0, 5, 1000)
stock_prices = trend + seasonal + noise
stock_df = pd.DataFrame({'date': dates, 'price': stock_prices})

# 2. Sales data (with strong seasonality)
sales_trend = np.linspace(1000, 2000, 1000)
sales_seasonal = 500 * np.sin(2 * np.pi * np.arange(1000) / 52) + 200 * np.sin(2 * np.pi * np.arange(1000) / 7)
sales_noise = np.random.normal(0, 100, 1000)
sales_data = sales_trend + sales_seasonal + sales_noise
sales_df = pd.DataFrame({'date': dates, 'sales': sales_data})

print("=== TIME SERIES ML MODELS EXAMPLES ===\n")

# ============================================================================
# 1. ARIMA MODEL
# ============================================================================
print("1. ARIMA MODEL - Predicting Stock Prices")
print("-" * 50)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Prepare data
train_size = int(len(stock_prices) * 0.8)
train_data = stock_prices[:train_size]
test_data = stock_prices[train_size:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5,1,0))  # (p,d,q)
arima_fitted = arima_model.fit()

# Make predictions
arima_forecast = arima_fitted.forecast(steps=len(test_data))
arima_mse = np.mean((test_data - arima_forecast) ** 2)

print(f"ARIMA Model Summary:")
print(f"Order: (5,1,0)")
print(f"MSE on test set: {arima_mse:.2f}")
print(f"Sample predictions: {arima_forecast[:5]}")
print()

# ============================================================================
# 2. SARIMA MODEL
# ============================================================================
print("2. SARIMA MODEL - Predicting Sales with Seasonality")
print("-" * 50)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prepare data
train_sales = sales_data[:train_size]
test_sales = sales_data[train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train_sales, 
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 52))  # Weekly seasonality
sarima_fitted = sarima_model.fit()

# Make predictions
sarima_forecast = sarima_fitted.forecast(steps=len(test_sales))
sarima_mse = np.mean((test_sales - sarima_forecast) ** 2)

print(f"SARIMA Model Summary:")
print(f"Order: (1,1,1) x (1,1,1,52)")
print(f"MSE on test set: {sarima_mse:.2f}")
print(f"Sample predictions: {sarima_forecast[:5]}")
print()

# ============================================================================
# 3. EXPONENTIAL SMOOTHING
# ============================================================================
print("3. EXPONENTIAL SMOOTHING - Triple Exponential (Holt-Winters)")
print("-" * 50)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt-Winters model
hw_model = ExponentialSmoothing(train_sales, 
                                trend='add', 
                                seasonal='add', 
                                seasonal_periods=52)
hw_fitted = hw_model.fit()

# Make predictions
hw_forecast = hw_fitted.forecast(steps=len(test_sales))
hw_mse = np.mean((test_sales - hw_forecast) ** 2)

print(f"Holt-Winters Model Summary:")
print(f"Trend: additive, Seasonal: additive")
print(f"Seasonal periods: 52")
print(f"MSE on test set: {hw_mse:.2f}")
print(f"Sample predictions: {hw_forecast[:5]}")
print()

# ============================================================================
# 4. RANDOM FOREST
# ============================================================================
print("4. RANDOM FOREST - Stock Price Prediction with Features")
print("-" * 50)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def create_features(data, window=5):
    """Create lag features and technical indicators"""
    df = pd.DataFrame({'price': data})
    
    # Lag features
    for i in range(1, window+1):
        df[f'lag_{i}'] = df['price'].shift(i)
    
    # Moving averages
    df['ma_5'] = df['price'].rolling(5).mean()
    df['ma_10'] = df['price'].rolling(10).mean()
    
    # Technical indicators
    df['rsi'] = calculate_rsi(df['price'])
    df['volatility'] = df['price'].rolling(10).std()
    
    return df.dropna()

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Create features
feature_df = create_features(stock_prices)
X = feature_df.drop('price', axis=1).values
y = feature_df['price'].values

# Split data
train_size_rf = int(len(X) * 0.8)
X_train, X_test = X[:train_size_rf], X[train_size_rf:]
y_train, y_test = y[:train_size_rf], y[train_size_rf:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)
rf_mse = np.mean((y_test - rf_predictions) ** 2)

print(f"Random Forest Model Summary:")
print(f"Features: lag_1-5, MA_5, MA_10, RSI, volatility")
print(f"N_estimators: 100")
print(f"MSE on test set: {rf_mse:.2f}")
print(f"Feature importance (top 3): {dict(zip(['lag_1', 'lag_2', 'ma_5'], rf_model.feature_importances_[:3]))}")
print()

# ============================================================================
# 5. XGBOOST
# ============================================================================
print("5. XGBOOST - Advanced Gradient Boosting")
print("-" * 50)

try:
    import xgboost as xgb
    
    # Fit XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    xgb_predictions = xgb_model.predict(X_test_scaled)
    xgb_mse = np.mean((y_test - xgb_predictions) ** 2)
    
    print(f"XGBoost Model Summary:")
    print(f"N_estimators: 100, max_depth: 6, learning_rate: 0.1")
    print(f"MSE on test set: {xgb_mse:.2f}")
    print(f"Sample predictions: {xgb_predictions[:5]}")
    print()
    
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    print()

# ============================================================================
# 6. SUPPORT VECTOR REGRESSION (SVR)
# ============================================================================
print("6. SUPPORT VECTOR REGRESSION (SVR)")
print("-" * 50)

from sklearn.svm import SVR

# Fit SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Make predictions
svr_predictions = svr_model.predict(X_test_scaled)
svr_mse = np.mean((y_test - svr_predictions) ** 2)

print(f"SVR Model Summary:")
print(f"Kernel: RBF, C: 100, gamma: 0.1")
print(f"MSE on test set: {svr_mse:.2f}")
print(f"Sample predictions: {svr_predictions[:5]}")
print()

# ============================================================================
# 7. LSTM NEURAL NETWORK
# ============================================================================
print("7. LSTM NEURAL NETWORK - Deep Learning Approach")
print("-" * 50)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    def create_lstm_sequences(data, seq_length=60):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    # Scale data for LSTM
    lstm_scaler = MinMaxScaler()
    scaled_prices = lstm_scaler.fit_transform(stock_prices.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_lstm, y_lstm = create_lstm_sequences(scaled_prices, seq_length=60)
    
    # Split data
    train_size_lstm = int(len(X_lstm) * 0.8)
    X_train_lstm = X_lstm[:train_size_lstm].reshape((train_size_lstm, 60, 1))
    X_test_lstm = X_lstm[train_size_lstm:].reshape((len(X_lstm) - train_size_lstm, 60, 1))
    y_train_lstm = y_lstm[:train_size_lstm]
    y_test_lstm = y_lstm[train_size_lstm:]
    
    # Build LSTM model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model (reduced epochs for demo)
    history = lstm_model.fit(X_train_lstm, y_train_lstm, 
                            batch_size=32, epochs=5, 
                            validation_split=0.1, verbose=0)
    
    # Make predictions
    lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
    lstm_predictions = lstm_scaler.inverse_transform(lstm_predictions_scaled).flatten()
    y_test_actual = lstm_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
    lstm_mse = np.mean((y_test_actual - lstm_predictions) ** 2)
    
    print(f"LSTM Model Summary:")
    print(f"Architecture: LSTM(50) -> Dropout -> LSTM(50) -> Dense(25) -> Dense(1)")
    print(f"Sequence length: 60")
    print(f"MSE on test set: {lstm_mse:.2f}")
    print(f"Sample predictions: {lstm_predictions[:5]}")
    print()
    
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    print()

# ============================================================================
# 8. FACEBOOK PROPHET
# ============================================================================
print("8. FACEBOOK PROPHET - Automated Forecasting")
print("-" * 50)

try:
    from prophet import Prophet
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': dates[:train_size],
        'y': sales_data[:train_size]
    })
    
    # Fit Prophet model
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    prophet_model.fit(prophet_df)
    
    # Make future dataframe
    future = prophet_model.make_future_dataframe(periods=len(test_sales), freq='D')
    
    # Make predictions
    prophet_forecast = prophet_model.predict(future)
    prophet_predictions = prophet_forecast['yhat'].iloc[train_size:].values
    prophet_mse = np.mean((test_sales - prophet_predictions) ** 2)
    
    print(f"Prophet Model Summary:")
    print(f"Seasonalities: yearly, weekly enabled")
    print(f"Changepoint prior scale: 0.05")
    print(f"MSE on test set: {prophet_mse:.2f}")
    print(f"Sample predictions: {prophet_predictions[:5]}")
    print()
    
except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
    print()

# ============================================================================
# 9. GRU (Alternative to LSTM)
# ============================================================================
print("9. GRU NEURAL NETWORK - Simpler Alternative to LSTM")
print("-" * 50)

try:
    from tensorflow.keras.layers import GRU
    
    # Build GRU model
    gru_model = Sequential([
        GRU(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    gru_model.fit(X_train_lstm, y_train_lstm, 
                  batch_size=32, epochs=5, verbose=0)
    
    # Make predictions
    gru_predictions_scaled = gru_model.predict(X_test_lstm)
    gru_predictions = lstm_scaler.inverse_transform(gru_predictions_scaled).flatten()
    gru_mse = np.mean((y_test_actual - gru_predictions) ** 2)
    
    print(f"GRU Model Summary:")
    print(f"Architecture: GRU(50) -> Dropout -> GRU(50) -> Dense(25) -> Dense(1)")
    print(f"MSE on test set: {gru_mse:.2f}")
    print(f"Sample predictions: {gru_predictions[:5]}")
    print()
    
except:
    print("GRU requires TensorFlow. Skipping...")
    print()

# ============================================================================
# 10. ENSEMBLE METHOD
# ============================================================================
print("10. ENSEMBLE METHOD - Combining Multiple Models")
print("-" * 50)

# Simple ensemble of available predictions
ensemble_predictions = []
weights = []

if 'rf_predictions' in locals():
    ensemble_predictions.append(rf_predictions)
    weights.append(0.3)

if 'svr_predictions' in locals():
    ensemble_predictions.append(svr_predictions)
    weights.append(0.2)

if 'arima_forecast' in locals() and len(arima_forecast) == len(y_test):
    ensemble_predictions.append(arima_forecast[:len(y_test)])
    weights.append(0.5)

if ensemble_predictions:
    # Weighted average ensemble
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    ensemble_result = np.average(ensemble_predictions, axis=0, weights=weights)
    ensemble_mse = np.mean((y_test - ensemble_result) ** 2)
    
    print(f"Ensemble Model Summary:")
    print(f"Models combined: {len(ensemble_predictions)}")
    print(f"Weights: {weights}")
    print(f"MSE on test set: {ensemble_mse:.2f}")
    print(f"Sample predictions: {ensemble_result[:5]}")
    print()

# ============================================================================
# REAL-WORLD TIPS AND BEST PRACTICES
# ============================================================================
print("=" * 60)
print("REAL-WORLD TIPS AND BEST PRACTICES")
print("=" * 60)

print("""
1. DATA PREPROCESSING:
   - Handle missing values (interpolation, forward fill)
   - Remove outliers or use robust models
   - Check for stationarity (use differencing if needed)
   - Feature scaling for ML models

2. MODEL SELECTION GUIDELINES:
   - ARIMA/SARIMA: Univariate, clear patterns
   - Prophet: Business data with holidays/events
   - LSTM/GRU: Complex sequential dependencies
   - Random Forest/XGBoost: Multivariate with engineered features
   - Ensemble: Combine different model strengths

3. FEATURE ENGINEERING:
   - Lag features (previous values)
   - Moving averages and technical indicators
   - Seasonal decomposition
   - External factors (weather, holidays, events)

4. VALIDATION STRATEGIES:
   - Time series split (no random shuffling!)
   - Walk-forward validation
   - Cross-validation with time gaps

5. COMMON PITFALLS:
   - Data leakage (using future information)
   - Not accounting for seasonality
   - Overfitting on small datasets
   - Ignoring business context

6. PERFORMANCE METRICS:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - MAPE (Mean Absolute Percentage Error)
   - Directional accuracy for trading

7. DEPLOYMENT CONSIDERATIONS:
   - Model retraining frequency
   - Real-time vs batch predictions
   - Monitoring for concept drift
   - Fallback mechanisms
""")

print("\n=== END OF EXAMPLES ===")
print("Install required packages:")
print("pip install numpy pandas matplotlib scikit-learn statsmodels")
print("pip install tensorflow xgboost prophet")