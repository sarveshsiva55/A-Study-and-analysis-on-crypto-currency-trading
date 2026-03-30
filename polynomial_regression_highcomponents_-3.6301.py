import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv('crypto_data.csv')  # Replace with your dataset
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_14'] = df['Close'].rolling(window=14).mean()
df['SMA_21'] = df['Close'].rolling(window=21).mean()
df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

# RSI Calculation
window_length = 14
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD Calculation
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands Calculation
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
df['BB_Upper'] = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
df['BB_Lower'] = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)

# Average True Range (ATR)
df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

# Rate of Change (ROC)
df['ROC'] = df['Close'].pct_change(periods=12) * 100

# Lagged Features
df['Close_1'] = df['Close'].shift(1)
df['Close_2'] = df['Close'].shift(2)
df['Close_3'] = df['Close'].shift(3)

# Drop NaN values after feature engineering
df.dropna(inplace=True)

# Select Features and Target Variable
features = df[['Open', 'High', 'Low', 'Volume', 'SMA_7', 'SMA_14', 'SMA_21', 'EMA_7', 'EMA_14', 'EMA_21', 'RSI', 'MACD', 'MACD_signal', 'BB_Upper', 'BB_Lower', 'ATR', 'ROC', 'Close_1', 'Close_2', 'Close_3']]
target = df['Close']

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# Feature Selection (Recursive Feature Elimination)
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=12)  # Keep only the best 12 features
X_train_selected = rfe.fit_transform(X_train, y_train)
X_test_selected = rfe.transform(X_test)

# Apply Polynomial Regression (Degree = 3 for Non-Linearity)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

# Train Polynomial Regression Model
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)

# Make Predictions
y_pred_scaled = lr_poly.predict(X_test_poly)

# Convert Predictions Back to Original Scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model Performance
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

# Print Metrics
print(f'Optimized Polynomial Regression Model Performance:')
print(f'Accuracy (R² Score): {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.legend()
plt.title('Optimized Polynomial Regression for Crypto Price Prediction')
plt.xlabel('Days')
plt.ylabel('Bitcoin Price')
plt.show()
