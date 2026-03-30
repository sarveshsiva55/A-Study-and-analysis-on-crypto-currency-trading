import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv('crypto_data.csv')  # Replace with your dataset
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()

# RSI Calculation
window_length = 14
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Drop NaN values after feature engineering
df.dropna(inplace=True)

# Select Features and Target Variable
features = df[['Open', 'High', 'Low', 'Volume', 'SMA_7', 'EMA_7', 'RSI']]
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
rfe = RFE(lr, n_features_to_select=5)  # Keep only the best 5 features
X_train_selected = rfe.fit_transform(X_train, y_train)
X_test_selected = rfe.transform(X_test)

# Train Linear Regression Model
lr.fit(X_train_selected, y_train)

# Make Predictions
y_pred_scaled = lr.predict(X_test_selected)

# Convert Predictions Back to Original Scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model Performance
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

# Print Metrics
print(f'Optimized Linear Regression Model Performance:')
print(f'Accuracy (R² Score): {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.legend()
plt.title('Optimized Linear Regression for Crypto Price Prediction')
plt.xlabel('Days')
plt.ylabel('Bitcoin Price')
plt.show()
