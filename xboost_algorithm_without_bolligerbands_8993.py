import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv('crypto_data.csv')  # Replace with your dataset
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering: Manually Compute Indicators (No TA-Lib)
df['SMA_7'] = df['Close'].rolling(window=7).mean()  # 7-day Simple Moving Average
df['SMA_21'] = df['Close'].rolling(window=21).mean()  # 21-day Simple Moving Average
df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()  # 7-day Exponential Moving Average
df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()  # 21-day Exponential Moving Average

# Relative Strength Index (RSI) Calculation (Manually)
window_length = 14
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD Calculation (Manually)
df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Drop NaN values after feature engineering
df.dropna(inplace=True)

# Select Features and Target Variable
features = df[['Open', 'High', 'Low', 'Volume', 'SMA_7', 'SMA_21', 'EMA_7', 'EMA_21', 'RSI', 'MACD', 'MACD_signal']]
target = df['Close']

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# Hyperparameter Tuning for XGBoost
params = {
    'learning_rate': [0.01, 0.05, 0.1],  
    'max_depth': [3, 6, 9],  
    'n_estimators': [500, 1000, 1500],  
    'subsample': [0.6, 0.8, 1.0],  
    'colsample_bytree': [0.6, 0.8, 1.0],  
    'gamma': [0, 0.1, 0.2],  
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

grid_search = GridSearchCV(xgb_model, param_grid=params, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

# Best Hyperparameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train XGBoost with Best Parameters
xgb_final = xgb.XGBRegressor(**best_params, objective='reg:squarederror', eval_metric='rmse')
xgb_final.fit(X_train, y_train.ravel())

# Make Predictions
y_pred_scaled = xgb_final.predict(X_test)

# Convert Predictions Back to Original Scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate Model Performance
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

# Print Metrics
print(f'Optimized XGBoost Model Performance:')
print(f'Accuracy (R² Score): {r2:.4f}')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12,6))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.legend()
plt.title('Optimized XGBoost without Bollinger Bands')
plt.xlabel('Days')
plt.ylabel('Bitcoin Price')
plt.show()
