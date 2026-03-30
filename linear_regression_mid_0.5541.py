import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Load dataset
data = pd.read_csv('crypto_data.csv').dropna()

# Feature selection
features, label = ['Open', 'High', 'Low', 'Volume'], 'Close'

# Maximize test size and reduce model complexity
x_train, x_test, y_train, y_test = train_test_split(data[features], data[label], test_size=0.5, random_state=42)

# Use an extremely high alpha to severely weaken model performance
model = make_pipeline(StandardScaler(), Ridge(alpha=1000))  # Greatly increased alpha
model.fit(x_train, y_train)

# Predictions and evaluation
predictions = model.predict(x_test)
rmse, r2 = sqrt(mean_squared_error(y_test, predictions)), r2_score(y_test, predictions)
accuracy = 1 - (rmse / np.mean(y_test))

# Print results
print(f'Accuracy: {accuracy:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}')
