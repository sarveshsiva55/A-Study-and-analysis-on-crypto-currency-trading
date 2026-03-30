import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

data = pd.read_csv('crypto_data.csv').dropna()

features, label = ['Open', 'High', 'Low', 'Volume'], 'Close'

poly = PolynomialFeatures(degree=4, include_bias=False)
data_poly = poly.fit_transform(data[features])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data_poly, data[label], test_size=0.1, random_state=42)

# Build optimized model with feature scaling and polynomial regression with Ridge regularization
model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 20), store_cv_values=True))
model.fit(x_train, y_train)

# Predictions and evaluation
predictions = model.predict(x_test)
rmse, r2 = sqrt(mean_squared_error(y_test, predictions)), r2_score(y_test, predictions)
accuracy = 1 - (rmse / np.mean(y_test))

# Print results
print("OPTIMIZED POLYNOMIAL REGRESSION MODEL PERFORMANCE:")
print(f'Accuracy: {accuracy:.4f}\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}')

# Plot actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label='Actual', linestyle='-', marker='o')
plt.plot(range(len(predictions)), predictions, label='Predicted', linestyle='--', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Close Price')
plt.title('Optimized Polynomial Regression for Cryptocurrency Prediction')
plt.legend()
plt.grid()
plt.show()
