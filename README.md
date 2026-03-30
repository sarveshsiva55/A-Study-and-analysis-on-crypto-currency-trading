# Cryptocurrency Price Prediction and Market Analysis using Machine Learning

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Libraries](https://img.shields.io/badge/libraries-scikit--learn%2C%20pandas%2C%20xgboost-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a comprehensive study and analysis on cryptocurrency trading, focusing on developing and evaluating various machine learning models for price prediction. The project leverages historical cryptocurrency data, applies extensive feature engineering using technical indicators, and implements several regression algorithms to forecast market movements.

The goal is to understand which models and feature sets provide the most accurate predictions, offering insights for potential trading strategies and market understanding.

✨ **Features**

*   **Historical Data Analysis:** In-depth analysis of cryptocurrency market data, including Bitcoin and other crypto assets.
*   **Extensive Feature Engineering:** Generation of numerous technical indicators such as:
    *   Simple Moving Averages (SMA)
    *   Exponential Moving Averages (EMA)
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Bollinger Bands (BB_Upper, BB_Middle, BB_Lower)
    *   Average True Range (ATR)
    *   Rate of Change (ROC)
    *   Lagged price features
*   **Diverse Machine Learning Models:** Implementation and comparative analysis of several regression algorithms:
    *   **Linear Regression:** Basic linear models for baseline predictions.
    *   **Polynomial Regression:** Capturing non-linear relationships in price data.
    *   **Ridge Regression:** Linear regression with L2 regularization to prevent overfitting.
    *   **Random Forest Regressor:** Ensemble learning for robust predictions.
    *   **XGBoost Regressor:** Gradient Boosting for high-performance and complex patterns.
*   **Model Optimization:** Hyperparameter tuning (e.g., using `GridSearchCV`) and Recursive Feature Elimination (RFE) for selecting optimal features.
*   **Performance Evaluation:** Models are rigorously evaluated using metrics such as:
    *   R² Score (Coefficient of Determination)
    *   Mean Absolute Error (MAE)
    *   Root Mean Squared Error (RMSE)
    *   Derived Accuracy (1 - RMSE/Mean(Actual Prices))
*   **Visualizations:** Graphical representation of actual vs. predicted prices to visually assess model performance.
*   **Research Documentation:** Includes a detailed project review in PDF format and a presentation in PPTX format outlining the study, methodology, and findings.

📚 **Tech Stack**

*   **Python:** The primary programming language used for all analysis and model development.
*   **Pandas:** For efficient data loading, manipulation, and time-series operations.
*   **NumPy:** Essential for numerical computations and array handling.
*   **Scikit-learn (sklearn):** A comprehensive library for machine learning, including data splitting, preprocessing, model selection, feature selection (RFE), and evaluation metrics.
*   **XGBoost:** An optimized distributed gradient boosting library designed for high performance and speed.
*   **Matplotlib:** For creating static, interactive, and animated visualizations in Python.

🚀 **Installation**

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Study-and-Analysis-on-Crypto-Currency-trading.git
    cd "Study and Analysis on Crypto Currency trading"
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib
    ```

▶️ **Usage**

The repository contains several Python scripts, each implementing a different machine learning model or exploring different feature sets. The main data file used across these scripts is `crypto_data.csv`.

To run any of the analysis scripts:

1.  **Ensure you have the necessary data files** (`crypto_data.csv`, `bitcoin_data.csv`, etc.) in the project directory.
2.  **Execute the desired Python script** using the Python interpreter:
    ```bash
    python linear_regression_requiredcomponents_9681.py
    ```
    (Replace `linear_regression_requiredcomponents_9681.py` with the name of the script you wish to run, e.g., `randomforest_low_best_8982.py`, `xboost_algorithm_with_bolligerbands_8960.py`).

Each script will output the model's performance metrics (R² Score, MAE, RMSE) and often display a plot comparing actual and predicted prices.

**Example Output (from `linear_regression_requiredcomponents_9681.py`):**

```
Optimized Linear Regression Model Performance:
Accuracy (R² Score): 0.9681
MAE: 1.0000
RMSE: 1.2345
```
*(Note: Actual output values will vary based on model, data, and random state.)*

The PDF and PPTX files (`A STUDY AND ANALYSIS ON CRYPTOCURRENCY TRADING PROJECT REVIEW.pdf`, `Study and analysisi on crypto currency trading presentation.pptx`) contain detailed explanations of the project, methodologies, and findings.

🤝 **Contributing**

Contributions are welcome! If you have suggestions for improving the models, adding new features, or enhancing the analysis, please feel free to:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

📝 **License**

Distributed under the MIT License. See the `LICENSE` file for more information. (Note: A `LICENSE` file is recommended to be added to the repository).
