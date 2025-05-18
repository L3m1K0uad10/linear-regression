# Real Estate Price Prediction (Taiwan)

This project is based on a real estate valuation dataset collected from the city of Taipei, Taiwan. The goal is to predict the **house price per unit area** based on various features such as proximity to the MRT station, the number of convenience stores nearby, and more.

The dataset contains **414 instances** and **7 attributes**, including the target variable (house price per unit area).

### 🔍 Project Aspiration

Build a **linear regression model from scratch** using only NumPy and compare its performance with **scikit-learn's LinearRegression** to validate correctness.

### 📊 Dataset Information

- Source: [Real estate valuation data set – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)
- Format: CSV
- Number of Instances: 414
- Features:
  - `X1 transaction date`
  - `X2 house age`
  - `X3 distance to the nearest MRT station`
  - `X4 number of convenience stores`
  - `X5 latitude`
  - `X6 longitude`
  - `Y house price of unit area` (target)

### ⚙️ Implementation Details

- Implemented a custom `LinearRegression` class that:
  - Computes weights using the **normal equation**.
  - Supports prediction for new instances.
- Used **StandardScaler** from `sklearn.preprocessing` to normalize input features.
- Compared predictions with `sklearn.linear_model.LinearRegression`.

### ✅ Results

After training the model on all but one data point and testing on the last row:

<pre> ```text Custom LinearRegression Prediction: 53.73984570083941 Scikit-learn LinearRegression Prediction: 53.739845700839425 Actual Price: 63.9 ``` </pre>

> The result of the custom implementation is **almost identical** to that of scikit-learn, validating the correctness of the manual model.

### 🛠️ Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn

