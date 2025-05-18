import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR

from linear_regression import LinearRegression



# Real estate dataset
df = pd.read_csv("dataset/Real_estate.csv")
## preprocessing

# remove the column "No" which is at 0
data = df.drop(columns = ["No"]).to_numpy()

rows, cols = data.shape


# Split data into X and y
X = data[:, :-1]
y = data[:, -1]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combine scaled X and y
data_scaled = np.column_stack((X_scaled, y))

# Remove last row (test)
test_x = X_scaled[-1]
data_scaled = data_scaled[:-1]

# Train model
model = LinearRegression(data_scaled, index=len(data_scaled[0])-1)
model.linear_model()

# Predict
pred = model.predict(test_x)
print(f"Predicted: {pred}, Actual: {y[-1]}")

lr = SklearnLR()
lr.fit(X_scaled[:-1], y[:-1])
sk_pred = lr.predict([X_scaled[-1]])
print(f"Sklearn Prediction: {sk_pred[0]}")