import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('C:/Users/manmo/OneDrive/Documents/Visual studio code/ml/cereal.csv')

# Data Cleaning and Preprocessing
data_cleaned = data.dropna()  # Drop missing values
data_cleaned = data_cleaned.drop_duplicates()  # Remove duplicates

# Linear Regression: Predict cereal ratings
# Feature selection for regression
regression_features = ['calories', 'protein', 'fat', 'fiber', 'carbo', 'potass', 'sugars', 'sodium', 'vitamins']
X_regression = data_cleaned[regression_features]
y_regression = data_cleaned['rating']

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.9, random_state=42)

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)

# Predictions and evaluation
y_pred_reg = linear_model.predict(X_test_reg)
print("\nLinear Regression Results:")
print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("R-squared:", r2_score(y_test_reg, y_pred_reg))


# Visualize Linear Regression results (Predicted vs Actual ratings)
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color='blue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2)
plt.title("Linear Regression: Predicted vs Actual Ratings")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.show()
