"""

Tells how accurately we can classify cereals as high sugar or low sugar based on nutritional features.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('C:/Users/manmo/OneDrive/Documents/Visual studio code/ml/cereal.csv')

# Data Cleaning and Preprocessing
data_cleaned = data.dropna()  # Drop missing values
data_cleaned = data_cleaned.drop_duplicates()  # Remove duplicates

# Binary Classification: Classify cereals as "High Sugar" or "Low Sugar"
# Define a threshold for high sugar content (e.g., mean sugar value)
sugar_threshold = data_cleaned['sugars'].mean()
data_cleaned['high_sugar'] = (data_cleaned['sugars'] > sugar_threshold).astype(int)  # Binary target

# Feature selection for classification
classification_features = ['calories', 'protein', 'fat', 'fiber', 'carbo', 'potass', 'sodium', 'vitamins']
X_classification = data_cleaned[classification_features]
y_classification = data_cleaned['high_sugar']

# Train-test split for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_classification, y_classification, test_size=0.8, random_state=42)

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_clf, y_train_clf)

# Predictions and evaluation
y_pred_clf = logistic_model.predict(X_test_clf)
print("Binary Classification Results:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Sugar (0)', 'High Sugar (1)'], 
            yticklabels=['Low Sugar (0)', 'High Sugar (1)'])
plt.title("Confusion Matrix for Binary Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()