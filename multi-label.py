import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('C:/Users/manmo/OneDrive/Documents/Visual studio code/ml/cereal.csv')

# Data Cleaning and Preprocessing
data_cleaned = data.dropna()  # Drop missing values
data_cleaned = data_cleaned.drop_duplicates()  # Remove duplicates

# Define rating categories (Low, Medium, High)
rating_bins = [0, 30, 60, 100]  # Define ranges
rating_labels = ['Low', 'Medium', 'High']  # Define labels
data_cleaned['rating_category'] = pd.cut(data_cleaned['rating'], bins=rating_bins, labels=rating_labels)

# Feature selection for multi-class classification
multi_class_features = ['calories', 'protein', 'fat', 'fiber', 'carbo', 'potass', 'sugars', 'sodium', 'vitamins']
X_multi = data_cleaned[multi_class_features]
y_multi = data_cleaned['rating_category']

# Train-test split
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.8, random_state=42)

# Train a Logistic Regression model
multi_class_model = RandomForestClassifier()
multi_class_model.fit(X_train_multi, y_train_multi)

# Predictions and evaluation
y_pred_multi = multi_class_model.predict(X_test_multi)
print("\nMulti-Class Classification Results:")
print("Accuracy:", accuracy_score(y_test_multi, y_pred_multi))
print("Classification Report:")
print(classification_report(y_test_multi, y_pred_multi))

# Confusion Matrix Visualization
conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi, labels=rating_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_multi, annot=True, fmt='d', cmap='Greens', xticklabels=rating_labels, yticklabels=rating_labels)
plt.title("Confusion Matrix for Multi-Class Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
