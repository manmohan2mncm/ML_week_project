import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'your_dataset.csv' with the actual path to your dataset
file_path = "C:/Users/manmo/OneDrive/Documents/Visual studio code/ml/cereal.csv"

# Read the dataset
df = pd.read_csv(file_path)

# Print initial dataset information
print("Initial dataset info:")
print(df.info())

# Step 1: Handle non-numeric data
# Identify numeric columns only
numeric_columns = df.select_dtypes(include=['number']).columns
print("\nNumeric columns:", numeric_columns)

# Step 2: Fill missing values for numeric columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 3: Display summary statistics for verification
print("\nSummary statistics for numeric columns after filling missing values:")
print(df[numeric_columns].describe())

# Step 4: Visualize data
# Example: Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Example: Boxplot for numeric columns to check for outliers
plt.figure(figsize=(15, 5))
df[numeric_columns].boxplot()
plt.title("Boxplot of Numeric Columns")
plt.show()

# Example: Pairplot for numeric columns
sns.pairplot(df[numeric_columns])
plt.show()