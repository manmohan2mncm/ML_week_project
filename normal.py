"""
he normal distribution simulated here models the normalized rating values, 
allowing you to visualize and statistically analyze how ratings behave in the dataset. 

it answers questions like:
What percentage of products will likely have ratings above a certain threshold?
How likely is it to achieve extreme ratings (very high or very low)?
"""

# Normal Distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/manmo/OneDrive/Documents/Visual studio code/ml/cereal.csv')

# Data Cleaning and Preprocessing
# Handle missing values
data_cleaned = data.dropna()

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()

# Identify numeric columns automatically
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Normalize numeric columns
data_normalized = data_cleaned.copy()
data_normalized[numeric_columns] = (data_cleaned[numeric_columns] - 
                                    data_cleaned[numeric_columns].mean()) / data_cleaned[numeric_columns].std()

# Example: Simulating the success of a product rating threshold (e.g., rating > 3.0)
threshold = 0  # Normalized threshold for rating > 3.0
success_prob = (data_normalized['rating'] > threshold).mean()

# Normal Distribution
# Example: Using the "rating" column to create a normal distribution
rating_mean = data_normalized['rating'].mean()
rating_std = data_normalized['rating'].std()

# Simulate a normal distribution
normal_results = np.random.normal(loc=rating_mean, scale=rating_std, size=1000)

# Output basic stats
print("Mean of Normal Distribution:", normal_results.mean())
print("Standard Deviation of Normal Distribution:", normal_results.std())

# Visualize Normal Distribution
plt.figure(figsize=(8, 5))
plt.hist(normal_results, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()