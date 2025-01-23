"""
The results give insights into the spread and frequency of sugar content in cereals, 
which can be useful for dietary analysis or product comparison.

it answers questions like:
What is the expected sugar distribution in a sample of cereals?
How often do cereals deviate from the average sugar content?
Are there many cereals with extremely high or low sugar content?

"""

# Poisson Distribution
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

# Poisson Distribution
# Use the original "sugars" column (non-normalized) for Poisson distribution
sugars_mean = data_cleaned['sugars'].mean()

# Simulate a Poisson distribution
poisson_results = np.random.poisson(lam=max(sugars_mean, 0), size=1000)  # Ensure lambda is non-negative

# Output basic stats
print("Mean of Poisson Distribution:", poisson_results.mean())
print("Standard Deviation of Poisson Distribution:", poisson_results.std())

# Visualize Poisson Distribution
plt.figure(figsize=(8, 5))
plt.hist(poisson_results, bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.title("Poisson Distribution")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.show()
