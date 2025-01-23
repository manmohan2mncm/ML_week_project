"""
The binomial distribution here models the number of products exceeding the rating threshold across multiple experiments.
It provides insights into the likelihood and variability of success counts in scenarios with binary outcomes, 
helping to predict performance or inform decisions based on probabilities.

it answers questions like:
If I launch 100 products, how many are likely to succeed?
What is the minimum number of successes I can expect with high confidence?
"""

# Binomial Distribution
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

# Simulate a binomial distribution
n_trials = 100  # Number of trials
n_experiments = 1000  # Number of experiments
binomial_results = np.random.binomial(n=n_trials, p=success_prob, size=n_experiments)

# Output basic stats
print("Success Probability (p):", success_prob)
print("Mean of Binomial Distribution:", binomial_results.mean())
print("Standard Deviation of Binomial Distribution:", binomial_results.std())

# Visualize Binomial Distribution
plt.figure(figsize=(8, 5))
plt.hist(binomial_results, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.show()