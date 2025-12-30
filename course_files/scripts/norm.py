import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Generate age and weight data
np.random.seed(42)
age = np.random.normal(45, 15, 100)  # 100 people, mean age 45, std 15
age = np.clip(age, 18, 80)  # Keep ages between 18-80

weight = 70 + (age - 45) * 0.3 + np.random.normal(0, 10, 100)  # Weight correlated with age
weight = np.clip(weight, 45, 120)  # Keep weights between 45-120 kg

print("Original data:")
print(f"Age: mean={age.mean():.1f}, std={age.std():.1f}")
print(f"Weight: mean={weight.mean():.1f}, std={weight.std():.1f}")

# 2. Normalize the data
scaler = StandardScaler()
data = np.column_stack((age, weight))
normalized_data = scaler.fit_transform(data)

age_normalized = normalized_data[:, 0]
weight_normalized = normalized_data[:, 1]

# Separate plots for each visualization

plt.figure(figsize=(6, 4))
plt.scatter(age, weight, alpha=0.6)
plt.title('Original Data')
plt.xlabel('Age')
plt.ylabel('Weight (kg)')
plt.grid(True, alpha=0.3)
plt.show()

# Histogram: Age (Original)
plt.figure(figsize=(6, 4))
plt.hist(age, bins=20, alpha=0.7)
plt.title('Age Distribution (Original)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Scatter plot: Normalized data
plt.figure(figsize=(6, 4))
plt.scatter(age_normalized, weight_normalized, alpha=0.6)
plt.title('Normalized Data')
plt.xlabel('Age (Z-score)')
plt.ylabel('Weight (Z-score)')
plt.grid(True, alpha=0.3)
plt.show()

# Histogram: Age (Normalized)
plt.figure(figsize=(6, 4))
plt.hist(age_normalized, bins=20, alpha=0.7)
plt.title('Age Distribution (Normalized)')
plt.xlabel('Age (Z-score)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.7)

plt.tight_layout()
plt.show()
