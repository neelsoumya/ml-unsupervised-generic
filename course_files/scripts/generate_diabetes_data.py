# This script generates a synthetic dataset of 100 diabetes patients with random values for age, glucose, BMI, blood pressure, and diabetes status.
# The data is saved as 'diabetes_sample_data.csv' for use in data analysis or machine learning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 2. READING DATA WITH PANDAS
# ============================================================================

# Let's create some sample data first
print("\n📊 Creating sample data...")

# Create sample data for diabetes patients
np.random.seed(42)  # For reproducible results
n_patients = 100

data = {
    'patient_id': range(1, n_patients + 1),
    'age': np.random.normal(55, 15, n_patients).round(1),
    'glucose': np.random.normal(140, 30, n_patients).round(1),
    'bmi': np.random.normal(28, 5, n_patients).round(1),
    'blood_pressure': np.random.normal(80, 10, n_patients).round(1),
    'diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3])
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('diabetes_sample_data.csv', index=False)
