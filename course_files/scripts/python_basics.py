# Python Tutorial: Importing Packages, Reading Data, and Plotting
# This tutorial covers the basics of data analysis in Python
# Modified to load data from GitHub URL

# ============================================================================
# 1. IMPORTING PACKAGES
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 2. READING DATA WITH PANDAS FROM GITHUB
# ============================================================================

print("\n Loading data from GitHub...")

# GitHub URL for the diabetes data
# Convert from GitHub web URL to raw data URL
github_url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/diabetes_sample_data.csv"

# Read CSV file directly from GitHub
diabetes_data = pd.read_csv(github_url)
    
# Display basic information about the data
print("\nData shape:", diabetes_data.shape)
print("\nFirst 5 rows:")
print(diabetes_data.head())
        
print("\nBasic statistics:")
print(diabetes_data.describe())
    
    
# ============================================================================
# 3. PLOTTING WITH MATPLOTLIB
# ============================================================================

# Plot 1: Histogram of Age
plt.figure(figsize=(10, 6))
plt.hist(diabetes_data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Age', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
#plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
