


# !pip install pca

# Load data

from pca import pca
import pandas as pd

# Load the US Arrests data
# Read the USArrests data directly from the GitHub raw URL
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
df = pd.read_csv(url, index_col=0)

print("US Arrests Data (first 5 rows):")
print(df.head())
print("\nData shape:", df.shape)

#Normalize the data


from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()
df_scaled = scaler_standard.fit_transform(df)

print("\nData shape after normalization:", df_scaled.shape)

#Perform PCA

model = pca(n_components=4)
out = model.fit_transform(df_scaled)
ax = model.biplot(n_feat=len(df.columns), legend=False)




from pca import pca
import matplotlib.pyplot as plt

# Fit PCA (2 components for a 2D plot)
model = pca(n_components=2, normalize = True)
out = model.fit_transform(df)

# Make the biplot
ax = model.biplot(n_feat=len(df.columns), legend=False)

# Add point labels (uses the row names from the dataset)
for name, x, y in zip(df.index, out['PC']['PC1'], out['PC']['PC2']):
    ax.text(x, y, name, fontsize=8)

plt.tight_layout()
plt.show()


# pip install pca scikit-learn pandas

import pandas as pd
from sklearn.datasets import load_iris
from pca import pca

# 1) Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
labels = [data.target_names[i] for i in data.target]

# 2) Fit PCA
model = pca(n_components=2)
_ = model.fit_transform(X, row_labels=labels)

# 3) Scatter plot with labels
model.scatter(label=True, legend=True)