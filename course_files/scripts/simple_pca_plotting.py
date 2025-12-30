################################################################
# Simple PCA plotting (Day 1)
################################################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the US Arrests data
# Read the USArrests data directly from the GitHub raw URL
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
X = pd.read_csv(url, index_col=0)

# what is in the data?
X.head()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)



plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on crime data")
plt.show()



# States come from the DataFrame index
states = X.index

# Map each state to a color using categorical codes
state_codes = pd.Categorical(states).codes

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=state_codes, cmap='tab20', s=60, edgecolor='k', alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on crime data — colored by State")
plt.show()


import seaborn as sns

df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "State": X.index})

plt.figure()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="State", palette="tab20", s=60, edgecolor="k", legend=False)
plt.title("PCA on crime data — colored by State")
plt.show()