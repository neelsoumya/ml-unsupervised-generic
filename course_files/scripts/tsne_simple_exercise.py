import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the US Arrests data
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
df = pd.read_csv(url, index_col=0)

# Prepare the data for t-SNE
X = df.values  # Convert to numpy array

# Run t-SNE to reduce the data to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# Plot the results
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.title("t-SNE visualization of US Arrests data")
plt.show()

#######################
# more complex code
#######################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the US Arrests data
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
df = pd.read_csv(url, index_col=0)

# Prepare the data for t-SNE
X = df.values  # Convert to numpy array

# Run t-SNE to reduce the data to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# Plot the results
plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

# Add state labels to the points
for i, state in enumerate(df.index):
    plt.annotate(state, (X_2d[i, 0], X_2d[i, 1]), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8)

plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.title("t-SNE visualization of US Arrests data")
plt.tight_layout()
plt.show()


