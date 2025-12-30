import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the US Arrests data
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
X = pd.read_csv(url, index_col=0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)  # coordinates of states

#loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # variable vectors

# Plot
fig, ax = plt.subplots()

# Scatter of states
ax.scatter(scores[:, 0], scores[:, 1], s=20, color="tab:blue", alpha=0.7)

# Label each state
# X.index has the state names
for i, state in enumerate(X.index):
    ax.text(scores[i, 0], scores[i, 1], state, fontsize=8, va="center", ha="left")


# Scale arrows to fit nicely
# arrow_scale = 2.5  # increase/decrease if arrows are too small/large
#for i, var in enumerate(X.columns):
#    ax.arrow(0, 0,
#             loadings[i, 0] * arrow_scale,
#             loadings[i, 1] * arrow_scale,
#             color="tab:red", width=0.005, head_width=0.15, length_includes_head=True)
#    ax.text(loadings[i, 0] * arrow_scale * 1.1,
#            loadings[i, 1] * arrow_scale * 1.1,
#            var, color="tab:red", fontsize=9, ha="center", va="center")

# Axes and aesthetics
#ax.axhline(0, color="gray", linewidth=0.8)
#ax.axvline(0, color="gray", linewidth=0.8)
#ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
#ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
#ax.set_title("PCA Biplot: US Arrests")

# Set limits to include both points and arrows
#all_x = np.concatenate([scores[:, 0], loadings[:, 0] * arrow_scale])
#all_y = np.concatenate([scores[:, 1], loadings[:, 1] * arrow_scale])
#pad_x = 0.1 * (all_x.max() - all_x.min())
#pad_y = 0.1 * (all_y.max() - all_y.min())
#ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
#ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)


ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Biplot: US Arrests")

plt.tight_layout()
plt.show()