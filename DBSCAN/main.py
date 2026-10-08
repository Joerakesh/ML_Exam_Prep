import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("../data/heart_disease_dataset.csv")

# Separate features
X = df.drop("Heart Disease", axis=1)

# Identify columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# DBSCAN
model = DBSCAN(eps=1.5, min_samples=5)
labels = model.fit_predict(X_processed)

# Add cluster labels
df["Cluster"] = labels

# Print clusters
print("Cluster Labels:")
print(df["Cluster"].value_counts().sort_index())

# Count noise points
noise_points = df[df["Cluster"] == -1]
print("\nNumber of Noise Points:", len(noise_points))

# Count clusters
number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of Clusters:", number_of_clusters)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plot
plt.figure(figsize=(9, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering")
plt.colorbar(label="Cluster")
plt.show()