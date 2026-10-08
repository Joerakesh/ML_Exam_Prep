# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 2. Load dataset
data = pd.read_csv("../data/Mall_Customers.csv")

# 3. Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Create dendrogram
linked = linkage(X, method="ward")

dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# 5. Create Hierarchical Clustering model
model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

# 6. Create clusters
data["Cluster"] = model.fit_predict(X)

# 7. Display results
print(data.head())

# 8. Visualize clusters
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering")
plt.show()