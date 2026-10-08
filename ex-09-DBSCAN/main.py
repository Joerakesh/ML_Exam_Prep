# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 2. Load dataset
data = pd.read_csv("../data/Mall_Customers.csv")

# 3. Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Create DBSCAN model
model = DBSCAN(
    eps=0.3,
    min_samples=5
)

# 6. Create clusters
data["Cluster"] = model.fit_predict(X_scaled)

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
plt.title("DBSCAN Clustering")
plt.show()