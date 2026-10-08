# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# 2. Load dataset
data = pd.read_csv("../data/Mall_Customers.csv")

# 3. Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Find the best K using Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

# 5. Plot Elbow
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 6. Create K-Means model
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# 7. Fit and create clusters
data["Cluster"] = model.fit_predict(X)

# 8. Display results
print(data.head())

# 9. Visualize clusters
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Clusters")
plt.show()