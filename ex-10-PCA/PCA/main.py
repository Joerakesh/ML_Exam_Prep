# 1. Import libraries
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 2. Load dataset
data = pd.read_csv("../data/breast_cancer_wisconsin.csv")

# 3. Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# 4. Convert diagnosis into numbers
data["diagnosis"] = data["diagnosis"].map({
    "B": 0,
    "M": 1
})

# 5. Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 6. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# 8. Check explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# 9. Check new shape
print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)