import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("../data/heart_disease_dataset.csv")

# Separate features
X = df.drop("Heart Disease", axis=1)

# Identify columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# Convert sparse matrix to array
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Variance Explained:")
print(total_variance)

print("\nPercentage of Variance Explained:")
print(total_variance * 100, "%")

# Visualization
plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=40
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.grid(True)

plt.show()