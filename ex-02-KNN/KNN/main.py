# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 2. Load dataset
data = pd.read_csv("../data/breast_cancer_wisconsin.csv")

# 3. Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# 4. Encode target
# B = 0, M = 1
encoder = LabelEncoder()
data["diagnosis"] = encoder.fit_transform(data["diagnosis"])

# 5. Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# 9. Train model
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate model
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
