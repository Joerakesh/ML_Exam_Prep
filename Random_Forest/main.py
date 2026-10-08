# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Load dataset
data = pd.read_csv("../data/breast_cancer_wisconsin.csv")

# 3. Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# 4. Convert diagnosis into numbers
# B = 0, M = 1
data["diagnosis"] = data["diagnosis"].map({
    "B": 0,
    "M": 1
})

# 5. Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 8. Train model
model.fit(X_train, y_train)

# 9. Predict
y_pred = model.predict(X_test)

# 10. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))