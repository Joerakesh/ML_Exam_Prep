# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

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

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create model
model = RandomForestClassifier(random_state=42)

# 8. Define parameters
parameters = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

# 9. Hyperparameter tuning
grid = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring="accuracy"
)

# 10. Train and find best parameters
grid.fit(X_train, y_train)

# 11. Print best parameters
print("Best Parameters:", grid.best_params_)

# 12. Predict using best model
y_pred = grid.predict(X_test)

# 13. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))