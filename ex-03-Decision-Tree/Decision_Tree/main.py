# 1. Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Load dataset
data = pd.read_csv("../data/titanic.csv")

# 3. Remove unnecessary columns
data = data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)

# 4. Handle missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# 5. Encode categorical columns
data["Sex"] = data["Sex"].map({
    "male": 0,
    "female": 1
})

data["Embarked"] = data["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# 6. Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Create Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# 9. Train
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
