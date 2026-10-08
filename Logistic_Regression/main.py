# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load dataset
data = pd.read_csv("../data/breast_cancer_wisconsin.csv")

# 3. Remove unnecessary columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# 4. Convert diagnosis into numbers
# M = 1 (Malignant), B = 0 (Benign)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# 5. Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# 6. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Create Logistic Regression model
model = LogisticRegression()

# 9. Train the model
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 12. Predict a new sample
new_data = X.iloc[[0]]
new_data = scaler.transform(new_data)

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")