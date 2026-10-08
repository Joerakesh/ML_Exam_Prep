import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("../data/heart_disease_dataset.csv")

# Separate features and target
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Identify columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Test different K values
k_values = [1, 3, 5, 7, 9, 11, 15, 21]

accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_processed, y_train)

    y_pred = knn.predict(X_test_processed)

    accuracy_values.append(accuracy_score(y_test, y_pred))
    precision_values.append(precision_score(y_test, y_pred, zero_division=0))
    recall_values.append(recall_score(y_test, y_pred, zero_division=0))
    f1_values.append(f1_score(y_test, y_pred, zero_division=0))

# Results
results = pd.DataFrame({
    "K": k_values,
    "Accuracy": accuracy_values,
    "Precision": precision_values,
    "Recall": recall_values,
    "F1 Score": f1_values
})

print(results)

# Plot K vs Accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_values, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Effect of K Value on KNN Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# Select best K based on F1 Score
best_index = np.argmax(f1_values)
best_k = k_values[best_index]

print("\nBest K:", best_k)

# Final KNN model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_processed, y_train)

final_pred = final_knn.predict(X_test_processed)

# Evaluation
final_accuracy = accuracy_score(y_test, final_pred)
final_precision = precision_score(y_test, final_pred, zero_division=0)
final_recall = recall_score(y_test, final_pred, zero_division=0)
final_f1 = f1_score(y_test, final_pred, zero_division=0)

print("\nAccuracy :", final_accuracy)
print("Precision:", final_precision)
print("Recall   :", final_recall)
print("F1 Score :", final_f1)

print("\nClassification Report:")
print(classification_report(y_test, final_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
print("\nConfusion Matrix:")
print(cm)

# Specificity
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)

print("Specificity:", specificity)

# AUC
probability = final_knn.predict_proba(X_test_processed)[:, 1]
auc = roc_auc_score(y_test, probability)

print("AUC:", auc)