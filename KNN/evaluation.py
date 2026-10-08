from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_test, y_pred):
    """Calculate and display classification metrics."""

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== MODEL PERFORMANCE =====")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    TN, FP, FN, TP = cm.ravel()

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== CONFUSION MATRIX DETAILS =====")
    print(f"True Negative (TN): {TN}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print(f"True Positive (TP): {TP}")

    # print("\n===== CONFUSION MATRIX =====")
    # print(confusion_matrix(y_test, y_pred))