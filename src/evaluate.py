import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
     roc_curve
)

import matplotlib.pyplot as plt



# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")


# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# Remove customer ID
df = df.drop("customerID", axis=1)


# Separate features and target
X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load preprocessor and model
preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

model = joblib.load(
    "models/logistic_regression.pkl"
)

tuned_random_forest = joblib.load(
    "models/tuned_random_forest.pkl"
)


# Transform test data
X_test_processed = preprocessor.transform(X_test)


# Make predictions
y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# Random Forest predictions
tuned_rf_pred = tuned_random_forest.predict(
    X_test_processed
)

tuned_rf_probability = tuned_random_forest.predict_proba(
    X_test_processed
)[:, 1]


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# Random Forest metrics
tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_roc_auc = roc_auc_score(
    y_test,
    tuned_rf_probability
)

print("\n===== TUNED RANDOM FOREST EVALUATION =====")
print(f"Accuracy:  {tuned_rf_accuracy:.4f}")
print(f"Precision: {tuned_rf_precision:.4f}")
print(f"Recall:    {tuned_rf_recall:.4f}")
print(f"F1 Score:  {tuned_rf_f1:.4f}")
print(f"ROC-AUC:   {tuned_rf_roc_auc:.4f}")


# Print results
print("\n===== MODEL EVALUATION =====")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\n===== TUNED RANDOM FOREST CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        tuned_rf_pred
    )
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)


tuned_rf_cm = confusion_matrix(
    y_test,
    tuned_rf_pred
)

print("\n===== TUNED RANDOM FOREST CONFUSION MATRIX =====")
print(tuned_rf_cm)



# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")

plt.tight_layout()
plt.show()




# Decision Threshold Analysis

print("\n===== TUNED RANDOM FOREST THRESHOLD ANALYSIS =====")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    y_pred_threshold = (
        tuned_rf_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        y_pred_threshold
    )

    threshold_recall = recall_score(
        y_test,
        y_pred_threshold
    )

    threshold_f1 = f1_score(
        y_test,
        y_pred_threshold
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {threshold_precision:.4f} | "
        f"Recall: {threshold_recall:.4f} | "
        f"F1: {threshold_f1:.4f}"
    )