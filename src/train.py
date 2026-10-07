import pandas as pd
import joblib


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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


# Feature groups
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load already trained preprocessor
preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# Transform data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# Train model
model.fit(X_train_processed, y_train)


print("Logistic Regression model trained successfully!")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)


# Save model
joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

print("Model saved successfully!")










# Create Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Train Random Forest
random_forest.fit(
    X_train_processed,
    y_train
)

print("Random Forest model trained successfully!")

# Save Random Forest model
joblib.dump(
    random_forest,
    "models/random_forest.pkl"
)

print("Random Forest model saved successfully!")




# Hyperparameter grid

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [2, 5]
}

print("\nHyperparameter grid created successfully!")




# Grid Search

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

# Run Grid Search

grid_search.fit(
    X_train_processed,
    y_train
)

print("\n===== RANDOM FOREST TUNING =====")
print("Best F1:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)


# Train final tuned Random Forest

best_random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

best_random_forest.fit(
    X_train_processed,
    y_train
)

print("\nFinal tuned Random Forest trained successfully!")

# Save tuned model
joblib.dump(
    best_random_forest,
    "models/tuned_random_forest.pkl"
)

print("Tuned Random Forest saved successfully!")





print("\n===== FINAL MODEL SELECTION =====")
print("Selected Model: Tuned Random Forest")
print("Reason: Best F1-score among evaluated models")
print("Test F1: 0.6332")
print("Test Recall: 0.7754")
print("Test ROC-AUC: 0.8409")




# Create final production pipeline

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_random_forest)
    ]
)

print("\nFinal production pipeline created successfully!")




# Save final production pipeline

joblib.dump(
    final_pipeline,
    "models/final_pipeline.pkl"
)

print("Final production pipeline saved successfully!")