import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




df = pd.read_csv("data/Telco-Customer-Churn.csv")

print(df.head())

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True)*100)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nTotalCharges problematic values:")

problematic_values = df[
    pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
]["TotalCharges"]

print(problematic_values)
print("Number of problematic values:", len(problematic_values))

print("\nRows with problematic TotalCharges:")

problematic_rows = df[
    pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
]


print(
    problematic_rows[
        ["customerID","tenure","MonthlyCharges","TotalCharges","Churn"]
    ]
)

# totalcharges ko numeric mein convert karna
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# missing totalcharges ko 0 se fill krna

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nTotalCharges data type after cleaning:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges after cleaning:")
print(df["TotalCharges"].isnull().sum())


print("\nFinal Dataset Shape:")
print(df.shape)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nFinal Data Types:")
print(df.dtypes)


df = df.drop("customerID", axis=1)

print("\nShape after removing customerID:")
print(df.shape)


print("\nRemaining columns:")
print(df.columns.tolist())


x = df.drop("Churn",axis=1)
y = df["Churn"]

print("\nFeatures (x) shape:")
print(x.shape)

print("\nTarget (y) shape:")
print(y.shape)

print("\nTarget values:")
print(y.unique())

y = y.map({
    "No": 0,
    "Yes": 1
})

print("\nTarget after encoding:")
print(y.value_counts())

print("\nTarget data type:")
print(y.dtype)


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


print("\nNumber of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))


numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessor created successfully!")








from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(x_train.shape)

print("\nTesting data shape:")
print(x_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))



preprocessor.fit(x_train)

print("\nPreprocessor fitted successfully!")


x_train_processed = preprocessor.transform(x_train)
x_test_processed = preprocessor.transform(x_test)

print("\nProcessed training shape:")
print(x_train_processed.shape)

print("\nProcessed testing shape:")
print(x_test_processed.shape)





print("\nFinal training data shape:")
print(x_train_processed.shape)

print("\nFinal testing data shape:")
print(x_test_processed.shape)



import joblib


joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nPreprocessor saved successfully!")