import pandas as pd
import joblib

# Load final production pipeline
final_pipeline = joblib.load(
    "models/final_pipeline.pkl"
)

# Create one sample customer
customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": 960.0
}])

# Predict
prediction = final_pipeline.predict(customer)

# Churn probability
probability = final_pipeline.predict_proba(customer)[0][1]

print("\n===== CUSTOMER CHURN PREDICTION =====")

if prediction[0] == 1:
    print("Prediction: Customer is likely to churn")
else:
    print("Prediction: Customer is likely to stay")

print(f"Churn Probability: {probability:.2%}")


# Determine churn risk level

if probability < 0.30:
    risk_level = "Low Risk"
elif probability < 0.60:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"

print(f"Risk Level: {risk_level}")





# Retention recommendation

if risk_level == "High Risk":

    if customer["Contract"].iloc[0] == "Month-to-month":
        recommendation = "Offer a discounted long-term contract."

    elif customer["TechSupport"].iloc[0] == "No":
        recommendation = "Offer free or discounted technical support."

    elif customer["PaymentMethod"].iloc[0] == "Electronic check":
        recommendation = "Encourage automatic payment with an incentive."

    else:
        recommendation = "Contact customer with a personalized retention offer."

elif risk_level == "Medium Risk":
    recommendation = "Monitor customer and provide targeted engagement."

else:
    recommendation = "No immediate retention action required."

print(f"Retention Recommendation: {recommendation}")