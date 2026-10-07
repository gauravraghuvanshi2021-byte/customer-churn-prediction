import pandas as pd

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Datatset Shape:")
print(df.shape)


print("\nFirst 5 Rows:")
print(df.head())


print("\nChurn Distribution:")
print(df["Churn"].value_counts())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nData Types:")
print(df.dtypes)



print("\nChurn Rate by Contract:")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)




print("\nAverage Tenure by Churn:")
print(
    df.groupby("Churn")["tenure"].mean()
)



print("\nAverage Monthly Charges by Churn:")

print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)



print("\nChurn Rate by Internet Service:")

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
)*100

print(internet_churn)



print("\nChurn Rate by Payment Method:")

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
)*100

print(payment_churn)




print("\nChurn Rate by Tech Support:")

techsupport_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(techsupport_churn)



print("\nChurn Rate by Online Security:")

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print(security_churn)








import matplotlib.pyplot as plt

# Churn rate by contract
contract_churn_rate = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\nContract Churn Rate:")
print(contract_churn_rate)

contract_churn_rate.plot(kind="bar")

plt.title("Churn Rate by Contract")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()







# Churn rate by Internet Service
internet_churn_rate = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\nInternet Service Churn Rate:")
print(internet_churn_rate)

internet_churn_rate.plot(kind="bar")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()








# Average Monthly Charges by Churn
monthly_charges = df.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage Monthly Charges by Churn:")
print(monthly_charges)

monthly_charges.plot(kind="bar")

plt.title("Average Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()