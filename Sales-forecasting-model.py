import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("data/business_kpi_sample_data.csv")

# Step 2: Feature engineering
# Add Profit and CAC as predictors
df["Profit"] = df["Revenue"] - df["Expenses"]
df["CAC"] = df["Expenses"] / df["Customers"]

# Select features and target
X = df[["Revenue", "Expenses", "Customers", "Profit", "CAC"]]
y = df["Revenue"]  # Forecasting revenue as target

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Step 7: Visualization
plt.figure(figsize=(8,6))
plt.plot(y_test.values, label="Actual", marker="o")
plt.plot(y_pred, label="Predicted", marker="x")
plt.title("Actual vs Predicted Sales Revenue")
plt.xlabel("Test Sample")
plt.ylabel("Revenue")
plt.legend()
plt.show()
