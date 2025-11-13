import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import mlflow
# import mlflow.sklearn

# -----------------------------
# 1. Setup and configuration
# -----------------------------
DATA_PATH = "./Data/AnalyzedData/SeoulBikeData_Analyzed.csv"
MODEL_DIR = "."
PLOT_DIR = "./Plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# 2. Load and prepare data
# -----------------------------
data = pd.read_csv(DATA_PATH)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

X = data.drop(columns=["Rented_Bike_Count", "Date"])
y = data["Rented_Bike_Count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# -----------------------------
# 3. Define and train model
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# with mlflow.start_run():
rf_model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate model
# -----------------------------
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Random Forest training complete.")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")

# Optional MLflow logging
# mlflow.log_params({
#     "model_type": "RandomForestRegressor",
#     "n_estimators": 100,
#     "max_depth": 10
# })
# mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

# -----------------------------
# 5. Feature importance plot
# -----------------------------
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, feature_importances)
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "rf_feature_importance.png")
plt.savefig(plot_path)
plt.close()

print(f"📊 Feature importance plot saved to {PLOT_DIR}")

# Optional MLflow logging
# mlflow.log_artifact(plot_path)
# mlflow.sklearn.log_model(rf_model, "rf_model")

# -----------------------------
# 6. Save trained model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
joblib.dump(rf_model, model_path)
print(f"💾 Model saved to {model_path}")
