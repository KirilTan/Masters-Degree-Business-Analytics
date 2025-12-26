# =========================
# HW1 – Polynomial Features + Linear Regression (Train/Test + Visualization)
# =========================
# What this script does:
# 1) Loads the homework dataset (Data.csv)
# 2) Cleans it (rename X3, drop Time, reorder)
# 3) Saves a cleaned copy (Data_clean.csv)
# 4) Splits into train/test sets
# 5) Compares baseline linear regression vs polynomial feature expansion
# 6) Picks the best polynomial degree by test R²
# 7) Visualizes predictions and residuals for the best model

# %%
# =========================
# Import libraries
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# %%
# =========================
# File configuration (robust paths)
# =========================
# Build paths relative to THIS script file to avoid FileNotFoundError
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DATASETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../datasets"))

raw_path = os.path.join(DATASETS_DIR, "raw", "HW_1")
processed_path = os.path.join(DATASETS_DIR, "processed", "HW_1")

file_name = "Data"
file_ext = ".csv"

raw_file = os.path.join(raw_path, file_name + file_ext)
clean_file = os.path.join(processed_path, file_name + "_clean" + file_ext)

# %%
# =========================
# Load data
# =========================
print("Loading dataset...")
df = pd.read_csv(raw_file)

print("Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")
print("Columns:", df.columns.tolist(), "\n")

# %%
# =========================
# Clean data
# =========================
# 1) Strip spaces from column names (safety)
df.columns = [c.strip() for c in df.columns]

# 2) Rename last column to X3
df = df.rename(columns={df.columns[-1]: "X3"})

# 3) Drop Time column
time_cols = [c for c in df.columns if c.lower() == "time"]
df = df.drop(columns=time_cols, errors="ignore")

# 4) Reorder into a consistent modeling order
df = df[["X1", "X2", "X3", "Y"]]

# 5) Save cleaned dataset
os.makedirs(processed_path, exist_ok=True)
df.to_csv(clean_file, index=False)

print(f"Saved cleaned dataset to: {clean_file}")
print(f"Final shape: {df.shape}\n")

# %%
# =========================
# Define features (X) and target (y)
# =========================
features = ["X1", "X2", "X3"]
target = "Y"

X = df[features]
y = df[target]

# %%
# =========================
# Train/Test Split
# =========================
# We use a fixed split for reproducibility.
test_size = 0.2
random_state = 430

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# %%
# =========================
# Baseline Linear Regression (degree=1)
# =========================
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
r2_baseline = r2_score(y_test, y_pred_baseline)

print("=== Baseline Linear Regression ===")
print(f"Features: {features}")
print(f"R² (test): {r2_baseline:.4f}")
print("Coefficients:", dict(zip(features, baseline_model.coef_)))
print("Intercept:", baseline_model.intercept_)
print()

# %%
# =========================
# Polynomial Features: try multiple degrees
# =========================
# PolynomialFeatures creates:
# - squared terms (X1^2, X2^2, ...)
# - interaction terms (X1*X2, X1*X3, X2*X3, ...)
# Then LinearRegression is fitted on those expanded features.

degrees_to_try = [x for x in range(1,11)]
poly_results = []

for d in degrees_to_try:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("linreg", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    poly_results.append((d, r2, model))
    print(f"=== Polynomial Regression (degree={d}) ===")
    print(f"R² (test): {r2:.4f}\n")

# %%
# =========================
# Pick the best degree (including baseline)
# =========================
best_degree = 1
best_model = baseline_model
best_pred = y_pred_baseline
best_r2 = r2_baseline

for d, r2, model in poly_results:
    if r2 > best_r2:
        best_degree = d
        best_model = model
        best_pred = model.predict(X_test)
        best_r2 = r2

print("=== Best model selected ===")
print(f"Best degree: {best_degree}")
print(f"Best R² (test): {best_r2:.4f}\n")

# %%
# =========================
# Visualization (best model)
# =========================

# ---- 1) Actual vs Predicted ----
plt.scatter(y_test, best_pred, color="blue", label="Predicted vs Actual")
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title(f"Actual vs Predicted (Test Set) – Best degree: {best_degree}")

min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())

plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", label="Perfect prediction")

plt.legend()
plt.show()

# ---- 2) Residuals vs Predicted ----
residuals = y_test - best_pred

plt.scatter(best_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Y")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title(f"Residuals vs Predicted (Test Set) – Best degree: {best_degree}")
plt.show()
