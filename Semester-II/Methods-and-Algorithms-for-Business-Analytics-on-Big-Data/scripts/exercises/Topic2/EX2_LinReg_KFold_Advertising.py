# =========================
# Multiple Linear Regression (Advertising) + K-Fold CV + Visualization
# =========================
# What this script does:
# 1) Loads the Advertising dataset
# 2) Performs basic inspection + simple raw-data scatter plots
# 3) Evaluates Linear Regression using K-Fold Cross-Validation (R²)
# 4) Creates honest plots using Out-Of-Fold (OOF) predictions
# 5) Trains a final model on the full dataset and prints coefficients


# =========================
# Import libraries
# =========================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score


# =========================
# File configuration
# =========================
# Base datasets directory (relative path from this script's location / working directory)
base_path = "../../../datasets"

# Where the raw dataset is stored
raw_path = os.path.join(base_path, "raw", "Topic2")

# Where processed/cleaned outputs would be stored
processed_path = os.path.join(base_path, "processed", "Topic2")

# Dataset name and extension
file_name = "Advertising"
file_ext = ".csv"

# Full file paths
raw_file = os.path.join(raw_path, file_name + file_ext)
clean_file = os.path.join(processed_path, file_name + "_clean" + file_ext)



# =========================
# Load data
# =========================
print("Loading dataset...")
df = pd.read_csv(raw_file)

print("Dataset loaded successfully.")
print(f"Initial shape (rows: {df.shape[0]}, columns: {df.shape[1]})\n")


# =========================
# Inspect data
# =========================
print("Columns:", df.columns.tolist(), "\n")

print("Dataset info:")
df.info()
print()

print("Missing values per column:")
missing_counts = df.isna().sum()
print(missing_counts, "\n")

print("First rows:")
print(df.head(), "\n")


# =========================
# OPTIONAL cleanup: remove accidental index column
# =========================
# Some CSV exports include an extra index column like "Unnamed: 0".
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("Dropped column: Unnamed: 0")
    print("New columns:", df.columns.tolist(), "\n")


# =========================
# Visual exploration (raw data)
# =========================
# These plots show how each feature relates to the target.
# This is NOT the model output plot. It helps you understand the data relationships.

plt.scatter(df["TV"], df["sales"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("TV vs Sales (raw data)")
plt.show()

plt.scatter(df["radio"], df["sales"])
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.title("Radio vs Sales (raw data)")
plt.show()

plt.scatter(df["newspaper"], df["sales"])
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.title("Newspaper vs Sales (raw data)")
plt.show()


# =========================
# Define features (X) and target (y)
# =========================
# X = independent variables (predictors/features)
# y = dependent variable (target)

X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

print("X shape:", X.shape)
print("y shape:", y.shape, "\n")


# =========================
# K-Fold Cross-Validation (evaluation)
# =========================
# K-Fold CV evaluates the model multiple times on different splits.
# Each split trains on K-1 folds and tests on the remaining fold.
# We report the R² for each fold + the mean and std across folds.

kf = KFold(
    n_splits=5,        # number of folds (K)
    shuffle=True,      # shuffle before splitting
    random_state=430   # reproducible splits
)

fold_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    # Split data by indices for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train a new model on the training fold
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test fold
    y_pred = model.predict(X_test)

    # Evaluate with R²
    r2 = r2_score(y_test, y_pred)
    fold_scores.append(r2)

    print(f"Fold {fold}: R² = {r2:.4f}")

mean_r2 = float(np.mean(fold_scores))
std_r2 = float(np.std(fold_scores, ddof=1))  # sample std
print(f"\nK-Fold CV R² mean: {mean_r2:.4f}")
print(f"K-Fold CV R² std:  {std_r2:.4f}\n")


# =========================
# Out-Of-Fold (OOF) predictions for honest visualization
# =========================
# OOF prediction: each observation is predicted by a model that did NOT train on that observation.
# This gives you a fair "actual vs predicted" visualization consistent with cross-validation.

y_oof = cross_val_predict(
    LinearRegression(),
    X,
    y,
    cv=kf
)

oof_r2 = r2_score(y, y_oof)
print(f"OOF R² (overall): {oof_r2:.4f}\n")


# =========================
# Visualization 1: Actual vs Predicted (OOF)
# =========================
plt.scatter(y, y_oof, color="blue", label="OOF predictions")

min_val = min(y.min(), y_oof.min())
max_val = max(y.max(), y_oof.max())

# Perfect prediction diagonal
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", label="Perfect prediction")

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales (OOF)")
plt.title("Actual vs Predicted Sales (Out-of-Fold, K-Fold CV)")
plt.legend()
plt.show()


# =========================
# Visualization 2: Residuals vs Predicted (OOF)
# =========================
# Residuals = Actual - Predicted
# Ideally: residuals are randomly scattered around 0 (no clear pattern).

residuals_oof = y - y_oof

plt.scatter(y_oof, residuals_oof)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Sales (OOF)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Out-of-Fold, K-Fold CV)")
plt.show()


# =========================
# Train final model on ALL data (recommended final step)
# =========================
# After evaluation, train a final model using all available data.
# This is typical if you want a final model for reporting or later use.

final_model = LinearRegression()
final_model.fit(X, y)

print("Final model trained on FULL dataset:")
print("Coefficients:", dict(zip(X.columns, final_model.coef_)))
print("Intercept:", final_model.intercept_)