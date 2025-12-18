# =========================
# Multiple Linear Regression (Advertising) + Train/Test Split + Visualization
# =========================
# What this script does:
# 1) Loads the Advertising dataset
# 2) Performs quick inspection + raw-data scatter plots
# 3) Splits the data into train/test sets
# 4) Trains a multiple linear regression model
# 5) Evaluates performance using R² on the test set
# 6) Visualizes predictions and residuals on the test set

# =========================
# Import libraries
# =========================
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# =========================
# File configuration
# =========================
# Paths are constructed relative to a datasets folder inside the repository.
# (If you run this from a different working directory, you may need the __file__-based approach.)

base_path = "../../../datasets"
raw_path = os.path.join(base_path, "raw", "Topic2")
processed_path = os.path.join(base_path, "processed", "Topic2")  # not used here, kept for consistency

file_name = "Advertising"
file_ext = ".csv"

raw_file = os.path.join(raw_path, file_name + file_ext)
clean_file = os.path.join(processed_path, file_name + "_clean" + file_ext)  # not used here


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
# The goal here is to confirm:
# - column names and types
# - whether there are missing values
# - whether the dataset looks as expected

print("Dataset info:")
df.info()
print()

print("Missing values per column:")
missing_counts = df.isna().sum()
print(missing_counts, "\n")

print("First rows:")
print(df.head(), "\n")


# =========================
# Visual exploration (raw data)
# =========================
# These scatter plots show feature → target relationships.
# This is NOT a model plot; it is exploratory visualization.

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
# X contains the independent variables (predictors).
# y contains the dependent variable (the target we want to predict).

X = df[["TV", "radio", "newspaper"]]
y = df["sales"]


# =========================
# Train/Test Split (validation)
# =========================
# We split the data to estimate generalization performance.
# - The model trains on X_train/y_train
# - The model is evaluated on X_test/y_test (unseen during training)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,     # 20% for testing
    random_state=430   # reproducible split
)


# =========================
# Train the model
# =========================
# LinearRegression learns:
# - one coefficient per feature (TV, radio, newspaper)
# - one intercept term (baseline)

model = LinearRegression()
model.fit(X_train, y_train)


# =========================
# Predict and evaluate (test set)
# =========================
# Predictions are generated only for the test set.
# R² is computed on the test set to measure how well the model generalizes.

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R² (test set): {r2:.4f}")

# Print learned parameters for interpretation:
# - coefficients show the direction and strength of each channel (holding others constant)
# - intercept is the baseline prediction when all features are zero
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)


# =========================
# Model visualization (test set)
# =========================

# ---- 1) Actual vs Predicted (test set) ----
# If predictions are perfect, points lie on the diagonal reference line.
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Test Set)")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    label="Perfect prediction"
)

plt.legend()
plt.show()


# ---- 2) Residuals vs Predicted (test set) ----
# Residuals = Actual - Predicted.
# Ideally: residuals are scattered randomly around 0 with no obvious pattern.

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Test Set)")
plt.show()
