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
print(f"Initial shape (rows: {df.shape[0]}, columns: {df.shape[1]})")
print()


# =========================
# Inspect data
# =========================
# Show column types and non-null counts
print("Dataset info:")
df.info()
print()

# Count missing values per column
print("Missing values per column:")
missing_counts = df.isna().sum()
print(missing_counts)
print()

# Print first rows to understand what the dataset looks like
print("First rows:")
print(df.head())
print()


# =========================
# Visual exploration
# =========================
# These scatter plots help visually inspect how each advertising channel relates to sales.
# Note: This is NOT the regression model plot. It’s raw-data exploration.

plt.scatter(df["TV"], df["sales"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("TV vs Sales")
plt.show()

plt.scatter(df["radio"], df["sales"])
plt.xlabel("Radio")
plt.ylabel("Sales")
plt.title("Radio vs Sales")
plt.show()

plt.scatter(df["newspaper"], df["sales"])
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.title("Newspaper vs Sales")
plt.show()


# =========================
# Define features (X) and target (y)
# =========================
# X = independent variables
# y = dependent variable

X = df[["TV", "radio", "newspaper"]]
y = df["sales"]


# =========================
# Train/Test Split
# =========================
# We split the dataset so we can evaluate how well the model generalizes to unseen data.
# test_size=0.2 means 20% of the data is used for testing.
# random_state ensures reproducibility (same split each run).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=430
)


# =========================
# Train the model
# =========================
# LinearRegression learns coefficients (slopes) for each feature + an intercept term.

model = LinearRegression()
model.fit(X_train, y_train)


# =========================
# Predict and evaluate
# =========================
# Predict Sales on the test set (unseen data)
y_pred = model.predict(X_test)

# R² measures how much variance in y is explained by the model (on the test set here)
r2 = r2_score(y_test, y_pred)

print(f"R² (test set): {r2:.4f}")

# Show learned parameters
# Coefficients correspond to [TV, radio, newspaper] in the same order as X.columns
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)



# =========================
# Model visualization (test set)
# =========================

# ---- 1) Actual vs Predicted ----
# If predictions are perfect, points lie on the diagonal line.
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Test Set)")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

# Diagonal (perfect prediction) reference line
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", label="Perfect prediction")

plt.legend()
plt.show()


# ---- 2) Residuals vs Predicted ----
# Residuals = Actual - Predicted
# Ideally residuals are randomly scattered around 0 with no clear pattern.
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Test Set)")
plt.show()