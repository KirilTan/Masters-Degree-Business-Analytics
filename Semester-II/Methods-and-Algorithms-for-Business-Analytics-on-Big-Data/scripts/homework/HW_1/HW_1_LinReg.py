# =========================
# HW1 – Linear Regression (Feature Search + Train/Test + Visualization)
# =========================
# What this script does:
# 1) Loads the homework dataset
# 2) Cleans it (rename X3, drop Time, reorder)
# 3) Saves a cleaned copy
# 4) Visualizes X1/X2/X3 vs Y
# 5) Tries all feature combinations and evaluates with a train/test split (R²)
# 6) Selects the best model and visualizes predictions and residuals
# 7) (Optional) Adds K-Fold CV evaluation for the best feature set

# %%
# =========================
# Import libraries
# =========================
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# %%
# =========================
# File configuration
# =========================
base_path = "../../../datasets"
raw_path = os.path.join(base_path, "raw", "HW_1")
processed_path = os.path.join(base_path, "processed", "HW_1")

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
print(f"Initial shape: {df.shape}\n")

print("Columns:", df.columns.tolist(), "\n")

# %%
# =========================
# Inspect data
# =========================
print("Dataset info:")
df.info()
print()

print("Missing values per column:")
print(df.isna().sum(), "\n")

print("First rows:")
print(df.head(), "\n")

# %%
# =========================
# Clean data
# =========================
# 1) Standardize column names slightly (strip spaces)
df.columns = [c.strip() for c in df.columns]

# 2) Rename the last column to X3 (often exported as "Unnamed: 4")
df = df.rename(columns={df.columns[-1]: "X3"})

# 3) Drop Time column (case-insensitive)
time_cols = [c for c in df.columns if c.lower() == "time"]
df = df.drop(columns=time_cols, errors="ignore")

# 4) Reorder columns (make sure these exist)
df = df[["X1", "X2", "X3", "Y"]]

# 5) Save cleaned dataset (ensure folder exists)
os.makedirs(processed_path, exist_ok=True)
df.to_csv(clean_file, index=False)

print(f"Saved cleaned dataset to: {clean_file}")
print(f"Final shape: {df.shape}\n")

# %%
# =========================
# Visual exploration (raw data)
# =========================
# These scatter plots show how each feature relates to Y.
# Useful for intuition, outliers, and whether a linear relationship is plausible.

plt.scatter(df["X1"], df["Y"], alpha=0.5)
plt.xlabel("X1")
plt.ylabel("Y")
plt.title("X1 vs Y")
plt.show()

plt.scatter(df["X2"], df["Y"], alpha=0.5)
plt.xlabel("X2")
plt.ylabel("Y")
plt.title("X2 vs Y")
plt.show()

plt.scatter(df["X3"], df["Y"], alpha=0.5)
plt.xlabel("X3")
plt.ylabel("Y")
plt.title("X3 vs Y")
plt.show()

# %%
# =========================
# Define candidate features
# =========================
features = ["X1", "X2", "X3"]
target = "Y"

# %%
# =========================
# Train/Test evaluation for all feature combinations
# =========================
# We evaluate 1-feature, 2-feature, and 3-feature models and pick the best test R².
# random_state ensures reproducibility.

test_size = 0.2
random_state = 430

results = []

for r in range(1, len(features) + 1):
    for combo in itertools.combinations(features, r):
        X = df[list(combo)]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "features": combo,
            "r2_test": r2
        })

# Show ranked results
results_df = pd.DataFrame(results).sort_values("r2_test", ascending=False).reset_index(drop=True)

print("=== Feature combinations ranked by Test R² ===")
print(results_df.head(10))
print()

best_features = list(results_df.loc[0, "features"])
best_r2 = results_df.loc[0, "r2_test"]

print(f"Best feature set: {best_features}")
print(f"Best Test R²: {best_r2:.4f}\n")

# %%
# =========================
# Outlier removal (TRAIN ONLY, based on residuals)
# =========================
# Train once, find the largest residuals on TRAIN, drop them, refit, re-evaluate on same test set.

# Split (same as before)
X_best = df[best_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_best, y, test_size=test_size, random_state=random_state
)

# Fit initial model
model = LinearRegression()
model.fit(X_train, y_train)

# Residuals on TRAIN only
train_pred = model.predict(X_train)
train_residuals = y_train - train_pred

# Define outliers as residuals beyond k standard deviations
k = 3.0
threshold = k * train_residuals.std()

outlier_mask = train_residuals.abs() > threshold
num_outliers = outlier_mask.sum()

print(f"Train residual std: {train_residuals.std():.2f}")
print(f"Outlier threshold (k={k}): {threshold:.2f}")
print(f"Outliers detected in TRAIN: {num_outliers}\n")

# Drop outliers from TRAIN only
X_train_clean = X_train.loc[~outlier_mask]
y_train_clean = y_train.loc[~outlier_mask]

# Refit on cleaned training data
model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train_clean)

# Evaluate on the ORIGINAL test set (untouched)
y_pred_clean = model_clean.predict(X_test)
r2_clean = r2_score(y_test, y_pred_clean)

print(f"R² (test set) BEFORE cleaning: {r2_score(y_test, model.predict(X_test)):.4f}")
print(f"R² (test set) AFTER  cleaning: {r2_clean:.4f}\n")

print("Coefficients (cleaned):", dict(zip(best_features, model_clean.coef_)))
print("Intercept (cleaned):", model_clean.intercept_)
print()


# %%
# =========================
# Train best model (on the same train/test split) + interpret parameters
# =========================
X_best = df[best_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_best, y, test_size=test_size, random_state=random_state
)

best_model = LinearRegression()
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
r2_final = r2_score(y_test, y_pred)

print("=== Best model parameters ===")
print(f"R² (test set): {r2_final:.4f}")
print("Coefficients:", dict(zip(best_features, best_model.coef_)))
print("Intercept:", best_model.intercept_)
print()

# %%
# =========================
# Visualization (best model, test set)
# =========================

# ---- 1) Actual vs Predicted ----
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title(f"Actual vs Predicted (Test Set) – Features: {best_features}")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         color="red", linestyle="--", label="Perfect prediction")

plt.legend()
plt.show()

# ---- 2) Residuals vs Predicted ----
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Y")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Test Set)")
plt.show()

# %%
# =========================
# (Optional) K-Fold CV for the best feature set
# =========================
# This gives a more stable estimate of performance than one train/test split.

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
cv_scores = []

for train_idx, test_idx in kf.split(X_best):
    X_tr, X_te = X_best.iloc[train_idx], X_best.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    m = LinearRegression()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    cv_scores.append(r2_score(y_te, pred))

print("=== K-Fold CV (best feature set) ===")
print(f"Mean R²: {np.mean(cv_scores):.4f}")
print(f"Std R²:  {np.std(cv_scores, ddof=1):.4f}")

