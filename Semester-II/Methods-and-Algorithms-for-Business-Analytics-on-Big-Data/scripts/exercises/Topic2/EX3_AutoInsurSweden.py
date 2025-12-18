# %%
# =========================
# Import libraries
# =========================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score


# %%
# =========================
# File configuration
# =========================
# Paths are constructed relative to a datasets folder inside the repository.

base_path = "../../../datasets"
raw_path = os.path.join(base_path, "raw", "Topic2")
processed_path = os.path.join(base_path, "processed", "Topic2")  # not used here

file_name = "AutoInsurSweden"
file_ext = ".Txt"

raw_file = os.path.join(raw_path, file_name + file_ext)


# %%
# =========================
# Load data
# =========================

print("Loading dataset...")

df = pd.read_csv(
    raw_file,
    sep=r"\s+",     # split on whitespace
    decimal=","     # decimal separator
)

print("Dataset loaded successfully.")
print(f"Initial shape: {df.shape}\n")


# %%
# =========================
# Inspect data
# =========================

print("Dataset info:")
df.info()
print()

print("Missing values per column:")
print(df.isna().sum())
print()


# %%
# =========================
# Visual exploration (raw data)
# =========================

plt.scatter(df["X"], df["Y"])
plt.xlabel("Feature (X)")
plt.ylabel("Target (Y)")
plt.title("Feature vs Target (raw data)")
plt.show()


# %%
# =========================
# Define features and target
# =========================

X = df[["X"]]   # must be 2D
y = df["Y"]


# %%
# =========================
# Train/Test Split (validation)
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=430
)


# %%
# =========================
# Train the model
# =========================

model = LinearRegression()
model.fit(X_train, y_train)


# %%
# =========================
# Predict and evaluate (test set)
# =========================

y_pred_test = model.predict(X_test)

r2_test = r2_score(y_test, y_pred_test)
print(f"R² (test set): {r2_test:.4f}\n")


# %%
# =========================
# Model interpretation
# =========================

coef = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Model:")
print(f"Y = {intercept:.4f} + {coef:.4f} * X\n")

print("Interpretation:")
print(f"- Intercept: expected Y when X = 0 → {intercept:.4f}")
print(f"- Coefficient: one-unit increase in X increases Y by {coef:.4f} on average\n")


# %%
# =========================
# Visualization: regression line
# =========================

plt.scatter(X, y, label="Observed data")

X_line = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.plot(X_line, y_line, color="red", label="Regression line")
plt.xlabel("Feature (X)")
plt.ylabel("Target (Y)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()


# %%
# =========================
# Residual analysis (test set)
# =========================

residuals = y_test - y_pred_test

plt.scatter(y_pred_test, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (test set)")
plt.show()

print("Residuals summary (test set):")
print(residuals.describe())
print()


# %%
# =========================
# K-Fold Cross-Validation (manual)
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=430
)

fold_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    model = LinearRegression()
    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_fold)
    r2_fold = r2_score(y_test_fold, y_pred_fold)

    fold_scores.append(r2_fold)
    print(f"Fold {fold}: R² = {r2_fold:.4f}")

print(f"\nK-Fold CV R² mean: {np.mean(fold_scores):.4f}")
print(f"K-Fold CV R² std:  {np.std(fold_scores, ddof=1):.4f}\n")


# %%
# =========================
# Cross-validated predictions (cross_val_predict)
# =========================
# Each observation is predicted by a model that did NOT see it during training.

cv_predictions = cross_val_predict(
    LinearRegression(),
    X,
    y,
    cv=kf
)

cv_r2 = r2_score(y, cv_predictions)
print(f"Cross-validated R² (cross_val_predict): {cv_r2:.4f}\n")


# %%
# =========================
# Visualization: CV predictions vs true values
# =========================

plt.scatter(y, cv_predictions)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--",
    color="red"
)
plt.xlabel("True values")
plt.ylabel("Cross-validated predictions")
plt.title("CV Predictions vs True Values")
plt.show()


# %%
# =========================
# Final conclusion
# =========================

print("Conclusion:")
print("- A simple linear regression model was fitted to the dataset.")
print("- The test-set R² indicates strong generalization performance.")
print("- K-Fold cross-validation confirms stable performance across folds.")
print("- cross_val_predict provides an unbiased estimate for all observations.")
print("- Residual analysis does not indicate major violations of linear assumptions.")
