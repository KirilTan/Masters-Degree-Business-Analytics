# %%
# =========================
# Import libraries
# =========================

import pandas as pd
import numpy as np
from pathlib import Path

# %%
# =========================
# Load data
# =========================

BASE_DIR = Path.cwd()

csv_path = (
    BASE_DIR
    / "Semester-II"
    / "Methods-and-Algorithms-for-Business-Analytics-on-Big-Data"
    / "datasets"
    / "raw"
    / "HW_1"
    / "Data.csv"
)

df = pd.read_csv(csv_path)

"""
# Spyder
df = pd.read_csv(
    "Semester-II/Methods-and-Algorithms-for-Business-Analytics-on-Big-Data/datasets/raw/HW_1/Data.csv"
)
"""

# %%
# =========================
# Inspect data
# =========================

print(f"Initial shape: {df.shape}\n")

print("Dataset info:")
df.info()
print()


print("Missing values per column:")
print(df.isna().sum())
print()

# %%
# 1) Rename last column to X3
df = df.rename(columns={df.columns[-1]: "X3"})

# 2) Drop 'time' column (safe even if it's missing)
df = df.drop(columns=["time"], errors="ignore")

# 3) Reorder columns
df = df[["X1", "X2", "X3", "Y"]]

# 4) Save cleaned CSV (ensure folder exists)
out_dir = Path(r"F:\PyCharm Projects\Masters-Degree-Business-Analytics\Semester-II\Methods-and-Algorithms-for-Business-Analytics-on-Big-Data\datasets\processed\HW_1")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "Data_clean.csv"
df.to_csv(out_path, index=False)

print(f"Saved cleaned dataset to: {out_path}")
print("Final columns:", list(df.columns))
print("Final shape:", df.shape)
