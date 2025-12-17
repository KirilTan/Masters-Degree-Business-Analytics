# =========================
# Import libraries
# =========================
import pandas as pd
import os


# =========================
# File configuration
# =========================
base_path = "../../../datasets"
raw_path = os.path.join(base_path, "raw", "Topic1")
processed_path = os.path.join(base_path, "processed", "Topic1")

file_name = "insurance"
file_ext = ".csv"

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
print("Dataset info:")
df.info()
print()

print("Missing values per column:")
missing_counts = df.isna().sum()
print(missing_counts)
print()


# =========================
# Clean data (drop NaN rows)
# =========================
total_missing = missing_counts.sum()

if total_missing > 0:
    print(f"Missing values detected: {total_missing}")
    print("Dropping rows with missing values...")
    df_clean = df.dropna()
else:
    print("No missing values detected.")
    print("No rows will be dropped.")
    df_clean = df

print("Cleaned dataset shape:", df_clean.shape)
print()


# =========================
# Save cleaned data
# =========================
print("Saving cleaned dataset...")
df_clean.to_csv(clean_file, index=False)
print(f"Cleaned dataset saved to:\n{clean_file}")

# =========================
# Optional cleanup
# =========================
# Uncomment the line below to delete the created file
# (useful when re-running exercises)
#
# os.remove(clean_file)
