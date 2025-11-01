import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("uber_data.csv", dtype=str)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Convert "null"/empty strings to NaN
df = df.replace({"null": np.nan, "NULL": np.nan, "None": np.nan, "": np.nan})

# Strip extra quotes like """CID123"""
def strip_quotes(s):
    if isinstance(s, str):
        return s.strip().strip('"').strip("'")
    return s
df = df.applymap(strip_quotes)

# ---------------------------
# Convert Columns to Correct Types
# ---------------------------
numeric_cols_declared = [
    "Avg VTAT","Avg CTAT","Cancelled Rides by Customer",
    "Cancelled Rides by Driver","Incomplete Rides",
    "Booking Value","Ride Distance","Driver Ratings","Customer Rating"
]
for col in numeric_cols_declared:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Date/time
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
if "Time" in df.columns:
    # keep as object (time) but parse safely
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time

# ---------------------------
# Handle Missing Values
# ---------------------------
# Drop columns with >50% nulls (simple rule)
null_ratio = df.isna().mean()
drop_cols = null_ratio[null_ratio > 0.5].index.tolist()
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
print(f"🧹 Dropped columns with >50% nulls: {drop_cols if drop_cols else 'None'}")

# Fill numeric NaNs with mean
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical NaNs with mode, fallback to "none"
for col in df.select_dtypes(exclude=[np.number]).columns:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        df[col].fillna(mode.iloc[0] if not mode.empty else "none", inplace=True)

print("✅ Null values handled")

# ---------------------------
# Handle Outliers (IQR → replace with median)
# ---------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

def replace_outliers_with_median(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return series  # nothing to do
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    if mask.any():
        med = series.median()
        series.loc[mask] = med
    return series

for col in num_cols:
    df[col] = replace_outliers_with_median(df[col])

print("✅ Outliers handled (replaced with median)")

# ---------------------------
# Normalise / Standardise Data
# ---------------------------
scale_cols = [c for c in ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance","Driver Ratings","Customer Rating"] if c in df.columns]
if scale_cols:
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("✅ Data normalised")

# ---------------------------
# One-Hot Encode Categorical Columns
# ---------------------------
cat_cols = [c for c in ["Booking Status","Vehicle Type","Payment Method"] if c in df.columns]
if cat_cols:
    # Use sparse=False for broad compatibility
    enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    ohe = enc.fit_transform(df[cat_cols])
    ohe_df = pd.DataFrame(ohe, columns=enc.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), ohe_df], axis=1)
print("✅ One-hot encoding complete")

# ---------------------------
# Save Cleaned Data (ML-ready)
# ---------------------------
df.to_csv("uber_data_cleaned.csv", index=False)
print(f"💾 Cleaned dataset saved as 'uber_data_cleaned.csv' ({df.shape[0]} rows, {df.shape[1]} cols)")

# ---------------------------
# Descriptive Statistics → TXT
# ---------------------------
with open("data_analysis_summary.txt", "w") as f:
    f.write("=== Uber Ride Prediction — Data Analysis Summary ===\n\n")
    f.write(f"Shape after cleaning: {df.shape}\n\n")
    f.write("=== Descriptive Statistics ===\n")
    f.write(str(df.describe()) + "\n\n")
    # Correlation (only if 2+ numeric columns)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        f.write("=== Correlation Matrix ===\n")
        f.write(str(corr) + "\n\n")
        # Top correlation pairs (absolute)
        top_pairs = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
                .abs()
                .unstack()
                .dropna()
                .sort_values(ascending=False)
                .head(15)
        )
        f.write("=== Top |correlation| pairs ===\n")
        f.write(str(top_pairs) + "\n")
    else:
        f.write("Not enough numeric columns for correlation.\n")

print("📝 Stats written to data_analysis_summary.txt")

# ---------------------------
# Plots → PNG files (headless)
# ---------------------------
# Correlation heatmap only if enough numeric features
if df.select_dtypes(include=[np.number]).shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (ML-ready)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()

# Boxplots for core numeric features if present
for col in ["Avg VTAT","Avg CTAT","Booking Value","Ride Distance"]:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot — {col}")
        plt.tight_layout()
        plt.savefig(f"boxplot_{col.replace(' ', '_')}.png", dpi=150)
        plt.close()

print("📈 Saved plots (if applicable): correlation_heatmap.png, boxplot_*.png")
print("🎯 Done.")