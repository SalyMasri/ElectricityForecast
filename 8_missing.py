# 7_check_missing.py

import pandas as pd

# Load the merged data
power_df = pd.read_csv("data/processed/stockholm_power.csv", index_col=0, parse_dates=True)

# Check total missing values
print("Missing values:\n", power_df.isnull().sum())

# Identify exact timestamps with any NaNs
null_hours = power_df[power_df.isnull().any(axis=1)].index
print("Timestamps with missing data:")
print(null_hours)

# Summary
print(f"\nTotal rows: {len(power_df)}")
print(f"Any NaNs?: {power_df.isnull().values.any()}")
