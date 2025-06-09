import pandas as pd

# === Step 1: Load merged power data ===
file_path = "data/entsoe/copenhagen_power.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# === Step 2: Add datetime-based features ===
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
df["month"] = df.index.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# === Step 3: Save back to same file ===
df.to_csv(file_path)
print(f"Features added and file updated: {file_path}")
print(df.head())
