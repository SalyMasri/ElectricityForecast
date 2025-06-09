
import pandas as pd

# Load your existing hourly power data
power_df = pd.read_csv("data/processed/stockholm_power.csv", parse_dates=["Unnamed: 0"])
power_df.set_index("Unnamed: 0", inplace=True)
power_df.index.name = "datetime"

# Resample to daily averages
daily_power_df = power_df.resample("D").mean()

# Keep only relevant columns (drop hour/day_of_week/etc.)
daily_power_df = daily_power_df[["Actual Load", "Price"]]

# Save to the same file, overwriting it
daily_power_df.to_csv("data/processed/stockholm_power.csv")

print(" Overwritten with daily-averaged power data")
