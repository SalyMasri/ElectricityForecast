
import pandas as pd

# This assumes you already standardized timezone and index
power_df = pd.read_csv("data/processed/copenhagen_power.csv", index_col=0, parse_dates=True)

# If needed, standardize timezone again just to be sure
if power_df.index.tz is None:
    power_df.index = power_df.index.tz_localize("Europe/Brussels").tz_convert("UTC")
power_df.index = power_df.index.tz_localize(None)

# Save to final processed CSV
final_path = "data/processed/copenhagen_power.csv"
power_df.to_csv(final_path, index=True)

print(f" Power data saved to: {final_path}")
print(" Final preview:")
print(power_df.head(3))
