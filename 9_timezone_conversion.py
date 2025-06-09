
import pandas as pd

# Load your merged demand + price dataset
power_df = pd.read_csv("data/processed/copenhagen_power.csv", index_col=0, parse_dates=True)

# If the index is already timezone-aware (like Europe/Brussels), convert to UTC
if power_df.index.tz is None:
    # Try localizing first to Brussels, then convert to UTC
    power_df.index = power_df.index.tz_localize("Europe/Brussels").tz_convert("UTC")
else:
    power_df.index = power_df.index.tz_convert("UTC")

# Optionally remove the timezone info to make the index "naive"
power_df.index = power_df.index.tz_localize(None)

# Confirm
print("Timezone conversion complete.")
print("Index timezone:", power_df.index.tz)
print("Index dtype:", power_df.index.dtype)
print(power_df.head(2))
