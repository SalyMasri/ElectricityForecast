import pandas as pd

# Input: original 15-minute data
input_file = "data/entsoe/oslo_demand.csv"
# Output: new hourly data file
output_file = "data/entsoe/oslo_demand_hourly.csv"

# Read CSV with datetime index
df = pd.read_csv(input_file, index_col=0, parse_dates=True)

# Inspect original
print("Original entries:", df.shape[0])
print("Original index sample:", df.index[:4])

# Resample to hourly averages
df_hourly = df.resample("1H").mean()

# Check result
print("Hourly rows:", df_hourly.shape[0])
print(df_hourly.head(3))

# Save to new file
df_hourly.to_csv(output_file)
print(f" Hourly data saved to: {output_file}")
