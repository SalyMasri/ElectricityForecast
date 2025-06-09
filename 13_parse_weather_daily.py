
import pandas as pd

# Load daily weather file
weather_df = pd.read_csv("data/weather/stockholm_current_new.csv", parse_dates=["datetime"])

# Set datetime index
weather_df.set_index("datetime", inplace=True)
weather_df.index.name = "date"

# Rename and select only needed columns
weather_df.rename(columns={
    "temp": "temp_C"
}, inplace=True)

weather_df = weather_df[["temp_C", "humidity"]]

# Sort just in case
weather_df.sort_index(inplace=True)

# Inspect
print("Parsed weather_df:")
print(weather_df.head(3))
print(weather_df.index[:3])
print("dtypes:\n", weather_df.dtypes)
