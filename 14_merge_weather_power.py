
import pandas as pd

# Load daily power data
power_df = pd.read_csv("data/processed/copenhagen_power.csv", parse_dates=["datetime"])
power_df.set_index("datetime", inplace=True)

# Load and prepare full weather data
weather_df = pd.read_csv("data/weather/copenhagen_current_new.csv", parse_dates=["datetime"])
weather_df.set_index("datetime", inplace=True)

# Convert temperature and keep all fields
weather_df["temp_C"] = weather_df["temp"]
weather_df.drop(columns=["temp"], inplace=True)

# === Step 1: Timezone handling ===
power_utc_df = power_df.copy()
power_utc_df.index = pd.to_datetime(power_utc_df.index, utc=True)
power_utc_df.index = power_utc_df.index.tz_convert(None)

weather_utc_df = weather_df.copy()
weather_utc_df.index = pd.to_datetime(weather_utc_df.index, utc=True)
weather_utc_df.index = weather_utc_df.index.tz_convert(None)

# === Step 2: Merge on date ===
full_df = power_utc_df.join(weather_utc_df, how="inner")

# === Step 3: Validation ===
print("Columns:", full_df.columns.tolist())
print(" Shape:", full_df.shape)
print(" Date Range:", full_df.index.min(), "→", full_df.index.max())
print(" Missing values:\n", full_df.isnull().sum())

# === Step 4: Save merged dataset ===
full_df.to_csv("data/processed/copenhagen_power_with_weather.csv")
print(" Saved to data/processed/copenhagen_power_with_weather.csv")
