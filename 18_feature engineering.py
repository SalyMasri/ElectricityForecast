import pandas as pd
import os

# Define cities and paths
cities = {
    "stockholm": "data/processed/stockholm_power_with_weather.csv",
    "oslo": "data/processed/oslo_power_with_weather.csv",
    "copenhagen": "data/processed/copenhagen_power_with_weather.csv"
}

# Output directory
output_dir = "data/features"
os.makedirs(output_dir, exist_ok=True)

for city, path in cities.items():
    print(f"\nProcessing: {city.title()}")

    # Load merged daily data
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # === LAG FEATURES ===
    df["demand_lag1"] = df["Actual Load"].shift(1)
    df["price_lag1"] = df["Price"].shift(1)
    df["demand_lag7"] = df["Actual Load"].shift(7)
    df["price_lag7"] = df["Price"].shift(7)

    # === ROLLING FEATURES ===
    df["demand_roll7"] = df["Actual Load"].rolling(window=7).mean().shift(1)
    df["price_roll7"] = df["Price"].rolling(window=7).mean().shift(1)
    df["temp_roll7"] = df["temp_C"].rolling(window=7).mean().shift(1)

    # === DIFFERENCES FROM PREVIOUS DAY/WEEK ===
    df["demand_diff1"] = df["Actual Load"] - df["demand_lag1"]
    df["demand_diff7"] = df["Actual Load"] - df["demand_lag7"]
    df["price_diff1"] = df["Price"] - df["price_lag1"]
    df["price_diff7"] = df["Price"] - df["price_lag7"]

    # === TARGETS FOR PREDICTION (T+1) ===
    df["demand_next"] = df["Actual Load"].shift(-1)
    df["price_next"] = df["Price"].shift(-1)

    # === DROP ROWS WITH ANY NaNs FROM LAGS/ROLLS/TARGETS ===
    df.dropna(inplace=True)

    # === Save ===
    out_path = os.path.join(output_dir, f"{city}_features.csv")
    df.to_csv(out_path)
    print(f"Saved engineered features to: {out_path}")
    print(f"Final shape: {df.shape}")

print("\nAll cities processed.")
