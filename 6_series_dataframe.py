import pandas as pd

# === Load demand ===
load_df = pd.read_csv("data/entsoe/copenhagen_demand.csv", index_col=0, parse_dates=True)

# === Load price from city_price.csv ===
price_df = pd.read_csv("data/entsoe/copenhagen_price.csv", parse_dates=["Datetime"])
price_df.set_index("Datetime", inplace=True)
price_df.rename(columns={"Day-Ahead Price": "Price"}, inplace=True)

# === Ensure timezone is set and matches demand ===
if price_df.index.tz is None:
    price_df.index = price_df.index.tz_localize("Europe/Brussels")
if load_df.index.tz is None:
    load_df.index = load_df.index.tz_localize("Europe/Brussels")

# === Sanity check previews ===
print("price_df preview:")
print(price_df.head(1))
print("price_df timezone:", price_df.index.tz)
print("load_df timezone:", load_df.index.tz)

# === Merge on datetime ===
power_df = load_df.join(price_df, how="inner")

# === Check merged data ===
print("\nMerged power_df:")
print(power_df.head())
print("Shape:", power_df.shape)
print("Missing values:\n", power_df.isnull().sum())
power_df.to_csv("data/entsoe/copenhagen_power.csv")
print(" Saved merged data to: data/entsoe/copenhagen_power.csv")
