import pandas as pd
import os

cities = ["oslo", "stockholm", "copenhagen"]
base_path = "data/processed"

for city in cities:
    print(f"\nValidating city: {city.capitalize()}")

    file_path = os.path.join(base_path, f"{city}_power_with_weather.csv")
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        continue

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # === Basic Preview ===
    print(df.head())
    print(df.info())

    # === 1. Date Continuity Check (Daily) ===
    print("Date range:", df.index.min(), "→", df.index.max())
    expected_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_days = expected_days.difference(df.index)
    print("Missing days:", missing_days)

    # === 2. Descriptive Stats ===
    print("\nDescriptive Statistics:")
    print(df.describe().T)

    # === 3. Missing Values ===
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # === 4. Outlier Detection: Actual Load ===
    Q1_load = df['Actual Load'].quantile(0.25)
    Q3_load = df['Actual Load'].quantile(0.75)
    IQR_load = Q3_load - Q1_load
    load_lower = Q1_load - 1.5 * IQR_load
    load_upper = Q3_load + 1.5 * IQR_load
    load_outliers = df[(df['Actual Load'] < load_lower) | (df['Actual Load'] > load_upper)]
    print(f"Outliers in Actual Load: {load_outliers.shape[0]}")

    # === 5. Outlier Detection: Price ===
    Q1_price = df['Price'].quantile(0.25)
    Q3_price = df['Price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    price_lower = Q1_price - 1.5 * IQR_price
    price_upper = Q3_price + 1.5 * IQR_price
    price_outliers = df[(df['Price'] < price_lower) | (df['Price'] > price_upper)]
    print(f"Outliers in Price: {price_outliers.shape[0]}")

    # === 6. Duplicate Timestamps ===
    dup_count = df.index.duplicated().sum()
    print("Duplicate timestamps:", dup_count)
