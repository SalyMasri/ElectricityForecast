import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta

# Configuration
CITIES = ["oslo", "stockholm", "copenhagen"]
FEATURE_DIR = "data/features"
MODEL_DIR = "models"
FORECAST_DIR = "data/forecast/"
EXCLUDE_COLS = ['demand_next', 'price_next', 'name', 'description']

def get_forecast(city, n_days=7):
    # Validation
    if not isinstance(n_days, int) or n_days <= 0:
        raise ValueError("n_days must be a positive integer.")

    df = pd.read_csv(os.path.join(FEATURE_DIR, f"{city}_features.csv"), parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    if len(df) < 7:
        raise ValueError(f"Not enough historical data for {city} (need at least 7 days).")

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

    # Load models
    try:
        ridge_d = pickle.load(open(f"{MODEL_DIR}/ridge_demand_{city}.pkl", "rb"))
        rf_d = pickle.load(open(f"{MODEL_DIR}/rf_demand_{city}.pkl", "rb"))
        xgb_d = pickle.load(open(f"{MODEL_DIR}/xgb_demand_{city}.pkl", "rb"))

        ridge_p = pickle.load(open(f"{MODEL_DIR}/ridge_price_{city}.pkl", "rb"))
        rf_p = pickle.load(open(f"{MODEL_DIR}/rf_price_{city}.pkl", "rb"))
        xgb_p = pickle.load(open(f"{MODEL_DIR}/xgb_price_{city}.pkl", "rb"))
    except FileNotFoundError as e:
        print(f"Model missing for {city}: {e}")
        return None

    # Forecasting loop
    history = df.copy()
    current_time = df.index[-1]
    predictions = []

    for _ in range(n_days):
        forecast_time = current_time + timedelta(days=1)
        last_row = history.iloc[-1].copy()
        input_row = last_row.copy()

        # Update lag features
        if 'demand_lag1' in feature_cols:
            input_row['demand_lag1'] = last_row['Actual Load']
        if 'price_lag1' in feature_cols:
            input_row['price_lag1'] = last_row['Price']
        if 'demand_diff1' in feature_cols and len(history) >= 2:
            input_row['demand_diff1'] = last_row['Actual Load'] - history.iloc[-2]['Actual Load']
        if 'price_diff1' in feature_cols and len(history) >= 2:
            input_row['price_diff1'] = last_row['Price'] - history.iloc[-2]['Price']
        if 'demand_lag7' in feature_cols and len(history) >= 7:
            input_row['demand_lag7'] = history.iloc[-7]['Actual Load']
        if 'price_lag7' in feature_cols and len(history) >= 7:
            input_row['price_lag7'] = history.iloc[-7]['Price']
        if 'demand_diff7' in feature_cols and len(history) >= 7:
            input_row['demand_diff7'] = last_row['Actual Load'] - history.iloc[-7]['Actual Load']
        if 'price_diff7' in feature_cols and len(history) >= 7:
            input_row['price_diff7'] = last_row['Price'] - history.iloc[-7]['Price']
        if 'demand_roll7' in feature_cols and len(history) >= 7:
            input_row['demand_roll7'] = history['Actual Load'].iloc[-7:].mean()
        if 'price_roll7' in feature_cols and len(history) >= 7:
            input_row['price_roll7'] = history['Price'].iloc[-7:].mean()
        if 'temp_roll7' in feature_cols and 'temp_C' in history.columns and len(history) >= 7:
            input_row['temp_roll7'] = history['temp_C'].iloc[-7:].mean()

        X_input = pd.DataFrame([input_row[feature_cols].values], columns=feature_cols)

        pred_demand = (
            0.2 * ridge_d.predict(X_input) +
            0.3 * rf_d.predict(X_input) +
            0.5 * xgb_d.predict(X_input)
        )[0]

        pred_price = (
            0.2 * ridge_p.predict(X_input) +
            0.3 * rf_p.predict(X_input) +
            0.5 * xgb_p.predict(X_input)
        )[0]

        predictions.append({
            "datetime": forecast_time,
            "predicted_demand": pred_demand,
            "predicted_price": pred_price
        })

        # Update history
        new_row = last_row.copy()
        new_row["Actual Load"] = pred_demand
        new_row["Price"] = pred_price
        new_row["datetime"] = forecast_time
        history.loc[forecast_time] = new_row
        current_time = forecast_time

    result_df = pd.DataFrame(predictions).set_index("datetime")
    result_df.to_csv(os.path.join(FORECAST_DIR, f"forecast_{city}.csv"))
    print(f"Saved forecast to forecast_{city}.csv")
    return result_df


# ==== Main Execution ====
for city in CITIES:
    print(f"\n=== {city.upper()} ===")

    # Forecast
    forecast_df = get_forecast(city, n_days=7)
    if forecast_df is None:
        continue

    # Load actuals
    actual_df = pd.read_csv(os.path.join(FEATURE_DIR, f"{city}_features.csv"), parse_dates=["datetime"], index_col="datetime")
    recent_demand = actual_df["Actual Load"].iloc[-7:]
    recent_price = actual_df["Price"].iloc[-7:]

    # Plot Demand
    plt.figure(figsize=(10, 3))
    plt.plot(recent_demand.index, recent_demand.values, label="Actual Demand (last 7d)", color="black")
    plt.plot(forecast_df.index, forecast_df["predicted_demand"], label="Forecast Demand (next 7d)", color="blue")
    plt.title(f"{city.title()} — Demand Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Price
    plt.figure(figsize=(10, 3))
    plt.plot(recent_price.index, recent_price.values, label="Actual Price (last 7d)", color="black")
    plt.plot(forecast_df.index, forecast_df["predicted_price"], label="Forecast Price (next 7d)", color="green")
    plt.title(f"{city.title()} — Price Forecast")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Optional: test different forecast durations
try:
    print("\n=== Testing Other Horizons ===")
    test_3d = get_forecast("oslo", n_days=3)
    print("3-day forecast:")
    print(test_3d)

    test_14d = get_forecast("oslo", n_days=14)
    print("14-day forecast length:", len(test_14d))

except Exception as e:
    print(f"Test error: {e}")
