import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration
FEATURE_DIR = "data/features"
MODEL_DIR = "models"
CITIES = ["oslo", "stockholm", "copenhagen"]
TEST_HOURS_DEFAULT = 24 * 30  # 30 days

# Store results
price_results = []
demand_results = []

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

for city in CITIES:
    print(f"\n=== Evaluating: {city.title()} ===")

    path = os.path.join(FEATURE_DIR, f"{city}_features.csv")
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    if len(df) < 10:
        print(f"Not enough data for {city.title()} (only {len(df)} rows). Skipping.")
        continue

    # Train/test split
    test_size = TEST_HOURS_DEFAULT if len(df) > TEST_HOURS_DEFAULT + 10 else int(len(df) * 0.2)
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    feature_cols = [col for col in df.columns if col not in ['demand_next', 'price_next', 'name', 'description']]
    X_test = test[feature_cols]

    # ---- PRICE ----
    y_test_price = test['price_next']
    last_price = train['Price'].iloc[-1]
    naive_price = np.full(shape=y_test_price.shape, fill_value=last_price)

    try:
        ridge_p = pickle.load(open(f"{MODEL_DIR}/ridge_price_{city}.pkl", 'rb'))
        rf_p = pickle.load(open(f"{MODEL_DIR}/rf_price_{city}.pkl", 'rb'))
        xgb_p = pickle.load(open(f"{MODEL_DIR}/xgb_price_{city}.pkl", 'rb'))
    except FileNotFoundError:
        print(f"Missing price models for {city.title()}. Skipping price evaluation.")
        continue

    ridge_p_pred = ridge_p.predict(X_test)
    rf_p_pred = rf_p.predict(X_test)
    xgb_p_pred = xgb_p.predict(X_test)  # Check explicitly
    ensemble_p = 0.2 * ridge_p_pred + 0.3 * rf_p_pred + 0.5 * xgb_p_pred

    for name, preds in zip(
        ["Naive", "Ridge", "Random Forest", "XGBoost", "Ensemble"],
        [naive_price, ridge_p_pred, rf_p_pred, xgb_p_pred, ensemble_p]
    ):
        mae, rmse = evaluate(y_test_price, preds)
        price_results.append({
            "City": city.title(),
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

    # ---- DEMAND ----
    y_test_demand = test['demand_next']
    last_demand = train['Actual Load'].iloc[-1]
    naive_demand = np.full(shape=y_test_demand.shape, fill_value=last_demand)

    try:
        ridge_d = pickle.load(open(f"{MODEL_DIR}/ridge_demand_{city}.pkl", 'rb'))
        rf_d = pickle.load(open(f"{MODEL_DIR}/rf_demand_{city}.pkl", 'rb'))
        xgb_d = pickle.load(open(f"{MODEL_DIR}/xgb_demand_{city}.pkl", 'rb'))
    except FileNotFoundError:
        print(f"Missing demand models for {city.title()}. Skipping demand evaluation.")
        continue

    ridge_d_pred = ridge_d.predict(X_test)
    rf_d_pred = rf_d.predict(X_test)
    xgb_d_pred = xgb_d.predict(X_test)  # Check explicitly
    ensemble_d = 0.2 * ridge_d_pred + 0.3 * rf_d_pred + 0.5 * xgb_d_pred

    for name, preds in zip(
        ["Naive", "Ridge", "Random Forest", "XGBoost", "Ensemble"],
        [naive_demand, ridge_d_pred, rf_d_pred, xgb_d_pred, ensemble_d]
    ):
        mae, rmse = evaluate(y_test_demand, preds)
        demand_results.append({
            "City": city.title(),
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        })

# ---- Print Summary Tables ----

if price_results:
    df_p = pd.DataFrame(price_results)
    print("\n=== PRICE MAE Comparison ===")
    print(df_p.pivot(index="City", columns="Model", values="MAE").to_string())

    print("\n=== PRICE RMSE Comparison ===")
    print(df_p.pivot(index="City", columns="Model", values="RMSE").to_string())

if demand_results:
    df_d = pd.DataFrame(demand_results)
    print("\n=== DEMAND MAE Comparison ===")
    print(df_d.pivot(index="City", columns="Model", values="MAE").to_string())

    print("\n=== DEMAND RMSE Comparison ===")
    print(df_d.pivot(index="City", columns="Model", values="RMSE").to_string())

if not price_results and not demand_results:
    print("\nNo results available. All evaluations were skipped or failed.")
