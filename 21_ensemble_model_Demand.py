import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt


# Config
CITIES = ['oslo', 'stockholm', 'copenhagen']
FEATURE_PATH = 'data/features'
MODEL_PATH = 'models'

os.makedirs(MODEL_PATH, exist_ok=True)

# Parameters for models
ridge_params = {'alpha': 1.0, 'random_state': 42}
rf_params = {'n_estimators': 100, 'random_state': 42}
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'random_state': 42}

# Evaluation metrics
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse

# Training and evaluation loop
for city in CITIES:
    print(f"\n=== Processing city: {city.title()} ===")

    # Load data
    file_path = os.path.join(FEATURE_PATH, f"{city}_features.csv")
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

    target_demand = 'demand_next'
    target_price = 'price_next'
    feature_cols = [col for col in df.columns if col not in [target_demand, target_price, 'name', 'description']]

    X = df[feature_cols]
    y_demand = df[target_demand]
    y_price = df[target_price]

    kf = KFold(n_splits=5, shuffle=False)

    print("\n--- Demand Forecasting ---")

    demand_preds_ridge = []
    demand_preds_rf = []
    demand_preds_xgb = []
    demand_true = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_demand.iloc[train_idx], y_demand.iloc[test_idx]

        ridge = Ridge(**ridge_params).fit(X_train, y_train)
        rf = RandomForestRegressor(**rf_params).fit(X_train, y_train)
        xgb = XGBRegressor(**xgb_params).fit(X_train, y_train)

        pred_ridge = ridge.predict(X_test)
        pred_rf = rf.predict(X_test)
        pred_xgb = xgb.predict(X_test)

        final_pred = (0.2 * pred_ridge + 0.3 * pred_rf + 0.5 * pred_xgb)

        demand_preds_ridge.extend(pred_ridge)
        demand_preds_rf.extend(pred_rf)
        demand_preds_xgb.extend(pred_xgb)
        demand_true.extend(y_test)

        print(f"Fold {i+1} done.")

    print("\nResults for Demand Prediction:")
    for name, preds in zip(['Ridge', 'Random Forest', 'XGBoost'], [demand_preds_ridge, demand_preds_rf, demand_preds_xgb]):
        mae, rmse = evaluate_model(demand_true, preds)
        print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    ensemble_pred = (0.2 * np.array(demand_preds_ridge) +
                     0.3 * np.array(demand_preds_rf) +
                     0.5 * np.array(demand_preds_xgb))
    mae_e, rmse_e = evaluate_model(demand_true, ensemble_pred)
    print(f"Ensemble: MAE = {mae_e:.2f}, RMSE = {rmse_e:.2f}")

    print("\n--- Saving final models ---")
    final_ridge = Ridge(**ridge_params).fit(X, y_demand)
    final_rf = RandomForestRegressor(**rf_params).fit(X, y_demand)
    final_xgb = XGBRegressor(**xgb_params).fit(X, y_demand)

    pickle.dump(final_ridge, open(f"{MODEL_PATH}/ridge_demand_{city}.pkl", 'wb'))
    pickle.dump(final_rf, open(f"{MODEL_PATH}/rf_demand_{city}.pkl", 'wb'))
    pickle.dump(final_xgb, open(f"{MODEL_PATH}/xgb_demand_{city}.pkl", 'wb'))

    print("Models saved for demand prediction.")

    plt.figure(figsize=(6, 6))
    plt.scatter(demand_true, ensemble_pred, alpha=0.7)
    plt.plot([min(demand_true), max(demand_true)], [min(demand_true), max(demand_true)], 'r--')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title(f"{city.title()} — Ensemble Prediction vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    dummy_pred = [np.mean(y_train)] * len(y_test)
    dummy_mae, dummy_rmse = evaluate_model(y_test, dummy_pred)
    print(f"Dummy baseline: MAE = {dummy_mae:.2f}, RMSE = {dummy_rmse:.2f}")
