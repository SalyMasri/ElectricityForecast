import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
CITIES = ['oslo', 'stockholm', 'copenhagen']
FEATURE_PATH = 'data/features'
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

# Parameters
ridge_params = {'alpha': 1.0, 'random_state': 42}
rf_params = {'n_estimators': 100, 'random_state': 42}
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'random_state': 42}

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Loop through cities
for city in CITIES:
    print(f"\n=== Processing city: {city.title()} ===")
    file_path = os.path.join(FEATURE_PATH, f"{city}_features.csv")
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

    # Setup
    target = 'price_next'
    drop_cols = ['demand_next', 'price_next', 'name', 'description']
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df[target]

    # Storage
    preds_ridge = []
    preds_rf = []
    preds_xgb = []
    y_true = []

    print("\n--- Price Forecasting ---")
    kf = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_ridge = Ridge(**ridge_params).fit(X_train, y_train)
        model_rf = RandomForestRegressor(**rf_params).fit(X_train, y_train)
        model_xgb = XGBRegressor(**xgb_params).fit(X_train, y_train)

        pred_r = model_ridge.predict(X_test)
        pred_rf = model_rf.predict(X_test)
        pred_xgb = model_xgb.predict(X_test)

        preds_ridge.extend(pred_r)
        preds_rf.extend(pred_rf)
        preds_xgb.extend(pred_xgb)
        y_true.extend(y_test)

        print(f"Fold {fold + 1} done.")

    # Evaluate models
    print("\nResults for Price Prediction:")
    for name, preds in zip(['Ridge', 'Random Forest', 'XGBoost'], [preds_ridge, preds_rf, preds_xgb]):
        mae, rmse = evaluate_model(y_true, preds)
        print(f"{name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    # Ensemble
    ensemble = 0.2 * np.array(preds_ridge) + 0.3 * np.array(preds_rf) + 0.5 * np.array(preds_xgb)
    mae_e, rmse_e = evaluate_model(y_true, ensemble)
    print(f"Ensemble: MAE = {mae_e:.2f}, RMSE = {rmse_e:.2f}")

    # Dummy baseline
    dummy_pred = [np.mean(y_train)] * len(y_test)
    dummy_mae, dummy_rmse = evaluate_model(y_test, dummy_pred)
    print(f"Dummy baseline: MAE = {dummy_mae:.2f}, RMSE = {dummy_rmse:.2f}")

    # Save models
    print("\n--- Saving final models ---")
    final_ridge = Ridge(**ridge_params).fit(X, y)
    final_rf = RandomForestRegressor(**rf_params).fit(X, y)
    final_xgb = XGBRegressor(**xgb_params).fit(X, y)

    pickle.dump(final_ridge, open(f"{MODEL_PATH}/ridge_price_{city}.pkl", 'wb'))
    pickle.dump(final_rf, open(f"{MODEL_PATH}/rf_price_{city}.pkl", 'wb'))
    pickle.dump(final_xgb, open(f"{MODEL_PATH}/xgb_price_{city}.pkl", 'wb'))
    print("Models saved for price prediction.")

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, ensemble, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{city.title()} — Ensemble Price Prediction vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
