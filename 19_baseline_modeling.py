import os
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
CITIES = ['oslo', 'stockholm', 'copenhagen']
FEATURE_PATH = 'data/features'
MODEL_PATH = 'models'
os.makedirs(MODEL_PATH, exist_ok=True)

for city in CITIES:
    print(f"\n=== Training model for: {city.title()} ===")

    # Load data
    filepath = f"{FEATURE_PATH}/{city}_features.csv"
    df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    print("Data shape:", df.shape)

    # Identify target columns
    target_demand = 'demand_next'
    target_price = 'price_next'

    # Drop non-numeric columns automatically (e.g. 'name', 'description')
    non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    if non_numeric_cols:
        print(f"Excluding non-numeric columns from features: {non_numeric_cols}")

    feature_cols = [col for col in df.columns if col not in [target_demand, target_price] + non_numeric_cols]

    # Train-test split: last 30 days for test
    test_days = 4
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]

    print(f"Train size: {train.shape}, Test size: {test.shape}")

    # Prepare X and y for demand
    X_train_demand = train[feature_cols]
    y_train_demand = train[target_demand]
    X_test_demand = test[feature_cols]
    y_test_demand = test[target_demand]

    # Prepare X and y for price
    X_train_price = train[feature_cols]
    y_train_price = train[target_price]
    X_test_price = test[feature_cols]
    y_test_price = test[target_price]

    # Train demand model
    model_demand = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_demand.fit(X_train_demand, y_train_demand)

    # Train price model
    model_price = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_price.fit(X_train_price, y_train_price)

    # Save models
    with open(f'{MODEL_PATH}/xgb_demand_{city}.pkl', 'wb') as f:
        pickle.dump(model_demand, f)
    with open(f'{MODEL_PATH}/xgb_price_{city}.pkl', 'wb') as f:
        pickle.dump(model_price, f)

    # Evaluate on training set
    train_pred_demand = model_demand.predict(X_train_demand)
    train_mae_d = mean_absolute_error(y_train_demand, train_pred_demand)
    train_rmse_d = mean_squared_error(y_train_demand, train_pred_demand) ** 0.5

    train_pred_price = model_price.predict(X_train_price)
    train_mae_p = mean_absolute_error(y_train_price, train_pred_price)
    train_rmse_p = mean_squared_error(y_train_price, train_pred_price) ** 0.5


    print(f"Training Demand - MAE: {train_mae_d:.2f}, RMSE: {train_rmse_d:.2f}")
    print(f"Training Price  - MAE: {train_mae_p:.2f}, RMSE: {train_rmse_p:.2f}")




# === Training model for: Oslo ===
# Data shape: (51, 22)
# Excluding non-numeric columns from features: ['name', 'description']
# Train size: (47, 22), Test size: (4, 22)
# Training Demand - MAE: 0.00, RMSE: 0.00
# Training Price  - MAE: 0.00, RMSE: 0.00

# === Training model for: Stockholm ===
# Data shape: (51, 22)
# Excluding non-numeric columns from features: ['name', 'description']
# Train size: (47, 22), Test size: (4, 22)
# Training Demand - MAE: 0.00, RMSE: 0.00
# Training Price  - MAE: 0.00, RMSE: 0.00

# === Training model for: Copenhagen ===
# Data shape: (51, 22)
# Excluding non-numeric columns from features: ['name', 'description']
# Train size: (47, 22), Test size: (4, 22)
# Training Demand - MAE: 0.00, RMSE: 0.00
# Training Price  - MAE: 0.00, RMSE: 0.00