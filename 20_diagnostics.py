import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

# Setup
CITIES = ['oslo', 'stockholm', 'copenhagen']
FEATURE_PATH = 'data/features'
TARGETS = ['demand_next', 'price_next']
NON_NUMERIC_TO_EXCLUDE = ['name', 'description']

# Plot setup
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Run diagnostics for each city
for city in CITIES:
    print("="*50)
    print(f" Dataset Diagnostics for {city.title()}")
    print("="*50)

    filepath = os.path.join(FEATURE_PATH, f"{city}_features.csv")
    df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    print(" Data loaded:", df.shape)

    # 1. Label completeness
    print("\n Label Check:")
    for target in TARGETS:
        if target in df.columns:
            n_missing = df[target].isnull().sum()
            print(f"  {target}: {' OK' if n_missing == 0 else f' Missing {n_missing} values'}")
        else:
            print(f"  {target}:  Column missing")

    # 2. Target distribution
    print("\n Target Distribution (Summary):")
    for target in TARGETS:
        if target in df.columns:
            desc = df[target].describe()
            print(f"  {target}: min={desc['min']:.2f}, max={desc['max']:.2f}, mean={desc['mean']:.2f}, std={desc['std']:.2f}")

            # Plot distribution
            plt.figure(figsize=(5, 2))
            sns.histplot(df[target], kde=True, bins=20)
            plt.title(f"{city.title()} — {target}")
            plt.tight_layout()
            plt.show()

    # 3. Outlier detection (numeric only)
    print("\n Outlier Detection (Isolation Forest):")
    numeric_df = df.select_dtypes(include=['number']).drop(columns=TARGETS, errors='ignore')
    if len(numeric_df) > 0:
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(numeric_df.fillna(0))
        outlier_count = np.sum(preds == -1)
        print(f"  Estimated outliers: {outlier_count} / {len(numeric_df)} rows")
    else:
        print("   No numeric features to run outlier detection.")

    # 4. Feature importance (XGBoost baseline for demand)
    print("\n Feature Importance (XGBoost - demand prediction):")
    feature_cols = [
        col for col in df.columns
        if col not in TARGETS + NON_NUMERIC_TO_EXCLUDE and df[col].dtype in ['float64', 'int64']
    ]
    if len(feature_cols) > 0:
        X = df[feature_cols].fillna(0)
        y = df['demand_next']
        model = XGBRegressor(n_estimators=50, objective='reg:squarederror', random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)

        print(importances.head(5))

        # Plot top 10
        plt.figure(figsize=(6, 3))
        importances.head(10).plot(kind='barh')
        plt.title(f"{city.title()} — Top Features for Demand")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("  No numeric features found.")

    print("\n Done with:", city.title())
    print("\n\n")
