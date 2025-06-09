import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No GUI backend
import matplotlib.pyplot as plt
import os
import numpy as np

cities = ["oslo", "stockholm", "copenhagen"]
base_path = "data/processed"

for city in cities:
    print(f"\n=== {city.capitalize()} ===")

    file_path = os.path.join(base_path, f"{city}_power_with_weather.csv")
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        continue

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # --- Feature Engineering ---
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # === 1. Average Load by Day of Week ===
    weekday_profile = df.groupby('day_of_week')["Actual Load"].mean()
    weekday_profile.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("Average Load by Day of Week:")
    print(weekday_profile.round(2))

    # Save plot
    weekday_profile.plot(kind="bar", title=f"{city.capitalize()} - Avg Load by Day of Week")
    plt.ylabel("Average Load (MW)")
    plt.tight_layout()
    plt.savefig(f"{city}_avg_load_by_weekday.png")
    plt.close()

    # === 2. Correlation Matrix ===
    corr_matrix = df[['Actual Load','Price','temp_C','humidity']].corr()
    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))

    # === 3. Scatter Stats – Load vs Temperature ===
    corr_temp = df["Actual Load"].corr(df["temp_C"])
    coef = np.polyfit(df["temp_C"], df["Actual Load"], 1)
    print(f"\nLoad vs Temperature:")
    print(f"  Correlation: {corr_temp:.3f}")
    print(f"  Linear fit: Load ≈ {coef[0]:.2f} * Temp + {coef[1]:.2f}")

    # Plot
    plt.scatter(df['temp_C'], df['Actual Load'], alpha=0.5)
    plt.title(f"{city.capitalize()} - Load vs Temperature")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(f"load_vs_temp_{city}.png")
    plt.close()

    # === 4. Scatter Stats – Price vs Load ===
    corr_price = df["Price"].corr(df["Actual Load"])
    print(f"\nPrice vs Load:")
    print(f"  Correlation: {corr_price:.3f}")

    plt.scatter(df['Actual Load'], df['Price'], alpha=0.5, c=df['temp_C'], cmap='coolwarm')
    plt.title(f"{city.capitalize()} - Price vs Load (color: Temp)")
    plt.xlabel("Load (MW)"); plt.ylabel("Price (EUR/MWh)")
    plt.tight_layout()
    plt.savefig(f"price_vs_load_{city}.png")
    plt.close()

    # === 5. Histograms (summarize as text too) ===
    load_counts, load_bins = np.histogram(df['Actual Load'], bins=20)
    price_counts, price_bins = np.histogram(df['Price'], bins=20)

    print("\nLoad Histogram (bin ranges and counts):")
    for i in range(len(load_counts)):
        print(f"  {load_bins[i]:.1f} – {load_bins[i+1]:.1f} : {load_counts[i]}")

    print("\nPrice Histogram (bin ranges and counts):")
    for i in range(len(price_counts)):
        print(f"  {price_bins[i]:.2f} – {price_bins[i+1]:.2f} : {price_counts[i]}")

    # Save histogram plots
    df["Actual Load"].hist(bins=20)
    plt.title(f"{city.capitalize()} - Load Distribution")
    plt.xlabel("Load (MW)"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"hist_load_{city}.png")
    plt.close()

    df["Price"].hist(bins=20, color="orange")
    plt.title(f"{city.capitalize()} - Price Distribution")
    plt.xlabel("Price (EUR/MWh)"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"hist_price_{city}.png")
    plt.close()
