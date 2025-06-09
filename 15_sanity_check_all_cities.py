import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

cities = ["oslo", "stockholm", "copenhagen"]
data_path = "data/processed"
plot_path = "plots"
os.makedirs(plot_path, exist_ok=True)

for city in cities:
    print(f"\nChecking: {city.capitalize()}")

    file_path = os.path.join(data_path, f"{city}_power_with_weather.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # Plot the first 48 hours and save to file
    df[['Actual Load', 'Price']].iloc[:48].plot(subplots=True, title=f"{city.capitalize()} - First 2 Days Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"{city}_sample.png"))
    plt.close()

    # Correlation check
    if 'temp_C' in df.columns:
        corr = df['Actual Load'].corr(df['temp_C'])
        print(f"Correlation (Load vs Temp_C): {corr:.3f}")
    else:
        print("Column 'temp_C' not found.")

    # Data type check
    print("Data Types:")
    print(df.dtypes)
