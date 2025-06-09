import pandas as pd
import os

cities = ["stockholm", "oslo", "copenhagen"]

for city in cities:
    filename = f"data/entsoe/{city}_demand.csv"
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

    print(f"\nInspecting {filename}")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    print("Head:")
    print(df.head(3))

    print("Tail:")
    print(df.tail(3))

    print("Info:")
    print(df.info())

    print("Missing values:")
    print(df.isnull().sum())

    print("Any NaNs?:", df.isnull().values.any())

    print("Total rows:", df.shape[0])
