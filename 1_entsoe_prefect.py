# entsoe_prefect.py
from prefect import flow, task
import os
from dotenv import load_dotenv
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta

load_dotenv(dotenv_path=".env")

# Define your target bidding zones
regions = {
    "Stockholm": "SE_3",
    "Oslo": "NO_1",
    "Copenhagen": "DK_1"
}

# Time window: past 60 days
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")

@task
def initialize_entsoe_client():
    token = os.getenv("ENTSOE_TOKEN")
    return EntsoePandasClient(api_key=token)

@task
def fetch_demand_data(client, city, country_code, start_date, end_date):
    print(f"Fetching for {city} ({country_code}) from {start_date} to {end_date}")
    start_ts = pd.Timestamp(start_date, tz="Europe/Brussels")
    end_ts = pd.Timestamp(end_date, tz="Europe/Brussels")

    load_df = client.query_load(country_code, start=start_ts, end=end_ts)
    load_df.columns = ["Actual Load"]  # Ensure proper column name

    # Save path
    os.makedirs("data/entsoe", exist_ok=True)
    filename = f"data/entsoe/{city.lower()}_demand.csv"

    # Append or create
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename, index_col=0, parse_dates=True)
        combined = pd.concat([existing_df, load_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        combined.to_csv(filename)
        print(f"✅ Updated {filename} with new data")
    else:
        load_df.to_csv(filename)
        print(f"✅ Created {filename} with initial data")

    return load_df


@flow
def entsoe_demand_flow():
    client = initialize_entsoe_client()
    for city, code in regions.items():
        fetch_demand_data(client, city, code, start_date, end_date)

if __name__ == "__main__":
    entsoe_demand_flow()
