from prefect import flow, task
import os
from dotenv import load_dotenv
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta

# Load environment variables (.env file must have ENTSOE_TOKEN)
load_dotenv(dotenv_path=".env")

# Define bidding zones for each city
regions = {
    "Stockholm": "SE_3",
    "Oslo": "NO_1",
    "Copenhagen": "DK_1"
}

# Define time period (past 60 days)
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")


@task
def initialize_entsoe_client():
    token = os.getenv("ENTSOE_TOKEN")
    return EntsoePandasClient(api_key=token)


@task
def fetch_price_data(client, city, country_code, start_date, end_date):
    print(f"Fetching prices for {city} ({country_code}) from {start_date} to {end_date}")

    start_ts = pd.Timestamp(start_date, tz="Europe/Brussels")
    end_ts = pd.Timestamp(end_date, tz="Europe/Brussels")

    try:
        price_series = client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
    except Exception as e:
        print(f"Error fetching data for {city}: {e}")
        return

    # Convert to DataFrame
    price_df = price_series.to_frame(name="Day-Ahead Price")
    price_df.index.name = "Datetime"

    # Save to CSV
    os.makedirs("data/entsoe", exist_ok=True)
    filename = f"data/entsoe/{city.lower()}_price.csv"

    if os.path.exists(filename):
        existing = pd.read_csv(filename, index_col=0, parse_dates=True)
        combined = pd.concat([existing, price_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_csv(filename)
        print(f"Updated: {filename}")
    else:
        price_df.to_csv(filename)
        print(f"Created: {filename}")


@flow
def entsoe_price_flow():
    client = initialize_entsoe_client()
    for city, code in regions.items():
        fetch_price_data(client, city, code, start_date, end_date)


if __name__ == "__main__":
    entsoe_price_flow()
