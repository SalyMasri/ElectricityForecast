from prefect import flow, task
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv(dotenv_path=".env")

@task
def fetch_and_append_weather(city, lat, lon):
    key = os.getenv("OWM_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}"
    resp = requests.get(url)

    if resp.status_code != 200:
        print(f"Failed for {city}: {resp.status_code}")
        return

    data = resp.json()

    # === Extract & format for your target CSV ===
    row = {
        "datetime": datetime.utcnow().strftime('%Y-%m-%d'),
        "name": f"{lat},{lon}",
        "temp": round(data["main"].get("temp", 0) - 273.15, 1),  # Kelvin to °C
        "humidity": data["main"].get("humidity"),
        "pressure": data["main"].get("pressure"),
        "description": data["weather"][0].get("description", ""),
        "windspeed": data["wind"].get("speed"),
        "cloudcover": data.get("clouds", {}).get("all")
    }

    df = pd.DataFrame([row])

    # === Save to CSV ===
    filename = f"data/weather/{city.lower()}_current_new.csv"
    os.makedirs("data/weather", exist_ok=True)

    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, mode="w", header=True, index=False)

    print(f"✅ {city} data appended to {filename}")

@flow
def weather_current_flow():
    cities = {
        "Stockholm": (59.3293, 18.0686),
        "Oslo": (59.9139, 10.7522),
        "Copenhagen": (55.6761, 12.5683)
    }

    for city, (lat, lon) in cities.items():
        fetch_and_append_weather(city, lat, lon)

if __name__ == "__main__":
    weather_current_flow()
