# Electricity Demand & Weather Analysis – Nordic Cities

This repository contains a data pipeline and exploratory analysis of electricity demand, pricing, and weather patterns across three major Nordic cities: **Stockholm**, **Oslo**, and **Copenhagen**. The goal is to prepare high-quality datasets and uncover preliminary insights to support future forecasting models.

---

## Project Overview

This project integrates three key data sources:
- **Electricity Demand (Actual Load)** from [ENTSO-E](https://transparency.entsoe.eu/)
- **Day-Ahead Market Prices** from the same ENTSO-E source
- **Weather Observations** from the [OpenWeatherMap API](https://openweathermap.org/api)

All data has been fetched, validated, merged, and enriched with time-based features for analysis. The current stage concludes the initial preprocessing and exploratory data analysis phase.

---

## Folder Structure
```
.
├── data/
│ ├── entsoe/ # Raw electricity demand and price data
│ ├── weather/ # Raw weather data (daily)
│ └── processed/ # Cleaned, merged datasets per city (daily granularity)
├── flows/ # Prefect-based data ingestion scripts
├── plots/ # Visualizations and diagnostics
├── scripts/ # Preprocessing, EDA, validation
├── .env # Contains API keys (excluded from version control)
└── README.md # Project summary and documentation
```

---

## Setup & Requirements

**Dependencies:**
- Python 3.10+
- [Prefect 2.x](https://docs.prefect.io/)
- pandas, numpy, matplotlib, requests, dotenv

**Environment:**
Create a `.env` file in the project root with the following:
ENTSOE_TOKEN=your_entsoe_api_key
OWM_API_KEY=your_openweathermap_api_key


---

## Data Collection & Ingestion

Data collection is automated using Prefect flows:

- **Electricity Demand (`entsoe_prefect.py`)**  
  Fetches 60 days of historical load data from ENTSO-E per city.

- **Electricity Price (`entsoe_price_flow`)**  
  Retrieves hourly day-ahead market prices for each city.

- **Weather Data (`weather_current_flow`)**  
  Collects daily weather metrics (temperature, humidity, etc.) from OpenWeatherMap.

The collected data is saved in CSV format under `data/entsoe/` and `data/weather/`.

---

## Preprocessing & Validation

Preprocessing included the following steps:

1. **Timezone Standardization**  
   All timestamps were normalized to UTC and made timezone-naive to ensure alignment between sources.

2. **Data Merging**  
   - Load and price data were merged by timestamp.
   - Weather data was matched by date using city-specific coordinates.
   - Final merged datasets saved under `data/processed/`.

3. **Validation Tasks**
   - Confirmed datetime continuity (daily granularity).
   - Verified expected data types (`float64` for all numerical fields).
   - Checked for and handled missing values.
   - Detected outliers using the interquartile range (IQR) method.
   - Confirmed no duplicate timestamps.

4. **Resampling**  
   Hourly data was resampled to daily averages where needed to match weather data resolution.

---

## Feature Engineering

New time-based features were added to enhance modeling:

- `hour`, `day_of_week`, `month`, `is_weekend`  
  These help capture seasonal and behavioral electricity usage patterns.

---

## Exploratory Data Analysis (EDA)

Exploratory steps and insights included:

1. **Descriptive Statistics**  
   Summary of key variables: load, price, temperature, and humidity.

2. **Correlation Analysis**
   - Oslo: Strong negative correlation between temperature and load (–0.72)
   - Stockholm: Moderate negative temp-load correlation (–0.57)
   - Copenhagen: Weak correlation (–0.16)
   - Load vs Price showed weak to moderate positive correlation in all cities

3. **Linear Fit Models**
   - Estimated the slope of load vs temperature:
     - Oslo: –75 MW/°C
     - Stockholm: –115 MW/°C
     - Copenhagen: –12 MW/°C

4. **Scatter Plots**
   - Load vs Temperature
   - Price vs Load (colored by temperature)

5. **Histograms**
   - Load distribution per city (to detect skew, peaks)
   - Price distribution (to detect volatility, spikes)

6. **Weekday Profiles**
   - Load compared across days of the week
   - Lower demand observed on weekends, consistent with industrial slowdown

---

## Data Quality Summary

- **Missing Values:** All handled or verified absent
- **Outliers:** Logged; extreme values investigated and optionally capped
- **Duplicates:** None detected
- **Final Data:** All cities have merged datasets at **daily resolution**, validated and enriched with features

---

## Current Progress

This commit finalizes the **Week 1 goals**:
- Complete data acquisition pipeline
- Data cleaning and transformation
- Visual and statistical EDA
- Early insight generation to guide model development

The project is now ready for the next phase: **modeling and forecasting electricity demand and price dynamics** using statistical or machine learning approaches.

---

## Next Steps

- Introduce lagged features (e.g., yesterday’s load or temp)
- Define modeling objectives (short-term forecasting, anomaly detection, etc.)
- Build initial models (e.g., linear regression, XGBoost, or LSTM)
- Evaluate and iterate

---

## Version Control

This README accompanies a new commit including:
- Full preprocessing and validation pipeline
- Final merged datasets for each city
- Initial insights and correlation metrics
- Diagnostic plots

---

## License

This project is intended for educational and research purposes. Please review and comply with the data source licenses:
- [ENTSO-E Data License](https://www.entsoe.eu/data/data-privacy-policy/)
- [OpenWeatherMap Terms of Use](https://openweathermap.org/terms)
