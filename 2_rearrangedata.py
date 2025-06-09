import pandas as pd

# === Define your mapping ===
name_to_coords = {
    "Stockholm, Stockholms län, Sverige": "59.3293,18.0686",
    "Oslo, Oslo, Norge": "59.9139,10.7522",
    "Copenhagen, Hovedstaden, Danmark": "55.6761,12.5683"
}

# === List of your files ===
files = [
    "data/weather/stockholm_current_new.csv",
    "data/weather/oslo_current_new.csv",
    "data/weather/copenhagen_current_new.csv"
]

# === Loop through files and update ===
for file in files:
    print(f"🔄 Processing: {file}")
    df = pd.read_csv(file, quotechar='"', skipinitialspace=True)
    
    if "name" in df.columns:
        df["name"] = df["name"].apply(lambda x: name_to_coords.get(x, x))  # Replace if match
        df.to_csv(file, index=False)
        print(f"✅ Updated and saved: {file}")
    else:
        print(f"⚠️ Skipped {file} (no 'name' column found)")
