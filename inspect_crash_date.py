import requests
import zipfile
import pandas as pd
import os

def inspect_collision(zip_path="data/aisdk-2021-12.zip"):
    """
    Extract key datapoints around the Scot Carrier / Karin Høj collision
    on December 13, 2021 at ~02:27 UTC
    """
    
    print("Opening zip and reading Dec 13 CSV...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("aisdk-2021-12-13.csv") as f:
            df = pd.read_csv(f)

    # Clean up column name
    df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')

    print(f"Total records on Dec 13: {len(df):,}")

    # --- STEP 1: Find both vessels by name ---
    print("\n--- Searching by vessel name ---")
    mask = df['Name'].str.contains('SCOT|KARIN', case=False, na=False)
    vessels = df[mask]
    print(f"Records found: {len(vessels)}")
    print("Unique names:", vessels['Name'].unique())
    print("Unique MMSIs:", vessels['MMSI'].unique())

    # --- STEP 2: Focus on collision window (02:00 - 03:00 UTC) ---
    print("\n--- Tracks around collision time (02:00 - 03:00 UTC) ---")
    window = vessels[
        (vessels['Timestamp'] >= '2021-12-13 02:00:00') &
        (vessels['Timestamp'] <= '2021-12-13 03:00:00')
    ]
    print(window[['Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude', 'SOG', 'COG', 'Navigational status']].sort_values('Timestamp').to_string())

    # --- STEP 3: Check what happens AFTER collision (03:30 - 04:30 UTC) ---
    print("\n--- Tracks after collision (03:30 - 04:30 UTC) ---")
    after = vessels[
        (vessels['Timestamp'] >= '2021-12-13 03:30:00') &
        (vessels['Timestamp'] <= '2021-12-13 04:30:00')
    ]
    print(after[['Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude', 'SOG', 'Navigational status']].sort_values('Timestamp').to_string())

    # --- STEP 4: Full day summary per vessel ---
    print("\n--- Full day summary per vessel ---")
    for mmsi in vessels['MMSI'].unique():
        v = vessels[vessels['MMSI'] == mmsi]
        print(f"\nMMSI: {mmsi} | Name: {v['Name'].iloc[0]}")
        print(f"  Records: {len(v)}")
        print(f"  First seen: {v['Timestamp'].min()}")
        print(f"  Last seen:  {v['Timestamp'].max()}")
        print(f"  SOG range:  {v['SOG'].min():.1f} - {v['SOG'].max():.1f} knots")
        print(f"  Lat range:  {v['Latitude'].min():.4f} - {v['Latitude'].max():.4f}")
        print(f"  Lon range:  {v['Longitude'].min():.4f} - {v['Longitude'].max():.4f}")
        print(f"  Nav status: {v['Navigational status'].unique()}")

if __name__ == "__main__":
    # Wait for zip to exist before running
    if not os.path.exists("data/aisdk-2021-12.zip"):
        print("Zip not downloaded yet, please wait...")
    else:
        inspect_collision()