import zipfile
import pandas as pd
import folium
import os

# --- CONFIGURATION ---
ZIP_PATH = "data/aisdk-2021-12.zip"
TARGET_CSV = "aisdk-2021-12-13.csv"
CANDIDATES_CSV = "suspected_collisions_list.csv"
OUTPUT_DIR = "encounter_maps_fullday"

def generate_fullday_trajectory_maps():
    print(f"1. Loading datasets...")
    try:
        candidates = pd.read_csv(CANDIDATES_CSV)
        
        # Read the raw data directly from the zip file, just like Phase 1
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(TARGET_CSV) as f:
                df = pd.read_csv(f)
        
        if '# Timestamp' in df.columns:
            df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df['MMSI'] = df['MMSI'].astype(str).str.strip()
        candidates['Timestamp_A'] = pd.to_datetime(candidates['Timestamp_A'])
    except Exception as e:
        print(f"Error loading files: {e}. Make sure Phase 1 has generated the CSV and the zip file is in data/.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n2. Generating FULL DAY maps for {len(candidates)} candidate pairs...\n")

    for idx, row in candidates.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a = str(row['MMSI_A'])
        mmsi_b = str(row['MMSI_B'])
        name_a = str(row['Name_A']).replace("/", "-") # Clean names for file saving
        name_b = str(row['Name_B']).replace("/", "-")
        
        # Extract FULL DAY tracks for both vessels
        track_a = df[df['MMSI'] == mmsi_a].sort_values('Timestamp')
        track_b = df[df['MMSI'] == mmsi_b].sort_values('Timestamp')
        
        if track_a.empty and track_b.empty:
            print(f"   -> Skipping {name_a} vs {name_b} (No data found)")
            continue
            
        # Find the center point for the map (closest ping to the impact time)
        try:
            if not track_a.empty:
                impact_point = track_a.iloc[(track_a['Timestamp'] - t_impact).abs().argsort()[:1]].iloc[0]
            else:
                impact_point = track_b.iloc[(track_b['Timestamp'] - t_impact).abs().argsort()[:1]].iloc[0]
                
            center_lat, center_lon = impact_point['Latitude'], impact_point['Longitude']
        except:
            continue # Fallback if coordinate extraction fails

        # Initialize the map zoomed out slightly to see the day's route
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB Positron')
        
        # --- PLOT VESSEL A (BLUE) ---
        if not track_a.empty:
            # Draw the exact continuous path
            coords_a = list(zip(track_a['Latitude'], track_a['Longitude']))
            folium.PolyLine(coords_a, color="blue", weight=2.5, opacity=0.8).add_to(m)
            
            # Add interactive dots (sub-sampled to every 15th ping to prevent browser crash)
            marker_sample_a = track_a.iloc[::15]
            for _, point in marker_sample_a.iterrows():
                time_str = point['Timestamp'].strftime('%H:%M:%S')
                popup_text = f"<b>{name_a}</b><br>Time: {time_str}<br>Speed: {point['SOG']} kts<br>Heading: {point['COG']}°"
                folium.CircleMarker(
                    location=[point['Latitude'], point['Longitude']],
                    radius=3, color="blue", fill=True, fill_color="blue", fill_opacity=0.6,
                    popup=folium.Popup(popup_text, max_width=200)
                ).add_to(m)

        # --- PLOT VESSEL B (RED) ---
        if not track_b.empty:
            # Draw the exact continuous path
            coords_b = list(zip(track_b['Latitude'], track_b['Longitude']))
            folium.PolyLine(coords_b, color="red", weight=2.5, opacity=0.8).add_to(m)
            
            # Add interactive dots (sub-sampled to every 15th ping)
            marker_sample_b = track_b.iloc[::15]
            for _, point in marker_sample_b.iterrows():
                time_str = point['Timestamp'].strftime('%H:%M:%S')
                popup_text = f"<b>{name_b}</b><br>Time: {time_str}<br>Speed: {point['SOG']} kts<br>Heading: {point['COG']}°"
                folium.CircleMarker(
                    location=[point['Latitude'], point['Longitude']],
                    radius=3, color="red", fill=True, fill_color="red", fill_opacity=0.6,
                    popup=folium.Popup(popup_text, max_width=200)
                ).add_to(m)

        # --- HIGHLIGHT THE ENCOUNTER POINT ---
        folium.Marker(
            [center_lat, center_lon],
            icon=folium.Icon(color='purple', icon='bolt', prefix='fa'),
            popup=f"<b>ENCOUNTER TRIGGERED</b><br>Time: {t_impact}"
        ).add_to(m)

        # Save the map
        safe_filename = f"FullDay_{idx:02d}_{name_a.replace(' ', '_')}_vs_{name_b.replace(' ', '_')}.html"
        save_path = os.path.join(OUTPUT_DIR, safe_filename)
        m.save(save_path)
        print(f"   -> Saved: {safe_filename}")

    print(f"\n✅ All full-day maps generated successfully in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    generate_fullday_trajectory_maps()