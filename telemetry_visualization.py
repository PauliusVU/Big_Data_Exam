import sys
import os
import pandas as pd
import folium
from folium import plugins

# =============================================================================
# USAGE: python visualize.py telemetry_KARIN_HOEJ_vs_MV_SCOT_CARRIER.csv
# =============================================================================

if len(sys.argv) < 2:
    print("Usage: python visualize.py <telemetry_csv_file>")
    sys.exit(1)

CSV_PATH = sys.argv[1]

if not os.path.exists(CSV_PATH):
    print(f"❌ File not found: {CSV_PATH}")
    sys.exit(1)

# --- Load data ---
df = pd.read_csv(CSV_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['MMSI'] = df['MMSI'].astype(str).str.strip()
df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)

vessels = df['MMSI'].unique()
if len(vessels) < 2:
    print("❌ Need at least 2 vessels in the telemetry file.")
    sys.exit(1)

# Get vessel names
name_map = df.groupby('MMSI')['Name'].first().to_dict() if 'Name' in df.columns else {}
vessel_colors = ['#e74c3c', '#3498db']  # red, blue

# Find collision point — closest pair of pings across the two vessels
mmsi_a, mmsi_b = vessels[0], vessels[1]
df_a = df[df['MMSI'] == mmsi_a].copy()
df_b = df[df['MMSI'] == mmsi_b].copy()

# Collision time = timestamp of the ping closest to the other vessel
# Use the row with the minimum timestamp difference as the impact moment
t_impact = None
min_dist = float('inf')

import math
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

for _, row_a in df_a.iterrows():
    for _, row_b in df_b.iterrows():
        dt = abs((row_a['Timestamp'] - row_b['Timestamp']).total_seconds())
        if dt <= 60:  # only consider pings within 1 minute of each other
            dist = haversine(row_a['Latitude'], row_a['Longitude'],
                             row_b['Latitude'], row_b['Longitude'])
            if dist < min_dist:
                min_dist = dist
                t_impact = row_a['Timestamp']
                collision_lat = (row_a['Latitude'] + row_b['Latitude']) / 2
                collision_lon = (row_a['Longitude'] + row_b['Longitude']) / 2

if t_impact is None:
    # Fallback: midpoint of all pings
    collision_lat = df['Latitude'].mean()
    collision_lon = df['Longitude'].mean()
    t_impact = df['Timestamp'].median()

# --- Build map ---
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='CartoDB positron'
)

# Add vessel trajectories
for i, mmsi in enumerate(vessels):
    vessel_df = df[df['MMSI'] == mmsi].sort_values('Timestamp')
    name = name_map.get(mmsi, mmsi)
    color = vessel_colors[i % len(vessel_colors)]

    coords = list(zip(vessel_df['Latitude'], vessel_df['Longitude']))

    if len(coords) >= 2:
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=name
        ).add_to(m)

    # Add ping markers with timestamps
    for _, ping in vessel_df.iterrows():
        ts_str = ping['Timestamp'].strftime('%H:%M:%S')
        sog = f"{ping['SOG']:.1f} kn" if 'SOG' in ping and pd.notna(ping['SOG']) else ''
        popup_text = f"<b>{name}</b><br>{ts_str}<br>{sog}"

        folium.CircleMarker(
            location=[ping['Latitude'], ping['Longitude']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"{name} @ {ts_str}"
        ).add_to(m)

    # Start marker (triangle/arrow feel via larger circle)
    first = vessel_df.iloc[0]
    folium.Marker(
        location=[first['Latitude'], first['Longitude']],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;color:{color};font-weight:bold;'
                 f'white-space:nowrap;text-shadow:1px 1px 2px white">'
                 f'▶ {name}</div>',
            icon_size=(200, 20),
            icon_anchor=(0, 10)
        )
    ).add_to(m)

# Collision point marker
folium.Marker(
    location=[collision_lat, collision_lon],
    icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon'),
    popup=folium.Popup(
        f"<b>⚠️ COLLISION POINT</b><br>"
        f"{t_impact.strftime('%Y-%m-%d %H:%M:%S')}<br>"
        f"Distance: {min_dist:.1f}m",
        max_width=250
    ),
    tooltip="Collision point"
).add_to(m)

# Legend
name_a = name_map.get(mmsi_a, mmsi_a)
name_b = name_map.get(mmsi_b, mmsi_b)
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
     background:white;padding:12px 16px;border-radius:8px;
     border:1px solid #ccc;font-family:sans-serif;font-size:13px;
     box-shadow:2px 2px 6px rgba(0,0,0,0.2)">
  <b>Vessel Trajectories</b><br>
  <span style="color:{vessel_colors[0]}">&#9644;</span> {name_a}<br>
  <span style="color:{vessel_colors[1]}">&#9644;</span> {name_b}<br>
  <span style="color:red">&#9654;</span> Collision point<br>
  <small>Window: ±{(df['Timestamp'].max() - df['Timestamp'].min()).seconds // 60 // 2} min</small>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Save
base_name = os.path.splitext(os.path.basename(CSV_PATH))[0]
output_file = f"{base_name}_map.html"
m.save(output_file)
print(f"✅ Map saved to: {output_file}")
print(f"   Vessels: {name_a} (red) vs {name_b} (blue)")
print(f"   Collision: {t_impact.strftime('%Y-%m-%d %H:%M:%S')} — {min_dist:.1f}m apart")