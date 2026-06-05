import os
import math
import pandas as pd
import folium

# Output directory: set via OUTPUT_DIR env variable in Docker, defaults to '.' locally
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_RESULTS_CSV   = os.path.join(OUTPUT_DIR, "FINAL_investigation_results.csv")
CLEAN_TELEMETRY_CSV = os.path.join(OUTPUT_DIR, "clean_suspect_telemetry.csv")
VIZ_WINDOW_MINUTES  = 10  # ±10 minutes as per assignment spec

VESSEL_COLORS = ['#e74c3c', '#3498db']  # red, blue


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_collision_point(df_a, df_b, t_impact):
    # Finds the closest pair of pings within ±60s of impact to place the marker.
    window = pd.Timedelta(seconds=60)
    near_a = df_a[df_a['Timestamp'].between(t_impact - window, t_impact + window)]
    near_b = df_b[df_b['Timestamp'].between(t_impact - window, t_impact + window)]

    min_dist = float('inf')
    col_lat  = (df_a['Latitude'].mean() + df_b['Latitude'].mean()) / 2
    col_lon  = (df_a['Longitude'].mean() + df_b['Longitude'].mean()) / 2

    for _, ra in near_a.iterrows():
        for _, rb in near_b.iterrows():
            dt = abs((ra['Timestamp'] - rb['Timestamp']).total_seconds())
            if dt <= 60:
                d = haversine_m(ra['Latitude'], ra['Longitude'],
                                rb['Latitude'], rb['Longitude'])
                if d < min_dist:
                    min_dist = d
                    col_lat  = (ra['Latitude'] + rb['Latitude']) / 2
                    col_lon  = (ra['Longitude'] + rb['Longitude']) / 2

    return col_lat, col_lon, min_dist


def build_map(name_a, name_b, df_a, df_b, t_impact, col_lat, col_lon, min_dist):
    center_lat = (df_a['Latitude'].mean() + df_b['Latitude'].mean()) / 2
    center_lon = (df_a['Longitude'].mean() + df_b['Longitude'].mean()) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

    for i, (name, vessel_df) in enumerate([(name_a, df_a), (name_b, df_b)]):
        color = VESSEL_COLORS[i % len(VESSEL_COLORS)]
        vessel_df = vessel_df.sort_values('Timestamp')
        coords = list(zip(vessel_df['Latitude'], vessel_df['Longitude']))

        if len(coords) >= 2:
            folium.PolyLine(locations=coords, color=color, weight=3,
                            opacity=0.8, tooltip=name).add_to(m)

        for _, ping in vessel_df.iterrows():
            ts_str     = ping['Timestamp'].strftime('%H:%M:%S')
            sog_str    = f"{ping['SOG']:.1f} kn" if pd.notna(ping.get('SOG')) else ''
            is_pre     = ping['Timestamp'] <= t_impact
            radius     = 5 if abs((ping['Timestamp'] - t_impact).total_seconds()) <= 60 else 3
            popup_html = f"<b>{name}</b><br>{ts_str}<br>{sog_str}"

            folium.CircleMarker(
                location=[ping['Latitude'], ping['Longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8 if is_pre else 0.4,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{name} @ {ts_str}"
            ).add_to(m)

        first = vessel_df.iloc[0]
        folium.Marker(
            location=[first['Latitude'], first['Longitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:{color};font-weight:bold;'
                     f'white-space:nowrap;text-shadow:1px 1px 2px white">'
                     f'> {name}</div>',
                icon_size=(200, 20),
                icon_anchor=(0, 10)
            )
        ).add_to(m)

    folium.Marker(
        location=[col_lat, col_lon],
        icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon'),
        popup=folium.Popup(
            f"<b>COLLISION</b><br>{t_impact.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"Distance: {min_dist:.1f}m",
            max_width=250
        ),
        tooltip="Collision point"
    ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:12px 16px;border-radius:8px;
         border:1px solid #ccc;font-family:sans-serif;font-size:13px;
         box-shadow:2px 2px 6px rgba(0,0,0,0.2)">
      <b>Vessel Trajectories</b><br>
      <span style="color:{VESSEL_COLORS[0]}">&#9644;</span> {name_a}<br>
      <span style="color:{VESSEL_COLORS[1]}">&#9644;</span> {name_b}<br>
      <span style="color:red">&#9654;</span> Collision point<br>
      <small>Window: +/-{VIZ_WINDOW_MINUTES} minutes</small><br>
      <small>{t_impact.strftime('%Y-%m-%d %H:%M:%S')} UTC</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def run_visualization():
    print("=" * 65)
    print("  AIS COLLISION DETECTION — VISUALIZATION")
    print("=" * 65)

    if not os.path.exists(FINAL_RESULTS_CSV):
        print(f"  Error: {FINAL_RESULTS_CSV} not found. Run Phase 1 and Phase 2 first.")
        return

    if not os.path.exists(CLEAN_TELEMETRY_CSV) and not os.path.isdir(CLEAN_TELEMETRY_CSV):
        print(f"  Error: {CLEAN_TELEMETRY_CSV} not found. Run Phase 1 first.")
        return

    results = pd.read_csv(FINAL_RESULTS_CSV)
    results['Timestamp_A'] = pd.to_datetime(results['Timestamp_A'])

    if results.empty:
        print("  No confirmed collisions to visualize.")
        return

    print(f"\n  Loading telemetry from {CLEAN_TELEMETRY_CSV}...")
    import glob as _glob
    _parts = _glob.glob(os.path.join(CLEAN_TELEMETRY_CSV, 'part-*.csv'))
    telemetry = pd.read_csv(_parts[0] if _parts else CLEAN_TELEMETRY_CSV)
    telemetry['Timestamp'] = pd.to_datetime(telemetry['Timestamp']).apply(lambda x: x.replace(tzinfo=None))
    telemetry['MMSI'] = telemetry['MMSI'].astype(str).str.strip()

    vessel_index = {
        mmsi: group.reset_index(drop=True)
        for mmsi, group in telemetry.groupby('MMSI')
    }

    print(f"  Generating {len(results)} map(s)...\n")

    for _, row in results.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a   = str(row['MMSI_A'])
        mmsi_b   = str(row['MMSI_B'])
        name_a   = row['Name_A']
        name_b   = row['Name_B']

        start = t_impact - pd.Timedelta(minutes=VIZ_WINDOW_MINUTES)
        end   = t_impact + pd.Timedelta(minutes=VIZ_WINDOW_MINUTES)

        full_a = vessel_index.get(mmsi_a, pd.DataFrame())
        full_b = vessel_index.get(mmsi_b, pd.DataFrame())

        if full_a.empty or full_b.empty:
            print(f"  Skipping {name_a} vs {name_b} — telemetry missing")
            continue

        df_a = full_a[full_a['Timestamp'].between(start, end)].copy()
        df_b = full_b[full_b['Timestamp'].between(start, end)].copy()

        if df_a.empty or df_b.empty:
            print(f"  Skipping {name_a} vs {name_b} — no pings in window")
            continue

        col_lat, col_lon, min_dist = find_collision_point(df_a, df_b, t_impact)
        m = build_map(name_a, name_b, df_a, df_b, t_impact, col_lat, col_lon, min_dist)

        safe_a   = name_a.replace(' ', '_').replace('/', '')
        safe_b   = name_b.replace(' ', '_').replace('/', '')
        out_file = os.path.join(OUTPUT_DIR, f"map_{safe_a}_vs_{safe_b}.html")
        m.save(out_file)

        print(f"  {name_a} vs {name_b}")
        print(f"    {t_impact.strftime('%Y-%m-%d %H:%M:%S')} — {min_dist:.1f}m apart")
        print(f"    {out_file}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_visualization()