import zipfile
import pandas as pd
import geopandas as gpd
import osmium
import numpy as np
import warnings
import os
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')

# --- CONFIGURATION & CONSTANTS ---
ZIP_PATH = "data/aisdk-2021-12.zip"
TARGET_CSV = "aisdk-2021-12-13.csv"
OUTPUT_CSV = "suspected_collisions_list.csv" # Feeds directly into Phase 2

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM = 50
RADIUS_KM = RADIUS_NM * 1.852
RADIUS_METERS = RADIUS_KM * 1000

DIRTY_MMSI = {"000000000", "111111111", "123456789"}
DIRTY_PREFIXES = ("111",)

OSM_FILES = [
    "data/denmark-latest.osm.pbf",
    "data/sweden-latest.osm.pbf",
]

BBOX = (
    CENTER_LON - 1.5,
    CENTER_LAT - 1.0,
    CENTER_LON + 1.5,
    CENTER_LAT + 1.0
)

PORT_TAGS = {
    'landuse':          {'port', 'harbour', 'industrial'},
    'industrial':       {'port', 'shipyard'},
    'waterway':         {'dock', 'basin'},
    'leisure':          {'marina'},
    'amenity':          {'ferry_terminal'},
    'man_made':         {'pier', 'quay', 'jetty', 'breakwater', 'groyne',
                         'floating_barrier', 'maritime_beacon', 'dolphins'},
    'harbour':          {'yes'},
    'water':            {'harbour', 'dock'},
    'seamark:type':     {'harbour', 'harbour_basin', 'port',
                         'anchorage', 'berth', 'dock',
                         'ferry_route', 'mooring', 'pilot_boarding_place',
                         'small_craft_facilities'},
    'building':         {'warehouse', 'hangar', 'ship_station', 'ferry_terminal'},
    'railway':          {'station'},
    'public_transport': {'stop_position', 'platform'},
}


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calculates the distance between two arrays of coordinates in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


class PortHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.polygons = []
        self.wktfab = osmium.geom.WKTFactory()

    def _is_port_feature(self, tags):
        for key, values in PORT_TAGS.items():
            if key in tags and tags[key] in values:
                return True
        return False

    def _in_bbox(self, geom):
        try:
            from shapely import wkt
            shape = wkt.loads(geom)
            c = shape.centroid
            return (BBOX[0] <= c.x <= BBOX[2] and
                    BBOX[1] <= c.y <= BBOX[3])
        except Exception:
            return False

    def way(self, w):
        if not self._is_port_feature(w.tags):
            return
        try:
            wkt_geom = self.wktfab.create_linestring(w)
            from shapely import wkt
            line = wkt.loads(wkt_geom)
            if line.is_ring:
                poly = Polygon(line)
                if self._in_bbox(poly.wkt):
                    self.polygons.append(poly)
        except Exception:
            pass

    def area(self, a):
        if not self._is_port_feature(a.tags):
            return
        try:
            wkt_geom = self.wktfab.create_multipolygon(a)
            from shapely import wkt
            shape = wkt.loads(wkt_geom)
            if self._in_bbox(shape.wkt):
                self.polygons.append(shape)
        except Exception:
            pass


def load_and_clean_data(zip_path, target_csv):
    print(f"1. Extracting and loading {target_csv} from {zip_path}...")

    with zipfile.ZipFile(zip_path) as z:
        with z.open(target_csv) as f:
            df = pd.read_csv(f)

    if '# Timestamp' in df.columns:
        df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df['MMSI'] = df['MMSI'].astype(str).str.strip()

    # Filter out fixed radio structures/windfarms based on your diagnostic run
    df = df[~df['MMSI'].str.startswith('99')]
    if 'Name' in df.columns:
        df = df[~df['Name'].astype(str).str.upper().str.contains('WINDFARM|PLATFORM', na=False)]

    df = df[(df['MMSI'].str.len() == 9) & (df['MMSI'].str.isnumeric()) &
            (~df['MMSI'].str.startswith('0')) & (~df['MMSI'].isin(DIRTY_MMSI)) &
            (~df['MMSI'].str.startswith(DIRTY_PREFIXES))]

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)]
    df = df[df['Latitude'].between(-90.0, 90.0) & df['Longitude'].between(-180.0, 180.0)]

    # SOG check: Convert 102.3 to NaN, keep <= 60 or missing SOGs
    df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
    df.loc[df['SOG'] == 102.3, 'SOG'] = np.nan
    df = df[(df['SOG'] <= 60) | (df['SOG'].isna())]

    # Chronologically sort for forward-fill
    df = df.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)
    df['SOG'] = df.groupby('MMSI')['SOG'].ffill().bfill().fillna(0.0)

    # Size Dimensions: Broadcast known static dimensions across asynchronous rows.
    # NO ARBITRARY FILLNA VALUES HERE: Missing dimensions are left as NaN.
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df['Width'] = pd.to_numeric(df['Width'], errors='coerce')
    df['Length'] = df.groupby('MMSI')['Length'].transform('max')
    df['Width'] = df.groupby('MMSI')['Width'].transform('max')
    df['Area'] = df['Length'] * df['Width']

    if 'Name' in df.columns:
        df['Name'] = df.groupby('MMSI')['Name'].transform('first').fillna("Unknown")
    else:
        df['Name'] = "Unknown"

    print(f"   -> Rows after data integrity cleaning: {len(df):,}")
    return df


def apply_spatial_and_state_filters(df):
    print("\n2. Applying Geographic & Vessel State filters...")
    df['dist_to_center_km'] = haversine_vectorized(
        df['Latitude'], df['Longitude'], CENTER_LAT, CENTER_LON
    )
    df = df[df['dist_to_center_km'] <= RADIUS_KM]

    parked_statuses = ['Moored', 'At anchor', '1', '5']
    df = df[~df['Navigational status'].astype(str).str.strip().isin(parked_statuses)]
    return df


def remove_port_zones(df):
    print("\n3. Loading port boundaries from local OSM PBF files...")

    all_polygons = []
    for osm_file in OSM_FILES:
        if not os.path.exists(osm_file):
            print(f"   -> WARNING: {osm_file} not found, skipping.")
            continue
        print(f"   -> Reading {osm_file}...")
        handler = PortHandler()
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        print(f"      -> Found {len(handler.polygons)} raw port polygons.")
        all_polygons.extend(handler.polygons)

    if not all_polygons:
        print("   -> Warning: No port features found, skipping port filter.")
        return df

    print(f"   -> Merging {len(all_polygons)} features and applying 600m buffer...")
    ports_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs="EPSG:4326")
    
    ports_gdf['geometry'] = ports_gdf.geometry.buffer(0)
    ports_gdf = ports_gdf.dissolve()
    
    ports_gdf = ports_gdf.to_crs("EPSG:3857")
    ports_gdf['geometry'] = ports_gdf.buffer(600)
    ports_gdf = ports_gdf.to_crs("EPSG:4326")

    gdf_ais = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )

    ais_in_ports = gpd.sjoin(gdf_ais, ports_gdf, how='inner', predicate='intersects')
    df_filtered = df[~df.index.isin(ais_in_ports.index)]

    print(f"   -> Dropped {len(df) - len(df_filtered)} AIS points inside port boundaries.")
    return df_filtered


def compute_rolling_max_speed(df):
    """Compute per-vessel 5-minute rolling max SOG with a proper datetime index."""
    def rolling_max(group):
        group = group.sort_values('Timestamp').set_index('Timestamp')
        group['max_speed_5m'] = group['SOG'].rolling('5min').max()
        return group.reset_index()

    return df.groupby('MMSI', group_keys=False).apply(rolling_max)


def detect_collision(df):
    print("\n4. Computing rolling speed baseline per vessel...")
    df = compute_rolling_max_speed(df)

    df['Time_Bucket'] = df['Timestamp'].dt.floor('min')
    df['Lat_Bucket']  = df['Latitude'].round(2)
    df['Lon_Bucket']  = df['Longitude'].round(2)

    print("\n5. Scanning for encounters using spatial bucketing with neighbor cells...")
    offsets = [-0.01, 0.0, 0.01]
    neighbor_pairs = [(dlat, dlon) for dlat in offsets for dlon in offsets]

    results = []
    for dlat, dlon in neighbor_pairs:
        temp = df.copy()
        temp['Lat_Bucket'] = (temp['Latitude'].round(2) + dlat).round(2)
        temp['Lon_Bucket'] = (temp['Longitude'].round(2) + dlon).round(2)
        merged = pd.merge(
            df, temp,
            on=['Time_Bucket', 'Lat_Bucket', 'Lon_Bucket'],
            suffixes=('_A', '_B')
        )
        merged = merged[merged['MMSI_A'] < merged['MMSI_B']]
        results.append(merged)

    merged = pd.concat(results).drop_duplicates(
        subset=['MMSI_A', 'MMSI_B', 'Timestamp_A', 'Timestamp_B']
    )

    merged['Ship_Distance_km'] = haversine_vectorized(
        merged['Latitude_A'], merged['Longitude_A'],
        merged['Latitude_B'], merged['Longitude_B']
    )

    collisions = merged[merged['Ship_Distance_km'] < 0.050].copy()

    if collisions.empty:
        print("   -> No pairs came within 50 meters.")
        return None

    print(f"   -> Found {len(collisions)} ultra-close pings.")

    # The ranking math handles missing values natively here without altering the data at rest.
    # If Area is NaN, it defaults the area factor inline to 1.0 so speed and distance still rank it.
    ke_a = collisions['Area_A'].fillna(1.0) * (collisions['max_speed_5m_A'] ** 2)
    ke_b = collisions['Area_B'].fillna(1.0) * (collisions['max_speed_5m_B'] ** 2)
    
    collisions['Total_Impact_Energy'] = ke_a + ke_b
    collisions['Distance_Meters'] = collisions['Ship_Distance_km'] * 1000
    collisions['Danger_Score'] = collisions['Total_Impact_Energy'] / (
        (collisions['Distance_Meters'] + 1) ** 2
    )

    collisions = collisions.sort_values(by='Danger_Score', ascending=False)
    collisions = collisions.drop_duplicates(subset=['MMSI_A', 'MMSI_B'], keep='first')

    export_df = collisions[[
        'Timestamp_A', 'MMSI_A', 'MMSI_B', 'Name_A', 'Name_B',
        'Distance_Meters', 'max_speed_5m_A', 'max_speed_5m_B',
        'Total_Impact_Energy', 'Danger_Score'
    ]].copy()

    export_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ PHASE 1 COMPLETE: Exported {len(export_df)} suspect pair(s) to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    df_clean        = load_and_clean_data(ZIP_PATH, TARGET_CSV)
    df_filtered_geo = apply_spatial_and_state_filters(df_clean)
    df_final        = remove_port_zones(df_filtered_geo)
    detect_collision(df_final)
