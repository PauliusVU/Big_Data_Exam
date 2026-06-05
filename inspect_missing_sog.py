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
OUTPUT_REPORT = "vessels_missing_all_sog.csv"

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM = 50
RADIUS_KM = RADIUS_NM * 1.852

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


def load_and_clean_base_data(zip_path, target_csv):
    print(f"1. Extracting and loading {target_csv}...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open(target_csv) as f:
            df = pd.read_csv(f)

    if '# Timestamp' in df.columns:
        df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df['MMSI'] = df['MMSI'].astype(str).str.strip()
    df = df[(df['MMSI'].str.len() == 9) & (df['MMSI'].str.isnumeric()) &
            (~df['MMSI'].str.startswith('0')) & (~df['MMSI'].isin(DIRTY_MMSI)) &
            (~df['MMSI'].str.startswith(DIRTY_PREFIXES))]

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)]
    df = df[df['Latitude'].between(-90.0, 90.0) & df['Longitude'].between(-180.0, 180.0)]

    # Standardize SOG: convert AIS default '102.3' missing value to true NaN
    df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
    df.loc[df['SOG'] == 102.3, 'SOG'] = np.nan
    
    # Fill missing Names with Unknown for aggregation
    if 'Name' in df.columns:
        df['Name'] = df['Name'].fillna("Unknown")
    else:
        df['Name'] = "Unknown"

    return df


def filter_geography_and_ports(df):
    print("2. Isolating 50nm target zone...")
    df['dist_to_center_km'] = haversine_vectorized(
        df['Latitude'], df['Longitude'], CENTER_LAT, CENTER_LON
    )
    df = df[df['dist_to_center_km'] <= RADIUS_KM]

    # Drop explicitly parked statuses to isolate active tracking
    parked_statuses = ['Moored', 'At anchor', '1', '5']
    df = df[~df['Navigational status'].astype(str).str.strip().isin(parked_statuses)]

    print("3. Loading and applying OpenStreetMap port zones filter...")
    all_polygons = []
    for osm_file in OSM_FILES:
        if not os.path.exists(osm_file):
            continue
        handler = PortHandler()
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        all_polygons.extend(handler.polygons)

    if not all_polygons:
        print("   -> No port files found, returning open sea dataset.")
        return df

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
    
    print(f"   -> Isolated {len(df_filtered):,} active pings outside port boundaries.")
    return df_filtered


def analyze_missing_sog(df):
    print("\n4. Finding vessels with 100% missing SOG data for the entire day...")
    
    # For each vessel, find total active open-sea pings and how many are NaN
    grouped = df.groupby('MMSI').agg(
        Total_Pings=('Timestamp', 'count'),
        Missing_SOG_Pings=('SOG', lambda x: x.isna().sum()),
        Sample_Name=('Name', 'first')
    ).reset_index()

    # Isolate vessels where ALL pings are missing SOG
    silent_vessels = grouped[grouped['Total_Pings'] == grouped['Missing_SOG_Pings']]

    if silent_vessels.empty:
        print("\n📊 DIAGNOSTIC RESULT: 0 vessels have completely missing SOG fields today.")
        print("   -> Every active vessel out at sea has successfully reported its speed.")
        return

    print(f"\n🚨 FOUND {len(silent_vessels)} SILENT VESSEL(S) WITH NO SOG LOGGED ALL DAY:")
    print("-" * 75)
    for _, row in silent_vessels.iterrows():
        print(f"MMSI: {row['MMSI']} | Name: {row['Sample_Name']:<20} | Active Pings: {row['Total_Pings']}")
    print("-" * 75)

    silent_vessels.to_csv(OUTPUT_REPORT, index=False)
    print(f"Saved complete audit report to '{OUTPUT_REPORT}'")


if __name__ == "__main__":
    df_base = load_and_clean_base_data(ZIP_PATH, TARGET_CSV)
    df_sea  = filter_geography_and_ports(df_base)
    analyze_missing_sog(df_sea)