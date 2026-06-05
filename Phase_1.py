import zipfile
import pandas as pd
import geopandas as gpd
import osmium
import numpy as np
import pygeohash as gh
import warnings
import os
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')

# --- CONFIGURATION & CONSTANTS ---
ZIP_PATH = "data/aisdk-2021-12.zip"
TARGET_CSV = "aisdk-2021-12-13.csv"
OUTPUT_CSV = "suspected_collisions_list.csv"
CLEAN_TELEMETRY_CSV = "clean_suspect_telemetry.csv"  # Surgical export for Phase 2

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM = 50
RADIUS_KM = RADIUS_NM * 1.852

# FIX 4b: Expanded from 50m to 100m to account for GPS reporting error.
# Phase 2's kinematic intersection check acts as the precision filter.
COLLISION_THRESHOLD_KM = 0.100

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
            return (BBOX[0] <= c.x <= BBOX[2] and BBOX[1] <= c.y <= BBOX[3])
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

    df = df[~df['MMSI'].str.startswith('99')]
    if 'Name' in df.columns:
        df = df[~df['Name'].astype(str).str.upper().str.contains('WINDFARM|PLATFORM', na=False)]

    df = df[(df['MMSI'].str.len() == 9) & (df['MMSI'].str.isnumeric()) &
            (~df['MMSI'].str.startswith('0')) & (~df['MMSI'].isin(DIRTY_MMSI)) &
            (~df['MMSI'].str.startswith(DIRTY_PREFIXES))]

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)]
    df = df[df['Latitude'].between(-90.0, 90.0) & df['Longitude'].between(-180.0, 180.0)]

    df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
    df.loc[df['SOG'] == 102.3, 'SOG'] = np.nan
    df = df[(df['SOG'] <= 60) | (df['SOG'].isna())]

    df = df.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # FIX 3: Inter-ping GPS anomaly filter.
    # For each vessel, compute the implied speed between consecutive pings.
    # Any ping that implies a speed above 50 knots is physically impossible
    # and is a GPS jump — drop it. This is O(n) via shift(), not a join.
    # -------------------------------------------------------------------------
    prev_lat  = df.groupby('MMSI')['Latitude'].shift(1)
    prev_lon  = df.groupby('MMSI')['Longitude'].shift(1)
    prev_time = df.groupby('MMSI')['Timestamp'].shift(1)

    dt_hours = (df['Timestamp'] - prev_time).dt.total_seconds() / 3600
    dist_km  = haversine_vectorized(df['Latitude'], df['Longitude'], prev_lat, prev_lon)
    implied_knots = (dist_km / dt_hours.replace(0, np.nan)) / 1.852

    before = len(df)
    # NaN implied_knots means first ping of a vessel — always keep those
    df = df[implied_knots.isna() | (implied_knots < 50)]
    print(f"   -> Dropped {before - len(df):,} GPS anomaly pings (implied speed > 50 knots)")
    # -------------------------------------------------------------------------

    df['SOG'] = df.groupby('MMSI')['SOG'].ffill().bfill().fillna(0.0)

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
        all_polygons.extend(handler.polygons)

    if not all_polygons:
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


def _get_geohash_neighborhood(geohash):
    """Returns the given geohash cell plus its 8 neighbours — a 3x3 grid."""
    n  = gh.get_adjacent(geohash, 'top')
    s  = gh.get_adjacent(geohash, 'bottom')
    e  = gh.get_adjacent(geohash, 'right')
    w  = gh.get_adjacent(geohash, 'left')
    ne = gh.get_adjacent(n, 'right')
    nw = gh.get_adjacent(n, 'left')
    se = gh.get_adjacent(s, 'right')
    sw = gh.get_adjacent(s, 'left')
    return [geohash, n, s, e, w, ne, nw, se, sw]


def detect_collision(df, df_clean):
    # -------------------------------------------------------------------------
    # FIX 7: Replace 9-pass offset bucketing with a geohash neighbourhood join.
    #
    # Old approach: 9 separate merges on shifted lat/lon buckets.
    # New approach: assign each ping a precision-7 geohash (~170m x 214m cell).
    #   Each ping is then expanded to cover itself AND its 8 neighbours, giving
    #   a single lookup key per candidate row. One merge on that key finds all
    #   pairs that share a cell or are in adjacent cells, guaranteed to catch
    #   any two pings within 100m of each other.
    #
    #   In PySpark this becomes a single join on the geohash column after an
    #   explode() of the neighbourhood list — one shuffle instead of nine.
    # -------------------------------------------------------------------------
    print("\n4. Assigning geohash keys (precision 7) for proximity detection...")

    df['Time_Bucket'] = df['Timestamp'].dt.floor('min')

    # Precision 7 cell is ~170m x 214m. A 3x3 neighbourhood spans ~510m x 642m,
    # which is safely larger than our 100m collision threshold.
    # List comprehension over zip() is significantly faster than df.apply() and
    # maps cleanly to a PySpark UDF when porting.
    df['Geohash'] = [
        gh.encode(lat, lon, precision=7)
        for lat, lon in zip(df['Latitude'], df['Longitude'])
    ]

    # Expand each ping to its full 9-cell neighbourhood so that a single merge
    # catches vessels straddling cell boundaries.
    print("   -> Expanding to 9-cell neighbourhoods...")
    df_expanded = df.copy()
    df_expanded['Geohash'] = df_expanded['Geohash'].apply(_get_geohash_neighborhood)
    df_expanded = df_expanded.explode('Geohash').reset_index(drop=True)

    print("   -> Running single geohash join to find candidate pairs...")
    merged = pd.merge(
        df,           # left side: each ping in its own cell
        df_expanded,  # right side: each ping expanded to 9 cells
        on=['Time_Bucket', 'Geohash'],
        suffixes=('_A', '_B')
    )
    merged = merged[merged['MMSI_A'] < merged['MMSI_B']]
    merged = merged.drop_duplicates(subset=['MMSI_A', 'MMSI_B', 'Timestamp_A', 'Timestamp_B'])

    # Precise haversine distance — geohash join is a broad net, this trims to 100m
    merged['Ship_Distance_km'] = haversine_vectorized(
        merged['Latitude_A'], merged['Longitude_A'],
        merged['Latitude_B'], merged['Longitude_B']
    )

    collisions = merged[merged['Ship_Distance_km'] < COLLISION_THRESHOLD_KM].copy()

    if collisions.empty:
        print(f"   -> No pairs came within {int(COLLISION_THRESHOLD_KM * 1000)} meters.")
        return

    collisions['Distance_Meters'] = collisions['Ship_Distance_km'] * 1000
    collisions = collisions.sort_values(by='Distance_Meters', ascending=True)
    collisions = collisions.drop_duplicates(subset=['MMSI_A', 'MMSI_B'], keep='first')

    export_df = collisions[[
        'Timestamp_A', 'MMSI_A', 'MMSI_B', 'Name_A', 'Name_B', 'Distance_Meters'
    ]].copy()

    export_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ PHASE 1 COMPLETE: Exported {len(export_df)} suspect pair(s) to '{OUTPUT_CSV}'")

    # -------------------------------------------------------------------------
    # FIX 4: Surgical telemetry export now sourced from df_clean (the fully
    # cleaned but NOT geo-filtered dataframe). This ensures Phase 2 has access
    # to pings from before a vessel entered the 50nm zone, giving the trauma
    # analysis a complete pre-collision baseline and an untruncated trajectory.
    # -------------------------------------------------------------------------
    print(f"\n5. Generating surgical clean data export for Phase 2...")
    suspect_mmsis = set(collisions['MMSI_A']).union(set(collisions['MMSI_B']))
    suspect_telemetry_df = df_clean[df_clean['MMSI'].isin(suspect_mmsis)]
    suspect_telemetry_df.to_csv(CLEAN_TELEMETRY_CSV, index=False)
    print(f"   -> Saved full-day telemetry for {len(suspect_mmsis)} vessels to '{CLEAN_TELEMETRY_CSV}'")


if __name__ == "__main__":
    df_clean        = load_and_clean_data(ZIP_PATH, TARGET_CSV)
    df_filtered_geo = apply_spatial_and_state_filters(df_clean)
    df_final        = remove_port_zones(df_filtered_geo)
    detect_collision(df_final, df_clean)