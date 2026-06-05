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
OUTPUT_CSV = "sog_imputation_post_filter_audit.csv"

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


def load_base_data(zip_path, target_csv):
    print(f"1. Extracting and loading {target_csv} from {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open(target_csv) as f:
            df = pd.read_csv(f)

    if '# Timestamp' in df.columns:
        df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df['MMSI'] = df['MMSI'].astype(str).str.strip()

    # Exclude known radio structures/windfarms from the profile
    df = df[~df['MMSI'].str.startswith('99')]
    if 'Name' in df.columns:
        df = df[~df['Name'].astype(str).str.upper().str.contains('WINDFARM|PLATFORM', na=False)]

    df = df[(df['MMSI'].str.len() == 9) & (df['MMSI'].str.isnumeric()) &
            (~df['MMSI'].str.startswith('0')) & (~df['MMSI'].isin(DIRTY_MMSI)) &
            (~df['MMSI'].str.startswith(DIRTY_PREFIXES))]

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df[(df['Latitude'] != 0.0) & (df['Longitude'] != 0.0)]
    df = df[df['Latitude'].between(-90.0, 90.0) & df['Longitude'].between(-180.0, 180.0)]

    # Standardize SOG fields into pure NaNs, dropping clear anomalies (>60)
    df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
    df.loc[df['SOG'] == 102.3, 'SOG'] = np.nan
    df = df[(df['SOG'] <= 60) | (df['SOG'].isna())]

    return df


def apply_spatial_and_state_filters(df):
    print("2. Filtering out spatial data outside 50nm and explicitly parked statuses...")
    df['dist_to_center_km'] = haversine_vectorized(
        df['Latitude'], df['Longitude'], CENTER_LAT, CENTER_LON
    )
    df = df[df['dist_to_center_km'] <= RADIUS_KM]

    parked_statuses = ['Moored', 'At anchor', '1', '5']
    df = df[~df['Navigational status'].astype(str).str.strip().isin(parked_statuses)]
    return df


def remove_port_zones(df):
    print("3. Loading and applying OpenStreetMap harbor boundaries filter...")
    all_polygons = []
    for osm_file in OSM_FILES:
        if not os.path.exists(osm_file):
            continue
        handler = PortHandler()
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        all_polygons.extend(handler.polygons)

    if not all_polygons:
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
    return df_filtered


def run_post_filter_audit():
    # Phase A: Ingest raw structural frames
    df_base = load_base_data(ZIP_PATH, TARGET_CSV)
    
    # Phase B: Apply all spatial/state exclusions BEFORE running the imputation analysis
    df_geo = apply_spatial_and_state_filters(df_base)
    df = remove_port_zones(df_geo)
    
    print(f"   -> Pure Open-Sea Active Rows Remaining to Audit: {len(df):,}")
    
    # Chronologically sort the remaining records to isolate accurate timeline steps
    df = df.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)

    print("\n4. Analyzing open-sea SOG missingness patterns...")
    is_initially_nan = df['SOG'].isna()

    # Step A: Forward-Fill evaluation
    df['SOG_after_ffill'] = df.groupby('MMSI')['SOG'].ffill()
    is_nan_after_ffill = df['SOG_after_ffill'].isna()
    filled_by_ffill = is_initially_nan & ~is_nan_after_ffill

    # Step B: Backward-Fill evaluation
    df['SOG_after_bfill'] = df.groupby('MMSI')['SOG_after_ffill'].bfill()
    is_nan_after_bfill = df['SOG_after_bfill'].isna()
    filled_by_bfill = is_nan_after_ffill & ~is_nan_after_bfill

    # Step C: Hard Zero Fallback evaluation
    df['SOG_final'] = df['SOG_after_bfill'].fillna(0.0)
    filled_by_fallback = is_nan_after_bfill

    # Aggregate Metrics
    total_rows = len(df)
    total_valid_initially = (~is_initially_nan).sum()
    total_missing_initially = is_initially_nan.sum()
    
    count_ffill = filled_by_ffill.sum()
    count_bfill = filled_by_bfill.sum()
    count_fallback = filled_by_fallback.sum()

    print("\n" + "="*80)
    print("📊 OPEN-SEA SOG IMPUTATION AUDIT DASHBOARD (POST-FILTER)")
    print("="*80)
    print(f"Total Active Open-Sea Rows:          {total_rows:,}")
    print(f"Initially Valid SOG Records:        {total_valid_initially:,} ({ (total_valid_initially/total_rows)*100 :.3f}%)")
    print(f"Initially Missing SOG Records:      {total_missing_initially:,} ({ (total_missing_initially/total_rows)*100 :.3f}%)")
    print("-"*80)
    print("IMPUTATION STAIRCASE BREAKDOWN FOR ACTIVE GAPS:")
    if total_missing_initially > 0:
        print(f"  ➔ Recovered via Forward-Fill (ffill):     {count_ffill:,} ({ (count_ffill/total_missing_initially)*100 :.2f}% of missing)")
        print(f"  ➔ Recovered via Backward-Fill (bfill):    {count_bfill:,} ({ (count_bfill/total_missing_initially)*100 :.2f}% of missing)")
        print(f"  ➔ Defaulted via Hard 0.0 Fallback:        {count_fallback:,} ({ (count_fallback/total_missing_initially)*100 :.2f}% of missing)")
    else:
        print("  0 active missing gaps detected on the open sea.")
    print("="*80)

    # Export Per-Vessel Profile Breakdown
    df['Filled_By_FFill'] = filled_by_ffill.astype(int)
    df['Filled_By_BFill'] = filled_by_bfill.astype(int)
    df['Filled_By_Fallback'] = filled_by_fallback.astype(int)
    df['Initially_Missing'] = is_initially_nan.astype(int)

    vessel_report = df.groupby('MMSI').agg(
        Vessel_Name=('Name', 'first'),
        Total_Pings=('Timestamp', 'count'),
        Valid_On_Arrival=('SOG', lambda x: x.notna().sum()),
        Total_Missing_Gaps=('Initially_Missing', 'sum'),
        Resolved_By_FFill=('Filled_By_FFill', 'sum'),
        Resolved_By_BFill=('Filled_By_BFill', 'sum'),
        Resolved_By_Fallback=('Filled_By_Fallback', 'sum')
    ).reset_index()

    vessel_report = vessel_report.sort_values(by='Total_Missing_Gaps', ascending=False)
    vessel_report.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Post-filtered open sea audit report saved to '{OUTPUT_CSV}'")
    print("\nTop 5 Active Open-Sea Vessels with Intermittent SOG Gaps:")
    print(vessel_report[['MMSI', 'Vessel_Name', 'Total_Pings', 'Total_Missing_Gaps', 'Resolved_By_FFill']].head(5).to_string(index=False))


if __name__ == "__main__":
    run_post_filter_audit()