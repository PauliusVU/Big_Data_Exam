import os
os.environ['PYSPARK_PYTHON'] = '/opt/anaconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/anaconda3/bin/python'

import math
import time
import zipfile
import tempfile
import warnings
from datetime import timedelta

import pandas as pd
import geopandas as gpd
import osmium
import pygeohash as gh
from shapely.geometry import Polygon, Point

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, DoubleType, BooleanType

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
ZIP_PATH              = "data/aisdk-2021-12.zip"
OUTPUT_CSV            = "suspected_collisions_list.csv"
CLEAN_TELEMETRY_CSV   = "clean_suspect_telemetry.csv"

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM             = 50
RADIUS_KM             = RADIUS_NM * 1.852
COLLISION_THRESHOLD_KM = 0.100
DIRTY_MMSI = {"000000000", "111111111", "123456789"}

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


# =============================================================================
# PROGRESS REPORTING
# =============================================================================

PIPELINE_START = None
STEP_START     = None
TOTAL_STEPS    = 5


def pipeline_start():
    global PIPELINE_START
    PIPELINE_START = time.time()
    print("=" * 65)
    print("  AIS COLLISION DETECTION — FULL 31-DAY PIPELINE")
    print("=" * 65)


def step_start(step_num, description):
    global STEP_START
    STEP_START = time.time()
    elapsed = timedelta(seconds=int(time.time() - PIPELINE_START))
    print(f"\n[{step_num}/{TOTAL_STEPS}] {description}")
    print(f"         Elapsed so far: {elapsed}")
    print(f"         " + "-" * 50)


def step_done(detail=""):
    elapsed_step  = timedelta(seconds=int(time.time() - STEP_START))
    elapsed_total = timedelta(seconds=int(time.time() - PIPELINE_START))
    suffix = f" — {detail}" if detail else ""
    print(f"         ✓ Done in {elapsed_step}{suffix}")
    print(f"         Total elapsed: {elapsed_total}")


def pipeline_done():
    total = timedelta(seconds=int(time.time() - PIPELINE_START))
    print("\n" + "=" * 65)
    print(f"  PIPELINE COMPLETE — Total runtime: {total}")
    print("=" * 65)


# =============================================================================
# STEP 0: BUILD PORT EXCLUSION GEOHASH SET (runs once on driver)
# -----------------------------------------------------------------------------
# Instead of broadcasting a complex Shapely geometry to Spark workers and
# running a Python UDF per row, we precompute the set of all precision-7
# geohash cells whose centre point falls inside any port zone.
#
# The result is a plain Python frozenset of strings. Spark filters with a
# native JVM isin() hash lookup — no Python UDF, no geometry deserialization,
# no per-row geometry math during the Spark stage.
#
# Precision-7 cells are ~170m x 214m. A 700m buffer on port features ensures
# any cell overlapping a port zone is comfortably within the exclusion area.
# Worst-case over-exclusion: ~170m beyond port boundary — acceptable given
# the 700m buffer already applied.
# =============================================================================

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


def build_port_geohash_set():
    """
    Parses OSM port features, applies a 700m buffer, then enumerates all
    precision-7 geohash cells whose centre point falls inside any port zone.
    Returns a frozenset of geohash strings used as a Spark isin() filter.
    """
    step_start(0, "Building port exclusion geohash set from OSM data (driver-side)...")
    all_polygons = []
    for osm_file in OSM_FILES:
        if not os.path.exists(osm_file):
            print(f"         WARNING: {osm_file} not found, skipping.")
            continue
        print(f"         Parsing {osm_file}...")
        handler = PortHandler()
        handler.apply_file(osm_file, locations=True, idx='flex_mem')
        all_polygons.extend(handler.polygons)

    if not all_polygons:
        print("         No port data found. Port filter will be skipped.")
        step_done("skipped")
        return None

    print(f"         Merging {len(all_polygons)} features and applying 700m buffer...")
    ports_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs="EPSG:4326")
    ports_gdf['geometry'] = ports_gdf.geometry.buffer(0)
    ports_gdf = ports_gdf.dissolve()
    ports_gdf = ports_gdf.to_crs("EPSG:3857")
    ports_gdf['geometry'] = ports_gdf.buffer(700)
    ports_gdf = ports_gdf.to_crs("EPSG:4326")
    merged_geom = ports_gdf.geometry.iloc[0]

    print("         Enumerating precision-7 geohash cells over port zones...")
    port_cells = set()
    lat_step = 0.0015
    lon_step = 0.0030
    lat = BBOX[1]
    while lat <= BBOX[3]:
        lon = BBOX[0]
        while lon <= BBOX[2]:
            if merged_geom.contains(Point(lon, lat)):
                port_cells.add(gh.encode(lat, lon, precision=7))
            lon += lon_step
        lat += lat_step

    port_cells = frozenset(port_cells)
    print(f"         Found {len(port_cells)} precision-7 cells inside port zones.")
    step_done(f"port geohash set ready ({len(port_cells)} cells)")
    return port_cells


# =============================================================================
# SPARK UDFs
# =============================================================================

@F.udf(ArrayType(StringType()))
def geohash_neighborhood_udf(lat, lon):
    """
    Returns a precision-7 geohash for (lat, lon) plus its 8 neighbours.
    Precision-7 cell is ~170m x 214m; the 3x3 neighbourhood spans ~510m x 642m,
    safely larger than the 100m collision threshold.
    explode() on this list enables a single join to replace 9 shuffle passes.
    """
    if lat is None or lon is None:
        return []
    center = gh.encode(lat, lon, precision=7)
    n  = gh.get_adjacent(center, 'top')
    s  = gh.get_adjacent(center, 'bottom')
    e  = gh.get_adjacent(center, 'right')
    w  = gh.get_adjacent(center, 'left')
    ne = gh.get_adjacent(n, 'right')
    nw = gh.get_adjacent(n, 'left')
    se = gh.get_adjacent(s, 'right')
    sw = gh.get_adjacent(s, 'left')
    return [center, n, s, e, w, ne, nw, se, sw]


@F.udf(StringType())
def geohash_center_udf(lat, lon):
    """Returns just the precision-7 geohash — used on the left side of the join
    and reused for port filtering to avoid a separate encoding pass."""
    if lat is None or lon is None:
        return None
    return gh.encode(lat, lon, precision=7)


@F.udf(DoubleType())
def haversine_udf(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometres between two (lat, lon) pairs."""
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# =============================================================================
# STEP 1: EXTRACT ZIP & CREATE SPARK SESSION
# =============================================================================

def extract_zip(zip_path, extract_dir):
    print(f"         Extracting CSVs from {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    print(f"         Extracted {len(csv_files)} daily CSV files.")
    return extract_dir


def create_spark_session():
    cores = max(1, os.cpu_count() - 1)
    print(f"         Starting Spark session (local[{cores}], 8g driver memory)...")
    spark = (
        SparkSession.builder
        .appName("AIS_Collision_Detection")
        .master(f"local[{cores}]")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"         Spark session ready on {cores} cores.")
    return spark


# =============================================================================
# STEP 2: CLEAN ONE DAY (called per-file)
# -----------------------------------------------------------------------------
# Each daily CSV is processed independently so that window functions
# (lag, first, last) operate within each day's partitions only —
# avoiding the expensive global MMSI shuffle that would be needed if all
# 31 days were loaded and processed together.
#
# The Geohash column is computed here and reused for both port filtering
# and collision detection — no redundant encoding pass.
# =============================================================================

def clean_one_day(spark, csv_path, port_cells):
    """
    Reads and cleans a single daily CSV. Geohash column is computed once
    and reused for port filtering (isin) and collision detection (join).
    """
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("quote", '"')
        .option("escape", '"')
        .csv(csv_path)
    )

    if '# Timestamp' in df.columns:
        df = df.withColumnRenamed('# Timestamp', 'Timestamp')

    df = (
        df
        .withColumn('Timestamp', F.to_timestamp('Timestamp', 'dd/MM/yyyy HH:mm:ss'))
        .withColumn('Latitude',  F.col('Latitude').cast(DoubleType()))
        .withColumn('Longitude', F.col('Longitude').cast(DoubleType()))
        .withColumn('SOG',       F.col('SOG').cast(DoubleType()))
        .withColumn('MMSI',      F.trim(F.col('MMSI').cast(StringType())))
    )

    # --- FILTER 1: Timestamps ---
    df = df.filter(F.col('Timestamp').isNotNull())

    # --- FILTER 2: MMSI integrity (cheapest, applied first) ---
    dirty_mmsi_list = list(DIRTY_MMSI)
    df = (
        df
        .filter(F.length('MMSI') == 9)
        .filter(F.col('MMSI').rlike('^[0-9]+$'))
        .filter(~F.col('MMSI').startswith('0'))
        .filter(~F.col('MMSI').startswith('99'))
        .filter(~F.col('MMSI').isin(dirty_mmsi_list))
        .filter(~F.col('MMSI').startswith('111'))
    )

    # --- FILTER 3: Wind farms / platforms ---
    if 'Name' in df.columns:
        df = df.filter(~F.upper(F.col('Name')).rlike('WINDFARM|PLATFORM'))

    # --- FILTER 4: Coordinate sanity ---
    df = (
        df
        .filter(F.col('Latitude').isNotNull() & F.col('Longitude').isNotNull())
        .filter((F.col('Latitude') != 0.0) & (F.col('Longitude') != 0.0))
        .filter(F.col('Latitude').between(-90.0, 90.0))
        .filter(F.col('Longitude').between(-180.0, 180.0))
    )

    # --- FILTER 5: SOG sanity ---
    df = df.withColumn('SOG',
        F.when(F.col('SOG') == 102.3, None).otherwise(F.col('SOG'))
    )
    df = df.filter((F.col('SOG') <= 60) | F.col('SOG').isNull())

    # --- FILTER 6: Geographic radius ---
    # Applied before window functions to shrink dataset maximally before
    # the more expensive per-MMSI operations below.
    df = df.withColumn('dist_km',
        F.lit(6371.0) * 2 * F.asin(F.sqrt(
            F.pow(F.sin((F.radians(F.col('Latitude')) - F.radians(F.lit(CENTER_LAT))) / 2), 2) +
            F.cos(F.radians(F.lit(CENTER_LAT))) *
            F.cos(F.radians(F.col('Latitude'))) *
            F.pow(F.sin((F.radians(F.col('Longitude')) - F.radians(F.lit(CENTER_LON))) / 2), 2)
        ))
    )
    df = df.filter(F.col('dist_km') <= RADIUS_KM).drop('dist_km')

    # --- FILTER 7: Stationary vessels ---
    parked = ['Moored', 'At anchor', '1', '5']
    df = df.filter(~F.trim(F.col('Navigational status')).isin(parked))

    # --- FILTER 8: Port zone exclusion via geohash isin() ---
    # Compute precision-7 geohash once — reused for port filtering here
    # and for collision detection in detect_collision().
    # isin() is a native JVM hash lookup — no Python UDF, no geometry math.
    df = df.withColumn('Geohash',
        geohash_center_udf(F.col('Latitude'), F.col('Longitude'))
    )
    if port_cells is not None:
        df = df.filter(~F.col('Geohash').isin(list(port_cells)))

    # --- TRANSFORM: Fill vessel name (within this day only) ---
    if 'Name' in df.columns:
        df = df.withColumn('Name',
            F.first('Name', ignorenulls=True).over(
                Window.partitionBy('MMSI')
                      .orderBy('Timestamp')
                      .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            )
        ).fillna({'Name': 'Unknown'})
    else:
        df = df.withColumn('Name', F.lit('Unknown'))

    # --- FILTER 9: GPS anomaly filter (inter-ping implied speed) ---
    # lag() operates within this day's partitions only — no cross-file shuffle.
    day_window = Window.partitionBy('MMSI').orderBy('Timestamp')
    df = (
        df
        .withColumn('prev_lat',  F.lag('Latitude',  1).over(day_window))
        .withColumn('prev_lon',  F.lag('Longitude', 1).over(day_window))
        .withColumn('prev_time', F.lag('Timestamp', 1).over(day_window))
    )
    df = df.withColumn('dt_hours',
        (F.unix_timestamp('Timestamp') - F.unix_timestamp('prev_time')) / 3600.0
    )
    df = df.withColumn('ping_dist_km',
        F.when(F.col('prev_lat').isNotNull(),
            F.lit(6371.0) * 2 * F.asin(F.sqrt(
                F.pow(F.sin((F.radians('Latitude') - F.radians('prev_lat')) / 2), 2) +
                F.cos(F.radians('prev_lat')) *
                F.cos(F.radians('Latitude')) *
                F.pow(F.sin((F.radians('Longitude') - F.radians('prev_lon')) / 2), 2)
            ))
        ).otherwise(None)
    )
    df = df.withColumn('implied_knots',
        F.when(
            F.col('ping_dist_km').isNotNull() & (F.col('dt_hours') > 0),
            (F.col('ping_dist_km') / F.col('dt_hours')) / 1.852
        ).otherwise(None)
    )
    df = df.filter(F.col('implied_knots').isNull() | (F.col('implied_knots') < 50))

    # --- TRANSFORM: SOG fill (within this day) ---
    df = df.withColumn('SOG',
        F.coalesce(
            F.col('SOG'),
            F.last('SOG', ignorenulls=True).over(
                Window.partitionBy('MMSI').orderBy('Timestamp')
                      .rowsBetween(Window.unboundedPreceding, 0)
            ),
            F.first('SOG', ignorenulls=True).over(
                Window.partitionBy('MMSI').orderBy('Timestamp')
                      .rowsBetween(0, Window.unboundedFollowing)
            ),
            F.lit(0.0)
        )
    )

    df = df.drop('prev_lat', 'prev_lon', 'prev_time', 'dt_hours',
                 'ping_dist_km', 'implied_knots')
    return df


# =============================================================================
# STEP 2 (orchestration): PROCESS ALL 31 DAYS
# =============================================================================

def load_and_clean_all(spark, csv_dir, port_cells):
    """
    Processes each daily CSV independently then unions the results.
    Window functions and port filtering run per-day, avoiding any
    global MMSI shuffle across the full 31-day dataset.
    """
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir) if f.endswith('.csv')
    ])
    print(f"         Registering {len(csv_files)} daily pipelines...")

    daily_dfs = []
    for i, csv_path in enumerate(csv_files):
        day_name = os.path.basename(csv_path)
        print(f"         [{i+1:02d}/{len(csv_files)}] {day_name}")
        daily_dfs.append(clean_one_day(spark, csv_path, port_cells))

    print("         Unioning all days (lazy)...")
    df = daily_dfs[0]
    for day_df in daily_dfs[1:]:
        df = df.union(day_df)

    print("         All transformations registered. Pipeline executes on cache().")
    return df


# =============================================================================
# STEP 3: COLLISION DETECTION (geohash neighbourhood join)
# -----------------------------------------------------------------------------
# The Geohash column is already computed per-day in clean_one_day() and
# reused directly on the left side of the join — no extra encoding pass.
#
# toPandas() is called exactly twice:
#   1. On the collision summary — a handful of rows
#   2. On the suspect vessel telemetry — a few thousand rows at most
# The full 22M row dataset is NEVER collected to the driver.
# =============================================================================

def detect_collision(spark, df):
    """
    Geohash neighbourhood join collision detection.
    Surgical telemetry export collects only suspect vessel pings to Pandas —
    never the full dataset.
    """
    step_start(3, "Detecting collision candidates via geohash neighbourhood join...")

    df = df.withColumn('Time_Bucket', F.date_trunc('minute', F.col('Timestamp')))

    # Left side: Geohash already computed in clean_one_day — reuse directly
    df_left = df

    # Right side: expand each ping to its 9-cell neighbourhood then explode.
    # explode() turns one row with a list into 9 rows, one per cell —
    # enabling a single join to replace 9 separate shuffle passes.
    df_right = (
        df
        .withColumn('Geohash_list',
            geohash_neighborhood_udf(F.col('Latitude'), F.col('Longitude')))
        .withColumn('Geohash', F.explode(F.col('Geohash_list')))
        .drop('Geohash_list')
    )

    print("         Running single geohash join...")
    merged = (
        df_left.alias('A')
        .join(df_right.alias('B'), on=['Time_Bucket', 'Geohash'], how='inner')
        .filter(F.col('A.MMSI') < F.col('B.MMSI'))
    )

    merged = merged.select(
        F.col('A.Timestamp').alias('Timestamp_A'),
        F.col('B.Timestamp').alias('Timestamp_B'),
        F.col('A.MMSI').alias('MMSI_A'),
        F.col('B.MMSI').alias('MMSI_B'),
        F.col('A.Name').alias('Name_A'),
        F.col('B.Name').alias('Name_B'),
        F.col('A.Latitude').alias('Latitude_A'),
        F.col('A.Longitude').alias('Longitude_A'),
        F.col('B.Latitude').alias('Latitude_B'),
        F.col('B.Longitude').alias('Longitude_B'),
    ).dropDuplicates(['MMSI_A', 'MMSI_B', 'Timestamp_A', 'Timestamp_B'])

    print("         Calculating precise haversine distances and filtering to 100m...")
    merged = merged.withColumn('Ship_Distance_km',
        haversine_udf(
            F.col('Latitude_A'), F.col('Longitude_A'),
            F.col('Latitude_B'), F.col('Longitude_B')
        )
    )

    collisions = merged.filter(F.col('Ship_Distance_km') < COLLISION_THRESHOLD_KM)
    collisions = collisions.withColumn('Distance_Meters', F.col('Ship_Distance_km') * 1000)

    # Keep only the closest encounter per vessel pair across all 31 days
    pair_window = Window.partitionBy('MMSI_A', 'MMSI_B').orderBy('Distance_Meters')
    collisions = (
        collisions
        .withColumn('rank', F.row_number().over(pair_window))
        .filter(F.col('rank') == 1)
        .drop('rank')
    )

    # Collect ONLY the collision summary — tiny result set (handful of pairs)
    print("         Collecting collision summary to driver...")
    collisions_pd = collisions.select(
        'Timestamp_A', 'MMSI_A', 'MMSI_B', 'Name_A', 'Name_B', 'Distance_Meters'
    ).toPandas()

    if collisions_pd.empty:
        step_done(f"no pairs came within {int(COLLISION_THRESHOLD_KM * 1000)}m")
        return

    step_done(f"{len(collisions_pd)} suspect pair(s) found")

    step_start(4, "Exporting results and surgical telemetry for Phase 2...")
    collisions_pd.to_csv(OUTPUT_CSV, index=False)
    print(f"         Saved {len(collisions_pd)} suspect pair(s) to '{OUTPUT_CSV}'")

    # -------------------------------------------------------------------------
    # SURGICAL TELEMETRY EXPORT
    # Filter df in Spark to just the suspect MMSIs, THEN collect to Pandas.
    # This is exactly what the single-day script always did — only a handful
    # of vessels. The full dataset is never collected to the driver.
    # -------------------------------------------------------------------------
    suspect_mmsis = list(
        set(collisions_pd['MMSI_A']).union(set(collisions_pd['MMSI_B']))
    )
    print(f"         Collecting telemetry for {len(suspect_mmsis)} suspect vessels...")
    suspect_telemetry_df = (
        df
        .filter(F.col('MMSI').isin(suspect_mmsis))
        .toPandas()
    )
    suspect_telemetry_df.to_csv(CLEAN_TELEMETRY_CSV, index=False)
    print(f"         Saved telemetry for {len(suspect_mmsis)} vessels to '{CLEAN_TELEMETRY_CSV}'")
    step_done()

    print(f"\n{'=' * 65}")
    print(f"  COLLISION CANDIDATES:")
    for _, row in collisions_pd.iterrows():
        print(f"  🚨 {row['Name_A']} vs {row['Name_B']} — "
              f"{row['Distance_Meters']:.1f}m apart @ {row['Timestamp_A']}")
    print(f"{'=' * 65}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pipeline_start()

    # Step 0: Build port geohash set on the driver before Spark starts.
    # Returns a frozenset of precision-7 geohash strings covering all port zones.
    port_cells = build_port_geohash_set()

    # Step 1: Extract ZIP and start Spark
    step_start(1, "Extracting ZIP and starting Spark session...")
    extract_dir = tempfile.mkdtemp(prefix="ais_spark_")
    extract_zip(ZIP_PATH, extract_dir)
    spark = create_spark_session()
    step_done("Spark running")

    try:
        # Step 2: Register per-day pipelines (lazy).
        # Port filtering is embedded inside each day's pipeline via isin() —
        # no separate port filter stage, no Python UDF, no repartition needed.
        step_start(2, "Registering per-day cleaning pipeline (lazy)...")
        df_final = load_and_clean_all(spark, extract_dir, port_cells)

        # ---------------------------------------------------------------
        # SINGLE MATERIALIZATION POINT
        # All 31 daily pipelines execute here in parallel across cores.
        # No global MMSI shuffle, no Python UDF per row, no repartition —
        # pure native Spark operations throughout.
        # ---------------------------------------------------------------
        print(f"\n[2/5] Executing pipeline and caching cleaned dataset...")
        t = time.time()
        df_final.cache()
        count = df_final.count()
        print(f"         ✓ Pipeline executed in {timedelta(seconds=int(time.time()-t))}")
        print(f"         {count:,} rows after all cleaning and filtering")
        print(f"         Total elapsed: {timedelta(seconds=int(time.time()-PIPELINE_START))}")

        # Steps 3 & 4: Collision detection + export
        # toPandas() called only on collision summary + suspect vessel subset
        detect_collision(spark, df_final)

    finally:
        spark.stop()
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        pipeline_done()