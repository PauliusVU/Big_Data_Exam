import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

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
from pyspark.sql.types import StringType, ArrayType, DoubleType

warnings.filterwarnings('ignore')

# Output directory: set via OUTPUT_DIR env variable in Docker, defaults to '.' locally
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory: set via DATA_DIR env variable in Docker, defaults to 'data' locally
DATA_DIR = os.environ.get('DATA_DIR', 'data')

# Configuration
ZIP_PATH            = os.path.join(DATA_DIR, "aisdk-2021-12.zip")
OUTPUT_CSV          = os.path.join(OUTPUT_DIR, "suspected_collisions_list.csv")
CLEAN_TELEMETRY_CSV = os.path.join(OUTPUT_DIR, "clean_suspect_telemetry.csv")

OSM_FILES = [
    os.path.join(DATA_DIR, "denmark-latest.osm.pbf"),
    os.path.join(DATA_DIR, "sweden-latest.osm.pbf"),
]

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM              = 50
RADIUS_KM              = RADIUS_NM * 1.852
COLLISION_THRESHOLD_KM = 0.100
DIRTY_MMSI             = {"000000000", "111111111", "123456789"}

BBOX = (
    CENTER_LON - 1.5,
    CENTER_LAT - 1.0,
    CENTER_LON + 1.5,
    CENTER_LAT + 1.0
)

# OSM tags used to identify port zones. Expanded exhaustively to handle
# inconsistent crowd-sourced tagging (e.g. Malmö port tagged as industrial).
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


# Progress reporting

PIPELINE_START = None
STEP_START     = None
TOTAL_STEPS    = 4


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
    print(f"         Elapsed: {elapsed}")
    print(f"         " + "-" * 50)


def step_done(detail=""):
    elapsed_step  = timedelta(seconds=int(time.time() - STEP_START))
    elapsed_total = timedelta(seconds=int(time.time() - PIPELINE_START))
    suffix = f" — {detail}" if detail else ""
    print(f"         Done in {elapsed_step}{suffix}")
    print(f"         Total elapsed: {elapsed_total}")


def pipeline_done():
    total = timedelta(seconds=int(time.time() - PIPELINE_START))
    print("\n" + "=" * 65)
    print(f"  Phase 1 complete — runtime: {total}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {CLEAN_TELEMETRY_CSV}")
    print("=" * 65)


# Port exclusion geohash set
#
# Instead of broadcasting Shapely geometry to Spark workers and running a
# Python UDF per row, we precompute the set of all precision-7 geohash cells
# whose centre point falls inside any port zone. Spark then filters via
# isin() — a native JVM hash lookup with no Python overhead.
#
# Precision-7 cells are ~170m x 214m. A 700m buffer ensures any cell
# overlapping a port zone is caught. The buffer is applied before dissolve
# to avoid topology errors from open LineString geometries in OSM.

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
    step_start(1, "Building port exclusion set from OSM data...")
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
        print("         No port data found — port filter disabled.")
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

    print("         Enumerating geohash cells over port zones...")
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
    print(f"         {len(port_cells)} precision-7 cells covering port zones.")
    step_done(f"{len(port_cells)} cells")
    return port_cells


# Spark UDFs

@F.udf(ArrayType(StringType()))
def geohash_neighborhood_udf(lat, lon):
    # Returns a precision-7 geohash for (lat, lon) plus its 8 neighbours.
    # Exploding this list on the right side of the join enables a single
    # shuffle to replace the 9 separate joins used in the Pandas prototype.
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
    # Computed once per ping and reused for both port filtering and
    # the left side of the collision join — no redundant encoding pass.
    if lat is None or lon is None:
        return None
    return gh.encode(lat, lon, precision=7)


@F.udf(DoubleType())
def haversine_udf(lat1, lon1, lat2, lon2):
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# Spark session and ZIP extraction

def extract_zip(zip_path, extract_dir):
    print(f"         Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    print(f"         {len(csv_files)} daily CSV files extracted.")
    return extract_dir


def create_spark_session():
    # Cores and driver memory are configurable via environment variables so the
    # same image can run on a small laptop or a larger workstation without code
    # changes. Defaults reproduce the original tested configuration: if
    # SPARK_CORES is unset, fall back to (CPU count - 1, capped at 4); if
    # SPARK_DRIVER_MEM is unset, use 4g. Override them in docker-compose.yml
    # (environment:) or via `docker run -e SPARK_CORES=... -e SPARK_DRIVER_MEM=...`.
    default_cores = str(min(4, max(1, os.cpu_count() - 1)))
    cores      = os.environ.get('SPARK_CORES', '4')
    driver_mem = os.environ.get('SPARK_DRIVER_MEM', '4g')
    print(f"         Starting Spark (local[{cores}], {driver_mem} driver)...")
    spark = (
        SparkSession.builder
        .appName("AIS_Collision_Detection")
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"         Spark ready on {cores} cores.")
    return spark


# Per-day cleaning pipeline
#
# Each daily CSV is processed independently so that window functions
# operate within each day's partitions only, avoiding the expensive
# global MMSI shuffle that a single-read approach requires. The initial
# PySpark version read all 31 files at once — it was significantly slower.
#
# Geohash is computed once per ping and reused for both port filtering
# and the collision detection join.

def clean_one_day(spark, csv_path, port_cells):
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

    if 'Length' in df.columns:
        df = df.withColumn('Length', F.col('Length').cast(DoubleType()))
    if 'Width' in df.columns:
        df = df.withColumn('Width', F.col('Width').cast(DoubleType()))

    df = df.filter(F.col('Timestamp').isNotNull())

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

    if 'Name' in df.columns:
        df = df.filter(~F.upper(F.col('Name')).rlike('WINDFARM|PLATFORM'))

    df = (
        df
        .filter(F.col('Latitude').isNotNull() & F.col('Longitude').isNotNull())
        .filter((F.col('Latitude') != 0.0) & (F.col('Longitude') != 0.0))
        .filter(F.col('Latitude').between(-90.0, 90.0))
        .filter(F.col('Longitude').between(-180.0, 180.0))
    )

    df = df.withColumn('SOG', F.when(F.col('SOG') == 102.3, None).otherwise(F.col('SOG')))
    df = df.filter((F.col('SOG') <= 60) | F.col('SOG').isNull())

    # Geographic radius filter applied before window functions to shrink
    # the dataset as much as possible before the more expensive per-MMSI ops.
    df = df.withColumn('dist_km',
        F.lit(6371.0) * 2 * F.asin(F.sqrt(
            F.pow(F.sin((F.radians(F.col('Latitude')) - F.radians(F.lit(CENTER_LAT))) / 2), 2) +
            F.cos(F.radians(F.lit(CENTER_LAT))) *
            F.cos(F.radians(F.col('Latitude'))) *
            F.pow(F.sin((F.radians(F.col('Longitude')) - F.radians(F.lit(CENTER_LON))) / 2), 2)
        ))
    )
    df = df.filter(F.col('dist_km') <= RADIUS_KM).drop('dist_km')

    parked = ['Moored', 'At anchor', '1', '5']
    df = df.filter(~F.trim(F.col('Navigational status')).isin(parked))

    df = df.withColumn('Geohash', geohash_center_udf(F.col('Latitude'), F.col('Longitude')))
    if port_cells is not None:
        df = df.filter(~F.col('Geohash').isin(list(port_cells)))

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

    # Inter-ping implied speed filter: drops GPS teleportation artefacts.
    # lag() runs within this day's partitions only — no cross-file shuffle.
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

    df = df.drop('prev_lat', 'prev_lon', 'prev_time', 'dt_hours', 'ping_dist_km', 'implied_knots')
    return df


def load_and_clean_all(spark, csv_dir, port_cells):
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir) if f.endswith('.csv')
    ])
    print(f"         Registering {len(csv_files)} daily pipelines...")
    daily_dfs = [clean_one_day(spark, p, port_cells) for p in csv_files]

    df = daily_dfs[0]
    for day_df in daily_dfs[1:]:
        df = df.union(day_df)
    return df


# Collision detection
#
# The Geohash column computed in clean_one_day() is reused directly on
# the left side of the join. The right side explodes each ping into 9 rows
# (itself + 8 neighbours), enabling a single join to replace the 9 separate
# merges used in the Pandas prototype.
#
# toPandas() is called exactly twice — once on the collision summary
# (a handful of rows) and once on the filtered suspect vessel telemetry.
# The full cleaned dataset is never collected to the driver.

def detect_collision(spark, df):
    step_start(3, "Running geohash neighbourhood join for collision candidates...")

    df = df.withColumn('Time_Bucket', F.date_trunc('minute', F.col('Timestamp')))

    df_left = df
    df_right = (
        df
        .withColumn('Geohash_list', geohash_neighborhood_udf(F.col('Latitude'), F.col('Longitude')))
        .withColumn('Geohash', F.explode(F.col('Geohash_list')))
        .drop('Geohash_list')
    )

    print("         Joining...")
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

    merged = merged.withColumn('Ship_Distance_km',
        haversine_udf(F.col('Latitude_A'), F.col('Longitude_A'),
                      F.col('Latitude_B'), F.col('Longitude_B'))
    )

    collisions = merged.filter(F.col('Ship_Distance_km') < COLLISION_THRESHOLD_KM)
    collisions = collisions.withColumn('Distance_Meters', F.col('Ship_Distance_km') * 1000)

    pair_window = Window.partitionBy('MMSI_A', 'MMSI_B').orderBy('Distance_Meters')
    collisions = (
        collisions
        .withColumn('rank', F.row_number().over(pair_window))
        .filter(F.col('rank') == 1)
        .drop('rank')
    )

    print("         Collecting collision summary...")
    collisions_pd = collisions.select(
        'Timestamp_A', 'MMSI_A', 'MMSI_B', 'Name_A', 'Name_B', 'Distance_Meters'
    ).toPandas()

    if collisions_pd.empty:
        step_done(f"no pairs found within {int(COLLISION_THRESHOLD_KM * 1000)}m")
        return

    step_done(f"{len(collisions_pd)} candidate pair(s)")

    step_start(4, "Exporting candidate list and suspect vessel telemetry...")
    collisions_pd.to_csv(OUTPUT_CSV, index=False)
    print(f"         {OUTPUT_CSV}")

    suspect_mmsis = list(set(collisions_pd['MMSI_A']).union(set(collisions_pd['MMSI_B'])))
    print(f"         Writing telemetry for {len(suspect_mmsis)} vessels...")

    telemetry_cols = [
        'Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude',
        'SOG', 'COG', 'Heading', 'ROT', 'Navigational status',
        'Type of mobile', 'Ship type', 'Length', 'Width'
    ]
    available_cols = [c for c in telemetry_cols if c in df.columns]
    (
        df
        .filter(F.col('MMSI').isin(suspect_mmsis))
        .select(available_cols)
        .coalesce(1)
        .write
        .option("header", "true")
        .mode("overwrite")
        .csv(CLEAN_TELEMETRY_CSV)
    )
    print(f"         {CLEAN_TELEMETRY_CSV}")
    step_done()


if __name__ == "__main__":
    pipeline_start()

    port_cells = build_port_geohash_set()

    step_start(2, "Extracting ZIP and starting Spark...")
    extract_dir = tempfile.mkdtemp(prefix="ais_spark_")
    extract_zip(ZIP_PATH, extract_dir)
    spark = create_spark_session()
    step_done("Spark running")

    try:
        df_final = load_and_clean_all(spark, extract_dir, port_cells)

        print(f"\n[3/{TOTAL_STEPS}] Executing pipeline and caching cleaned dataset...")
        t = time.time()
        df_final.cache()
        count = df_final.count()
        print(f"         Done in {timedelta(seconds=int(time.time()-t))}")
        print(f"         {count:,} rows after cleaning and filtering")
        print(f"         Total elapsed: {timedelta(seconds=int(time.time()-PIPELINE_START))}")

        detect_collision(spark, df_final)

    finally:
        spark.stop()
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        pipeline_done()