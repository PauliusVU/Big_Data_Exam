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

# TEST MODE — single day only, no port filter
TEST_MODE = True
TEST_DAY  = "aisdk-2021-12-13.csv"  # the collision day

PIPELINE_START = None
STEP_START     = None
TOTAL_STEPS    = 5


def pipeline_start():
    global PIPELINE_START
    PIPELINE_START = time.time()
    print("=" * 65)
    print("  AIS COLLISION DETECTION — TEST MODE (1 day, no port filter)")
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


@F.udf(ArrayType(StringType()))
def geohash_neighborhood_udf(lat, lon):
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


def extract_zip(zip_path, extract_dir):
    print(f"         Extracting CSVs from {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    print(f"         Extracted {len(csv_files)} daily CSV files.")
    return extract_dir


def create_spark_session():
    cores = max(1, os.cpu_count() - 1)
    print(f"         Starting Spark session (local[{cores}])...")
    spark = (
        SparkSession.builder
        .appName("AIS_Test")
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

    # Geohash + port filter
    df = df.withColumn('Geohash', geohash_center_udf(F.col('Latitude'), F.col('Longitude')))
    if port_cells is not None:
        df = df.filter(~F.col('Geohash').isin(list(port_cells)))

    if 'Name' in df.columns:
        df = df.withColumn('Name',
            F.first('Name', ignorenulls=True).over(
                Window.partitionBy('MMSI').orderBy('Timestamp')
                      .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            )
        ).fillna({'Name': 'Unknown'})
    else:
        df = df.withColumn('Name', F.lit('Unknown'))

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


def detect_collision(spark, df):
    """
    Collision detection + surgical telemetry export.
    toPandas() is only called on the tiny suspect vessel subset —
    never on the full dataset.
    """
    step_start(4, "Detecting collision candidates via geohash neighbourhood join...")

    df = df.withColumn('Time_Bucket', F.date_trunc('minute', F.col('Timestamp')))

    df_left = df

    df_right = (
        df
        .withColumn('Geohash_list', geohash_neighborhood_udf(F.col('Latitude'), F.col('Longitude')))
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

    print("         Calculating haversine distances...")
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

    # Collect ONLY the collision summary — tiny result set
    print("         Collecting collision summary to driver...")
    collisions_pd = collisions.select(
        'Timestamp_A', 'MMSI_A', 'MMSI_B', 'Name_A', 'Name_B', 'Distance_Meters'
    ).toPandas()

    if collisions_pd.empty:
        step_done(f"no pairs came within {int(COLLISION_THRESHOLD_KM * 1000)}m")
        return

    step_done(f"{len(collisions_pd)} suspect pair(s) found")

    step_start(5, "Exporting results and surgical telemetry for Phase 2...")
    collisions_pd.to_csv(OUTPUT_CSV, index=False)
    print(f"         Saved {len(collisions_pd)} suspect pair(s) to '{OUTPUT_CSV}'")

    # -------------------------------------------------------------------------
    # SURGICAL TELEMETRY EXPORT
    # Filter df in Spark to just the suspect MMSIs, THEN collect to Pandas.
    # This is what the single-day script always did — only a handful of vessels.
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


if __name__ == "__main__":
    pipeline_start()

    # TEST MODE: skip port filter
    port_cells = None
    print("\n[TEST MODE] Port filter skipped, processing single day only.")

    step_start(1, "Extracting ZIP and starting Spark session...")
    extract_dir = tempfile.mkdtemp(prefix="ais_spark_")
    extract_zip(ZIP_PATH, extract_dir)
    spark = create_spark_session()
    step_done("Spark running")

    try:
        step_start(2, f"Processing single day: {TEST_DAY}...")
        csv_path = os.path.join(extract_dir, TEST_DAY)
        df_final = clean_one_day(spark, csv_path, port_cells)

        print(f"\n[2/5] Caching cleaned dataset...")
        t = time.time()
        df_final.cache()
        count = df_final.count()
        print(f"         ✓ Cached in {timedelta(seconds=int(time.time()-t))} — {count:,} rows")
        print(f"         Total elapsed: {timedelta(seconds=int(time.time()-PIPELINE_START))}")

        step_start(3, "Running collision detection...")
        detect_collision(spark, df_final)

    finally:
        spark.stop()
        import shutil
        shutil.rmtree(extract_dir, ignore_errors=True)
        pipeline_done()