# AIS Collision Detection Pipeline
# Runs the full pipeline in sequence:
#   1. Phase 1 (PySpark) — 31-day collision candidate detection
#   2. Phase 2 (Pandas)  — forensic verification of candidates
#   3. Visualization     — trajectory maps for confirmed collisions
#
# Each phase is launched as a subprocess rather than imported and called
# directly. This keeps each phase's Spark session and memory footprint fully
# isolated — a SparkSession started in Phase 1 would otherwise persist in
# the same Python process and conflict with Phase 2's Pandas-only environment.
# Subprocess isolation also means each phase can be re-run independently
# without restarting the entire pipeline:
#     python PySpark_Docker_Phase_1.py
#     python PySpark_Docker_Phase_2.py
#     python PySpark_Docker_visualize.py


import subprocess
import sys
import os
import time
import pandas as pd
from datetime import timedelta

START = time.time()

# Output directory: set via OUTPUT_DIR env variable in Docker, defaults to '.' locally
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# run() uses check=False rather than check=True so that we can print a
# labelled failure message with elapsed time before exiting. check=True would
# raise CalledProcessError immediately with no context about which phase failed
# or how long it ran.
def run(script, label):
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}\n")
    t = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), script)],
        check=False
    )
    elapsed = timedelta(seconds=int(time.time() - t))
    if result.returncode != 0:
        print(f"\n  {label} failed after {elapsed}. Aborting.")
        sys.exit(result.returncode)
    print(f"\n  {label} completed in {elapsed}")


# Reads results from disk rather than accepting in-memory arguments — phases
# run as isolated subprocesses with no shared state. The telemetry glob handles
# both the Spark directory form (part-*.csv) and a flat CSV from local runs.
def print_final_summary():
    final_csv     = os.path.join(OUTPUT_DIR, "FINAL_investigation_results.csv")
    telemetry_csv = os.path.join(OUTPUT_DIR, "clean_suspect_telemetry.csv")

    if not os.path.exists(final_csv):
        print("  FINAL_investigation_results.csv not found.")
        return

    results = pd.read_csv(final_csv)
    results['Timestamp_A'] = pd.to_datetime(results['Timestamp_A'])

    if results.empty:
        print("  No confirmed collisions.")
        return

    tel = None
    import glob as _glob
    if os.path.exists(telemetry_csv) or os.path.isdir(telemetry_csv):
        _parts = _glob.glob(os.path.join(telemetry_csv, 'part-*.csv'))
        tel = pd.read_csv(_parts[0] if _parts else telemetry_csv)
        tel['Timestamp'] = pd.to_datetime(tel['Timestamp']).apply(lambda x: x.replace(tzinfo=None))
        tel['MMSI'] = tel['MMSI'].astype(str).str.strip()

    print(f"\n{'=' * 65}")
    print(f"  CONFIRMED COLLISION(S)")
    print(f"{'=' * 65}")

    for _, row in results.iterrows():
        mmsi_a = str(row['MMSI_A'])
        mmsi_b = str(row['MMSI_B'])
        name_a = row['Name_A']
        name_b = row['Name_B']
        ts     = row['Timestamp_A']
        dist   = row['Distance_Meters']

        col_lat, col_lon = None, None
        if tel is not None:
            window = pd.Timedelta(seconds=60)
            near_a = tel[(tel['MMSI'] == mmsi_a) &
                         (tel['Timestamp'].between(ts - window, ts + window))]
            near_b = tel[(tel['MMSI'] == mmsi_b) &
                         (tel['Timestamp'].between(ts - window, ts + window))]
            if not near_a.empty and not near_b.empty:
                col_lat = (near_a.iloc[0]['Latitude'] + near_b.iloc[0]['Latitude']) / 2
                col_lon = (near_a.iloc[0]['Longitude'] + near_b.iloc[0]['Longitude']) / 2

        print(f"\n  Vessel A:   {name_a}")
        print(f"  MMSI A:     {mmsi_a}")
        print(f"  Vessel B:   {name_b}")
        print(f"  MMSI B:     {mmsi_b}")
        print(f"  Timestamp:  {ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        if col_lat is not None:
            print(f"  Latitude:   {col_lat:.6f}")
            print(f"  Longitude:  {col_lon:.6f}")
        print(f"  Distance:   {dist:.1f} metres")

    print(f"\n{'=' * 65}")
    print(f"  Output files in {OUTPUT_DIR}/")
    print(f"    suspected_collisions_list.csv   — Phase 1 candidates")
    print(f"    clean_suspect_telemetry.csv     — suspect vessel telemetry")
    print(f"    FINAL_investigation_results.csv — confirmed collisions")
    print(f"    telemetry_*.csv                 — per-collision telemetry")
    print(f"    map_*.html                      — trajectory maps")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run("PySpark_Docker_Phase_1.py", "PHASE 1 — Collision Detection (PySpark)")
    run("PySpark_Docker_Phase_2.py", "PHASE 2 — Forensic Verification")
    run("PySpark_Docker_visualize.py", "PHASE 3 — Trajectory Visualization")

    total = timedelta(seconds=int(time.time() - START))
    print(f"\n{'=' * 65}")
    print(f"  FULL PIPELINE COMPLETE — Total runtime: {total}")
    print(f"{'=' * 65}")

    print_final_summary()
