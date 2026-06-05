# AIS Collision Detection — Big Data Analysis Exam

Detection and forensic verification of vessel collisions from the full Danish AIS dataset (December 2021) using PySpark on Docker.

---

## Project Structure

```
.
├── data/                          # Input data (not in git — see Requirements)
├── outputs/                       # Pipeline outputs (generated at runtime)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── *.py                           # See Scripts below
```

---

## Scripts

### Main Docker Pipeline (entry points)

These four scripts are the ones invoked by the Docker container and constitute the full reproducible pipeline:

| Script | Purpose |
|---|---|
| `main.py` | Container entry point — orchestrates Phase 1 → Phase 2 → Visualization |
| `PySpark_Docker_Phase_1.py` | PySpark 31-day AIS ingestion, filtering, and collision candidate detection |
| `PySpark_Docker_Phase_2.py` | Forensic verification of collision candidates |
| `PySpark_Docker_visualize.py` | Trajectory map generation for confirmed collision pairs |

### Analysis & Development Scripts

The remaining scripts were written for exploratory analysis, debugging, and iterative development during the exam. They have very specific purposes and are not part of the Docker pipeline, but are included for full reproducibility of the research process.

| Script | Purpose |
|---|---|
| `Phase_1.py` | Early prototype of Phase 1 logic (non-Spark) |
| `Phase_2.py` | Early prototype of Phase 2 logic (non-Spark) |
| `PySpark_Phase_1.py` | Intermediate PySpark Phase 1 before Docker refactor |
| `PySpark_Phase_1_1_day_test.py` | Single-day subset test for rapid iteration |
| `collision_phase_2_v3.py` | Iterative version of Phase 2 collision verification |
| `collision_inspection_Claude_v2.py` | Deep inspection of individual collision candidates |
| `audit_sog_imputation.py` | Audit of speed-over-ground (SOG) imputation quality |
| `inspect_missing_sog.py` | Investigation of missing SOG values in raw AIS data |
| `inspect_crash_date.py` | Date-range sanity checks on the crash events |
| `fullday_mapping.py` | Full-day encounter map generation |
| `telemetry_visualization.py` | Telemetry plotting for individual vessel tracks |
| `port_zone_map.py` | Port zone boundary mapping |
| `phase_2_test.py` | Unit/integration tests for Phase 2 logic |
| `download.py` | Helper to download AIS data files |
| `download_test.py` | Smoke test for the download helper |
| `test.py` | General scratch/test script |

---

## Requirements

- Docker + Docker Compose
- **4 CPU cores and 8GB RAM allocated to Docker** (tested configuration — see note below)
- The following data files placed in a `data/` directory:
  - `aisdk-2021-12.zip` — Danish AIS data ([aisdata.ais.dk](http://aisdata.ais.dk/))
  - `denmark-latest.osm.pbf` — OpenStreetMap Denmark ([Geofabrik](https://download.geofabrik.de/europe/denmark.html))
  - `sweden-latest.osm.pbf` — OpenStreetMap Sweden ([Geofabrik](https://download.geofabrik.de/europe/sweden.html))

> **Memory note.** "8GB allocated to Docker" means the memory available to the
> container, not the host's total RAM. On Docker Desktop, set the memory slider
> (Settings → Resources) to at least 8GB. The full December 2021 pipeline was run
> end-to-end in this configuration and produced the correct result.

---

## Running the Pipeline

```bash
docker compose up
```

Outputs are written to `./outputs/`.

---

## Tuning for Your Machine

Spark cores and driver memory are exposed as environment variables in
`docker-compose.yml`, so the same image runs on a small laptop or a larger
workstation without editing any code:

```yaml
environment:
  - SPARK_CORES=4        # number of Spark cores
  - SPARK_DRIVER_MEM=4g  # Spark driver heap
deploy:
  resources:
    limits:
      memory: 8g         # container memory ceiling
```

Defaults (`4` cores, `4g` driver, `8g` container) match the tested configuration.
On a larger machine, raise these for more speed. On a smaller one, the pipeline
still runs but more slowly; if Spark reports an out-of-memory error, lower
`SPARK_DRIVER_MEM` to `3g`.

If you run the image directly with `docker run` instead of `docker compose up`
(for example, after pulling from Docker Hub without the compose file), pass the
same settings as flags:

```bash
docker run \
  -e SPARK_CORES=4 \
  -e SPARK_DRIVER_MEM=4g \
  -e OUTPUT_DIR=/app/outputs \
  --memory=8g \
  --shm-size=2g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  pauliusvu/ais-collision-detection:v1.0
```

---

## Expected Runtime

Measured on the tested configuration (4 cores, 8GB RAM allocated to Docker,
full December 2021 dataset):

| Phase | Description | Time |
|---|---|---|
| Phase 1 | PySpark 31-day pipeline | ~25 min |
| Phase 2 | Forensic verification (480 candidates) | <1 min |
| Phase 3 | Trajectory visualization | <1 min |
| **Total** | | **~26 min** |

Runtime scales with the cores and memory allocated. A machine with more
resources will complete faster; the figures above are a conservative reference
from a deliberately modest 8GB configuration.
