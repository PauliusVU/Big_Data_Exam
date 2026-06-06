# AIS Collision Detection — Technical Report

## The Task

Identify vessels that collided within a 50nm radius of 55.225°N, 14.245°E using Danish AIS data for December 2021. The known ground truth: **KARIN HØEJ** and **MV SCOT CARRIER** collided on **2021-12-13 at 02:27:29 UTC** at a closest recorded proximity of **4.1 metres**.

**Result.** The pipeline identifies a single collision: **KARIN HØEJ** (MMSI 219021240) and **MV SCOT CARRIER** (MMSI 232018267), on **2021-12-13 at 02:27:29 UTC**, at **latitude 55.223699, longitude 14.248298**, with the two vessels **4.1 metres** apart at closest approach.

---

## Where We Started — The Single-Day Prototype

The initial prototype was a single-day Pandas script (`Phase_1.py`, `Phase_2.py`) targeting December 13th. Several critical data engineering problems had to be solved before scaling to PySpark:

**The RAM Crash & Genesis of Spatial Bucketing** — the earliest iteration attempted to self-join all vessels active in the same minute, generating a Cartesian product that instantly exhausted local memory. This forced the implementation of spatial bucketing to ensure distance calculations only ran for vessels already in the same space-time block. The final PySpark implementation replaces this with a single geohash neighbourhood join.

**The OpenStreetMap Tagging Problem** — `osmnx` was used to mask port zones, but OSM's crowd-sourced tagging caused blind spots (e.g. Malmö port tagged as `landuse=industrial` rather than `port`). Tag queries were expanded exhaustively and geometry processing was strictly sequenced (buffer before dissolve) to produce a reliable 700m exclusion zone.

**The AIS Message Type 5 Static Data Gap** — AIS static messages (vessel dimensions) broadcast only every 6 minutes, so Length/Width had to be forward-filled across dynamic pings to preserve the asymmetric size logic in Phase 2.

**Inter-ping GPS anomaly filter** — added a `lag()`-based implied speed filter (>50 knots = GPS noise), O(n) with negligible cost.

**SOG shock directionality** — changed from speed drop detection to `abs()` to catch both deceleration and acceleration (KARIN HØEJ was shoved forward, spiking from 6.1 to 10.3 knots at impact).

**Variance floor** — when a vessel has near-zero pre-collision speed variance, any tiny change produces an arbitrarily large Z-score. A floor of 0.5 was applied when std is undefined or zero.

---

## Porting to PySpark

The full 31-day dataset is ~16GB compressed. The core challenge was making this tractable without running out of memory or spending hours on unnecessary computation.

### Per-Day Processing

Each daily CSV is processed independently rather than loading all 31 days at once. Window functions like `lag()`, `first()`/`last()`, and SOG forward-fill require co-locating all pings per MMSI — doing this globally forces Spark to shuffle every MMSI's pings across all 31 files. Processing day-by-day keeps window functions entirely local to each day's partitions, eliminating the cross-file shuffle. The 31 daily DataFrames are unioned lazily and materialised exactly once via `cache().count()`.

### Geohash Port Filter

A Python UDF with Shapely geometry per row would serialise geometry objects across the network for all 22M pings. Instead, port zones were precomputed on the driver from OSM PBF files, buffered by 700m, and enumerated into a frozenset of ~28,963 precision-7 geohash strings (~170m × 214m cells). Spark filters via `.isin()` — a native JVM hash lookup with no Python overhead.

### Single Geohash Join

The original 9-merge bucketing approach would become 9 distributed shuffles in Spark. Instead, each ping on the right side of the join is expanded into 9 rows (itself + 8 neighbours), enabling a single inner join on `(Time_Bucket, Geohash)` to replace all 9 shuffles.

### The toPandas() Bug

The first working PySpark version collected the entire 22M-row cleaned dataset to Pandas before filtering to suspect vessels, causing memory crashes. The fix was to filter in Spark to just the suspect MMSIs first, then call `toPandas()` on only those few hundred vessels.

**Runtime.** On the development machine (11 cores, Apple M-series) Phase 1 ran in ~18 minutes. The reproducible reference is the containerised run on 4 cores / 8GB Docker: ~25 minutes for Phase 1 and ~26 minutes end-to-end.

---

## Phase 2 — Forensic Verification

Phase 1 produced 480 candidate pairs across 31 days. Phase 2 reduced these to the actual collision(s).

### Speed

The initial Phase 2 scanned the full telemetry DataFrame on every candidate lookup (~10 minutes for 480 pairs). Fixed by pre-indexing once at startup:
```python
vessel_index = {mmsi: group for mmsi, group in df.groupby('MMSI')}
```
This dropped runtime to under 30 seconds.

### Iterative False Positive Reduction


**ROT spike alone is insufficient** — a sharp turn is normal maritime behaviour; ROT spike now requires corroboration from at least one other signal.

**Catastrophic blackout alone is insufficient** — a transponder fault also causes silence; blackout alone no longer confirms trauma.

**These observations led to the Dual-signal requirement** — a single anomaly is weak evidence since a speed change or heading skid can result from normal manoeuvring. "Dual-signal trauma" requires at least two of the four independent signals (SOG shock, heading/COG skid, ROT spike, catastrophic blackout) within the ±5-minute window.

**"Both vessels must show trauma depending on their size"** — initially reduced false positives by requiring both vessels to show trauma. But adjusted this to account for vessel size. If size information is available (length, width - worst case scenario fallback based on Class), we take this into account on whether both vessels should show trauma:

For **symmetric vessels** both must show dual-signal trauma. For **asymmetric pairs** (size ratio ≥2×) only the smaller vessel is required to — a large ship may not detectably register a small-vessel impact. This size-aware rule (`trauma_required()`) helped reduce false positives.

**Post-window gap truncation** — PILOT 213 SE was generating a spurious Z=90.8 SOG shock from a 4-minute AIS blackout mid-window. The post window is now truncated at any gap >2 minutes. This is generalisable for future cases as well.

### Final Result

After all filters: **1 confirmed collision**.

| Field | Value |
|---|---|
| Vessel A | KARIN HØEJ (MMSI: 219021240) |
| Vessel B | MV SCOT CARRIER (MMSI: 232018267) |
| Timestamp | 2021-12-13 02:27:29 UTC |
| Coordinates | 55.223699, 14.248298 |
| Closest proximity | 4.1 metres |
| Trauma — KARIN HØEJ | SOG Shock (Z=5.4), Heading/COG Skid (Z=5.4) |
| Trauma — MV SCOT CARRIER | SOG Shock (Z=14.7), Heading/COG Skid (Z=4.9) |

### On the Choice of Thresholds

None of the numerical thresholds were reverse-engineered to single out KARIN HØEJ:

- **Z-score threshold of 3.0** — the standard three-sigma cutoff (~99.7th percentile). The observed trauma scores (Z=5.4 to 14.7) sit well above it.
- **Variance floor of 0.5 knots** — guards against division-by-zero on rock-steady vessels; required for any dataset where a vessel holds a perfectly steady course.
- **50-knot teleport filter** — sits above any plausible vessel speed; an implied speed beyond it marks a position jump rather than real movement.
- **2-minute gap truncation** — discards the same teleportation artefact in the time dimension.
- **100m proximity** — For phase 1 w select a deliberately loose distance to absorb AIS positional jitter; a tolerance band, not a definition of collision distance.
- **Size ratio of 2.0** — a defensible line for "materially different in size" grounded in the physics of impact asymmetry.

---

## Computational Architecture — Design Rationale

### Where Spark Is Used, and Where Pandas Is

All large-scale processing runs in PySpark — reading 31 daily CSVs, schema casting, every cleaning filter, the geohash neighbourhood join, and proximity ranking — covering **22,464,519 rows** after cleaning and filtering. Pandas is used only after Spark has reduced the problem to a small size: the 480-candidate summary and the few-hundred-vessel suspect telemetry. This follows the standard big-data pattern of distributing heavy work and collecting to the driver only once the data is small.

### Parallel Execution vs. Sequenced Stages

The 31 daily files are not processed one after another. `load_and_clean_all` builds lazy DataFrames; nothing executes during the loop. They are unioned into one logical plan and materialised by a single `cache().count()`, at which point Spark schedules all 31 days' work in parallel across the allocated cores. The three phases (detection → verification → visualization) run sequentially because each consumes the prior phase's output files — a data dependency, not a missed parallelisation.

### Interpreting CPU Utilisation

For parts of the run, CPU sits near one core rather than saturating all four. Two causes account for this: OSM PBF parsing and ZIP extraction are inherently single-threaded (~5 minutes combined), and reading ~16GB of CSV from disk is I/O-bound — cores wait on disk rather than compute. The CPU-bound transformation stages do parallelise across all allocated cores, as the concurrent-task progress lines in the run logs confirm.

### Why the Cleaned Dataset Is Cached Whole (and Not Batched)

The cleaned 22.5M-row dataset is cached once and reused for both the collision join and the suspect-telemetry export, avoiding recomputation of the cleaning DAG. Batched processing was considered for larger inputs but is unnecessary at this scale — the 8GB configuration completes successfully. Batching would add boundary bookkeeping (accumulating the global suspect-MMSI set across batches, carrying ping overlap across seams) without reducing runtime. For a substantially longer timeframe the per-day design lends itself to that extension, since each day is already cleaned independently with no cross-day shuffle.

---

## Known Limitations

**Midnight boundary gap** — per-day processing cannot detect collisions whose closest pings straddle day boundaries. Given AIS pings every 2–10 seconds, this is an astronomically unlikely scenario for any real collision.

**AIS self-reported data** — vessel dimensions and ship type codes are crew-reported and may be missing or incorrect. The asymmetric trauma logic degrades gracefully when data is absent.
