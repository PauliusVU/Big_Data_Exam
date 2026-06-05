# AIS Collision Detection — Technical Report

## The Task

Identify vessels that collided within a 50nm radius of 55.225°N, 14.245°E using Danish AIS data for December 2021. The known ground truth: **KARIN HØEJ** and **MV SCOT CARRIER** collided on **2021-12-13 at 02:27:29 UTC** at a closest recorded proximity of **4.1 metres**.

**Result.** The pipeline identifies a single collision: **KARIN HØEJ** (MMSI 219021240) and **MV SCOT CARRIER** (MMSI 232018267), on **2021-12-13 at 02:27:29 UTC**, at **latitude 55.223699, longitude 14.248298**, with the two vessels **4.1 metres** apart at closest approach.

---

## Where We Started — The Single-Day Prototype

The initial prototype was a single-day Pandas script (`Phase_1.py`, `Phase_2.py`) targeting December 13th. Before scaling to PySpark, several critical data engineering bottlenecks and physics anomalies had to be solved locally:

**The RAM Crash & Genesis of Spatial Bucketing** — the earliest iteration attempted to self-join all vessels active in the same minute to calculate proximity. This generated a massive Cartesian product that instantly exhausted local memory (`[1] 46249 killed` on macOS). This forced the implementation of strict spatial bucketing — rounding coordinates to create a grid, ensuring the CPU only calculated exact Haversine distances for vessels in the same space-time block. The final PySpark implementation replaces this with a single geohash neighbourhood join.

**The OpenStreetMap Tagging Problem** — `osmnx` was used to mask port zones and prevent stationary vessels from triggering alerts. However, OpenStreetMap's crowd-sourced nature caused significant blind spots. The entire commercial port of Malmö was tagged as `landuse=industrial` rather than `port`, and marine infrastructure like breakwaters were mapped as open `LineString` geometries, causing GeoPandas' `buffer(0)` topology fix to silently destroy parts of the exclusion mask. Tag queries had to be expanded exhaustively, and the processing order strictly sequenced (buffer first, then dissolve) to produce a reliable 700m exclusion zone around port areas.

**The AIS Message Type 5 Static Data Gap** — AIS dynamic messages (location/speed) broadcast every few seconds, but static messages (Length/Width) broadcast only every 6 minutes. Vessel dimensions needed to be forward-filled across dynamic pings to avoid large vessels defaulting to missing dimension data, which would destroy the asymmetric size logic in Phase 2.

**Inter-ping GPS anomaly filter** — the script had no check for vessels "teleporting" between consecutive pings. Added a `lag()`-based implied speed filter (>50 knots = GPS noise), which is O(n) and adds negligible cost.

**SOG shock directionality** — the original test only caught speed *drops* post-collision. Changed to `abs()` to catch both deceleration and acceleration — crucial, as KARIN HØEJ was violently shoved forward, spiking from 6.1 to 10.3 knots at impact.

**Variance floor** — when a vessel has near-zero pre-collision speed variance, any tiny real-world change produces an arbitrarily large Z-score. A floor of 0.5 was applied when the true std is undefined or zero.

---

## Porting to PySpark

The full 31-day dataset is ~16GB compressed. The core challenge was making this tractable without either running out of memory or spending hours on unnecessary computation.

### Per-Day Processing

The most important architectural decision was processing each daily CSV independently rather than loading all 31 days into a single global DataFrame. This was not the first approach — the initial PySpark version read all CSVs at once with `spark.read.csv("*.csv")` and ran window functions over the full global dataset. It was significantly slower and this is what drove the switch.

The reason: window functions like `lag()` (inter-ping GPS anomaly filter), `first()`/`last()` (vessel name fill), and SOG forward-fill all require co-locating all pings for each MMSI. Doing this globally across 31 files requires Spark to shuffle every MMSI's pings across the cluster — an expensive operation proportional to the full dataset size. Processing each day independently keeps all window functions entirely local to each day's partitions, eliminating the cross-file shuffle entirely.

A common suggested alternative is to read all files at once and partition by `(MMSI, Date)` rather than `MMSI` alone. This still requires a global shuffle to co-locate pings by MMSI+Date — it does not eliminate the cost, it just makes the partition key more granular. The per-day approach avoids the shuffle entirely by never needing cross-day MMSI co-location in the first place.

The 31 daily DataFrames are unioned lazily into a single logical plan and materialised exactly once via `cache().count()`.

### Geohash Port Filter

Vessels in port generate enormous numbers of false proximity pairs. The naive fix — a Python UDF with Shapely geometry per row — would serialize geometry objects across the network and call the Python interpreter for every one of 22 million pings. Instead, port zones were precomputed on the driver from OSM data (Denmark + Sweden PBF files), buffered by 700m, and enumerated into a frozenset of ~28,963 precision-7 geohash strings (~170m × 214m cells). Spark then filters via `.isin()` — a native JVM hash lookup, no Python overhead, no network transfer.

### Single Geohash Join

The original 9-merge bucketing approach would become 9 distributed shuffles in Spark. This was replaced with a single join: the right side of the join explodes each ping into 9 rows (itself + 8 geohash neighbours), enabling a single inner join on `(Time_Bucket, Geohash)` to replace all 9 shuffles.

### The toPandas() Bug

The first working PySpark version had a critical flaw: it collected the entire 22M row cleaned dataset to Pandas before filtering to suspect vessels. This caused memory crashes. The fix was to filter in Spark to just the suspect MMSIs first, then call `toPandas()` on only those few hundred vessels. The full dataset never touches the driver.

**Runtime.** On the development machine (11 cores, Apple M-series) the Phase 1 pipeline ran in ~18 minutes. The reproducible reference figure is the containerised run on a deliberately modest 4-core / 8GB Docker configuration: ~25 minutes for Phase 1 and ~26 minutes end-to-end (see *Computational Architecture* below and the README). Both are reported so the development and tested-container timings are not confused for one another.

---

## Phase 2 — Forensic Verification

Phase 1 produced 480 candidate pairs across 31 days. Phase 2 needed to reduce these to the actual collision(s).

### Speed

The initial Phase 2 scanned the full telemetry DataFrame on every candidate lookup — ~10 minutes for 480 pairs. Fixed by pre-indexing once at startup:
```python
vessel_index = {mmsi: group for mmsi, group in df.groupby('MMSI')}
```
This dropped runtime to under 30 seconds.

### Iterative False Positive Reduction

This was the most iterative part of the project. Starting from 16 confirmed collisions, we worked through the following reductions:

**"Both vessels must show trauma"** — requiring trauma on both vessels (rather than either) eliminated pilot boarding and tug escort false positives where only the manoeuvring small vessel showed a signal. Went from 16 to 4.

**ROT spike alone is insufficient** — a sharp turn is normal maritime behaviour. ROT spike now requires corroboration from at least one other signal.

**Catastrophic blackout alone is insufficient** — a vessel going silent could be a transponder fault. Blackout alone no longer confirms trauma.

**Post-window gap truncation** — discovered that PILOT 213 SE was generating a spurious Z=90.8 SOG shock from a 4-minute AIS blackout mid-window (disappeared at 7.9 knots, reappeared at 24.2 knots). The post window is now truncated at any gap >2 minutes.

**Dual-signal requirement** — the filter that had the largest effect on the candidate count. The reasoning: a single anomaly is weak evidence of a collision, since a vessel showing only a heading/COG skid may simply be turning intentionally, and a lone speed change can come from normal manoeuvring, whereas a physical impact tends to produce several anomalies at once. On that basis, "dual-signal trauma" was defined as a vessel exhibiting **at least two** of the four independent signals within the same impact window — SOG shock, heading/COG skid, ROT spike, or catastrophic blackout (`has_dual_signal()` counts the distinct signal types present and requires ≥2). The signals need not coincide on the same ping; each is detected over the ±5-minute window around the candidate impact, so two flags being set means both anomalies appear somewhere in that span rather than at the same instant.

How that requirement is applied depends on the **relative size** of the two vessels, because the physics is not symmetric:

- **Similar-sized vessels (symmetric).** *Both* vessels are required to independently show dual-signal trauma. The thinking is that if two comparable ships collide, both would be expected to register the impact, so two signals on each is the threshold applied.
- **Significantly different sizes (asymmetric, area ratio ≥2×).** Only the **smaller** vessel is required to show dual-signal trauma. The motivation is that a large ship may not detectably feel the impact of a small one — the small vessel can be shoved violently while the large one barely registers it — so requiring two signals on the large vessel risks discarding what may be a real collision.

This is what `trauma_required()` encodes: it returns `dual_a and dual_b` for symmetric pairs, but only the smaller vessel's `dual_*` when one vessel is larger. Size is determined by `is_asymmetric()`, which prefers a length×width area proxy, falls back to length alone, and finally to `Type of mobile` (Class A vs Class B) when dimensions are missing; vessel dimensions (Length, Width) are exported by Phase 1 so this comparison is available. In combination, the dual-signal rule and its size-aware application took the confirmed count from 9 to 1: the symmetric case removed single-signal false positives, while the asymmetric branch was intended to keep a genuine large-vs-small impact from being filtered out along with them.

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

### On the Choice of Thresholds and Generalisability

The verification logic depends on a small number of numerical thresholds. None of them is a value reverse-engineered to single out KARIN HØEJ; each is grounded in a statistical convention or a physical fact that holds independently of this dataset:

- **Z-score threshold of 3.0.** The trauma tests operate on quantities already normalised into Z-scores, and 3.0 is the standard three-sigma cutoff — roughly the 99.7th percentile of a normal distribution — for treating an observation as anomalous. It is a convention applied to a standardised scale, not a number chosen for this event; the observed trauma scores here (Z = 5.4 to 14.7) sit well above it, so the outcome does not hinge on the exact value.
- **Variance floor of 0.5.** This is not a tuned parameter but a guard against a mathematical pathology: when a vessel holds an almost perfectly steady course its pre-collision standard deviation approaches zero, and dividing by it would turn any trivial change into an arbitrarily large Z-score. The floor caps that division and would be required for any dataset.
- **50-knot teleport filter and 2-minute gap truncation.** Both reject GPS artefacts rather than encode anything about December 2021. 50 knots sits above any plausible vessel speed, so an implied speed beyond it marks a position jump rather than real movement; truncating the post-impact window at a multi-minute gap discards the same teleportation artefact in the time dimension (a vessel vanishing and reappearing elsewhere).
- **100 m proximity.** This is deliberately loose. Real collisions occur far closer than 100 m; the threshold is sized to absorb AIS positional jitter and ping inconsistency, not to define how near a collision is. It is a tolerance band, set generously so a genuine impact is not missed because two reported positions disagree by a few metres.
- **Size ratio of 2.0.** The most judgement-based of the set, but still a reasonable estimator for "materially different in size": a factor-of-two difference in area is a defensible line, and the asymmetry rule it gates is motivated by physics — a large ship may not detectably register striking a small one.

What *was* refined against the known collision is not these constants but the **selection of rules** layered on top of them — requiring dual signals, requiring corroboration for an ROT spike or a blackout, and the size-asymmetry exception. The iterative reduction from 16 candidates to 1 came from adding these physically-motivated rules with the ground truth visible. That is ordinary development rather than threshold-fitting, but it means the framework is not claimed to return exactly one collision for an arbitrary month. The same tests are applied uniformly to every pair, with no special-casing of the target vessels; fed a different month, the pipeline runs identical logic, and whether that yields zero, one, or several confirmed collisions would depend on that month's traffic.

---

## Computational Architecture — Design Rationale

This section explains the deliberate engineering choices behind how work is distributed, why the pipeline is structured the way it is, and how to read its runtime behaviour. The figures are from an end-to-end run on the full December 2021 dataset in a deliberately modest configuration: 4 CPU cores and 8GB RAM allocated to Docker, total runtime ~26 minutes.

### Where Spark Is Used, and Where Pandas Is

All large-scale data processing runs in PySpark. Reading the 31 daily CSVs, schema casting, every cleaning filter (MMSI validation, the 50nm radius filter, stationary-vessel and port exclusion, the `lag()`-based GPS-anomaly filter, and the SOG forward-fill), the geohash neighbourhood join, and the proximity ranking all execute as Spark operations over the full dataset — **22,464,519 rows after cleaning and filtering**. Pandas is used only *after* Spark has reduced the problem to a trivially small size: the 480-candidate summary and the few-hundred-vessel suspect telemetry. Phase 2's forensic verification and Phase 3's visualization then operate on sub-thousand-row inputs.

This division follows the common big-data pattern of distributing the work that is too large for one machine and collecting to the driver only once the data is small. (The `toPandas()` fix described above keeps to this boundary — the full 22.5M-row dataset never reaches the driver.) Reserving Pandas for the small post-Spark steps avoids putting distributed-execution overhead on sub-thousand-row data, which would add cost without benefit. Measured by data processed, the large majority of the work is in Spark — the Pandas stages run on kilobytes.

### Parallel Execution vs. Sequenced Stages

Two senses of "sequential" are worth distinguishing, because only one applies.

The 31 daily files are **not** processed one after another. `load_and_clean_all` loops over the files, but each `clean_one_day` call returns a *lazy* DataFrame — nothing executes during the loop. The 31 plans are unioned into one logical plan and materialised by a single action (`cache().count()`). At that point Spark schedules one job spanning all 31 days' work and runs it in parallel across all allocated cores. The progress lines in the run logs confirm this: entries such as `(4 + 4) / 14` show four tasks executing concurrently on the four cores. The per-day structure (detailed above) exists to keep window functions local to each day's partitions and avoid a cross-file shuffle — it is not serial execution.

What *is* sequenced is the chain of operations — clean, then join, then rank — because each stage depends on the previous stage's output. A join cannot run before the geohash column it joins on exists; candidates cannot be ranked before they are found. This is ordinary data-dependency, the directed acyclic graph of stages that every pipeline has. Within each stage the work is parallel; the stages run in order because they must. Likewise, the three phases (detection → verification → visualization) run sequentially because each consumes the prior phase's output files — a pipeline, not a missed parallelisation.

### Interpreting CPU Utilisation

For parts of the run, observed CPU sits near one core rather than saturating all four. This reflects two causes, neither of which is a parallelisation failure.

First, several stages are inherently single-threaded driver work that sits outside Spark's parallel engine: building the port-exclusion set from the OpenStreetMap PBF files (`osmium` parsing plus Shapely geometry, ~2 minutes) and extracting the 31 CSVs from the source ZIP (`zipfile`, ~3 minutes) both run on a single thread by nature. Roughly five minutes of wall-clock time is therefore single-core regardless of how many cores are available.

Second, even within Spark, not every stage is CPU-bound. Reading ~16GB of CSV from disk and writing shuffle data between stages are I/O-bound; cores wait on disk rather than compute, so low CPU there indicates the bottleneck is I/O, not idle inefficiency. Running a distributed engine in local mode on a single machine with one disk makes this expected. The CPU-bound transformation stages do parallelise across all allocated cores, as the concurrent-task progress lines show.

### Why the Cleaned Dataset Is Cached Whole (and Not Batched)

The pipeline caches the filtered 22.5M-row dataset once and reuses it for both the collision join and the suspect-telemetry export, avoiding recomputation of the cleaning DAG. Caching the whole set was chosen with scale in mind. At one month, those rows fit within 8GB alongside the Spark driver — the pipeline completes and returns the correct collision — so a single cached plan was a straightforward way to feed every downstream step from one materialisation.

A natural alternative for larger inputs is batched processing: handling a subset of days at a time, writing each batch's candidates and telemetry to disk, then concatenating, so peak memory tracks one batch rather than the whole timeframe. It was not used here. At the assignment's scale it would lower a peak-memory figure that is not currently a constraint — the 8GB run already succeeds — while adding complexity. A batch implementation would pay Spark's startup and OSM-parsing cost once and run at roughly the same speed, but it brings batch-boundary bookkeeping: accumulating the global suspect-MMSI set across batches before writing telemetry, and carrying ping overlap across batch seams (otherwise the midnight-boundary limitation noted below would recur at every seam). For a substantially longer timeframe — a full year, or a widened radius — the filtered set could exceed memory and the trade-off would shift toward batching, or toward a larger cluster. The per-day design lends itself to that extension, since each day is already cleaned independently with no cross-day shuffle, so a batched loop would mainly be a change to driver orchestration rather than to the per-day transformation logic.

---

## Known Limitations

**Midnight boundary gap** — per-day processing cannot detect collisions whose closest pings straddle day boundaries. Given AIS pings every 2–10 seconds, this is an astronomically unlikely scenario for any real collision, and the KARIN HØEJ collision at 02:27 is safely within a single day.

**AIS self-reported data** — vessel dimensions and ship type codes are crew-reported and may be missing or incorrect. The asymmetric trauma logic degrades gracefully when data is absent.