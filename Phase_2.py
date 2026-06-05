import pandas as pd
import numpy as np
from shapely.geometry import LineString
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CLEAN_TELEMETRY_CSV = "clean_suspect_telemetry.csv"
CANDIDATES_CSV = "suspected_collisions_list.csv"
TRAUMA_WINDOW_MINUTES = 5

# Fallback standard deviation used when the true std is undefined or zero.
# Applies in exactly two cases:
#   1. Fewer than 2 pre-collision pings exist (std is NaN — mathematically undefined)
#   2. All pre-collision readings are identical (std is exactly 0 — division by zero)
# In all other cases the real std is used regardless of how small it is.
VARIANCE_FLOOR = 0.5

# Z-score threshold for flagging anomalous post-collision behaviour.
# 3.0 is the standard statistical threshold for outlier detection (3-sigma rule).
Z_SCORE_THRESHOLD = 3.0


def check_kinematic_intersection(vessel_index, mmsi_a, mmsi_b, t_impact):
    """
    Short-Circuiting Kinematic Filter.
    Uses pre-indexed vessel data (vessel_index) for O(1) MMSI lookup
    instead of scanning the full dataframe on every call.
    """
    windows = [(30, "30"), (60, "60"), (120, "120"), (300, "300")]

    df_a = vessel_index.get(mmsi_a, pd.DataFrame())
    df_b = vessel_index.get(mmsi_b, pd.DataFrame())

    for sec, label in windows:
        t_start = t_impact - pd.Timedelta(seconds=sec)
        t_end   = t_impact + pd.Timedelta(seconds=sec)

        track_a = df_a[df_a['Timestamp'].between(t_start, t_end)].sort_values('Timestamp')
        track_b = df_b[df_b['Timestamp'].between(t_start, t_end)].sort_values('Timestamp')

        if len(track_a) >= 2 and len(track_b) >= 2:
            line_a = LineString(zip(track_a['Longitude'], track_a['Latitude']))
            line_b = LineString(zip(track_b['Longitude'], track_b['Latitude']))

            if line_a.intersects(line_b):
                return True, label
            else:
                return False, f"Paths did not cross during active {label}s window"

    return False, "Insufficient pings to form paths"


def check_physical_trauma(vessel_index, mmsi, t_impact):
    """
    Pure Z-Score Anomaly Detection.
    Uses pre-indexed vessel data for O(1) MMSI lookup.
    """
    df = vessel_index.get(mmsi, pd.DataFrame())
    if df.empty:
        return False, []

    pre_start = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
    post_end  = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)

    pre  = df[df['Timestamp'].between(pre_start, t_impact)]
    post = df[df['Timestamp'].between(t_impact, post_end)]

    trauma_flags = []

    # 1. Blackout Check (Sinking / Complete Transponder Loss)
    if post.empty:
        trauma_flags.append("Catastrophic Blackout")
        return True, trauma_flags

    if not pre.empty:
        # --- TEST A: SOG Shock ---
        pre_sog_mean  = pre['SOG'].mean()
        post_sog_mean = post['SOG'].mean()

        if pd.notna(pre_sog_mean) and pd.notna(post_sog_mean):
            pre_sog_std = pre['SOG'].std()
            if pd.isna(pre_sog_std) or pre_sog_std == 0:
                pre_sog_std = VARIANCE_FLOOR
            speed_change_z = abs(pre_sog_mean - post_sog_mean) / pre_sog_std
            if speed_change_z > Z_SCORE_THRESHOLD:
                trauma_flags.append(f"SOG Shock (Z={speed_change_z:.1f})")

        # --- TEST B: Heading/COG Skid ---
        if all(c in pre.columns for c in ['Heading', 'COG']) and \
           all(c in post.columns for c in ['Heading', 'COG']):
            # 511 is the NMEA/AIS standard placeholder for "heading unavailable".
            pre_hdg_valid  = pre[pre['Heading'] != 511]
            post_hdg_valid = post[post['Heading'] != 511]

            if not pre_hdg_valid.empty and not post_hdg_valid.empty:
                pre_diff  = abs(pre_hdg_valid['Heading'] - pre_hdg_valid['COG']) % 360
                pre_skid  = np.minimum(pre_diff, 360 - pre_diff)
                pre_skid_std = pre_skid.std()
                if pd.isna(pre_skid_std) or pre_skid_std == 0:
                    pre_skid_std = VARIANCE_FLOOR
                post_diff = abs(post_hdg_valid['Heading'] - post_hdg_valid['COG']) % 360
                post_skid = np.minimum(post_diff, 360 - post_diff)
                skid_z = (post_skid.max() - pre_skid.mean()) / pre_skid_std
                if skid_z > Z_SCORE_THRESHOLD:
                    trauma_flags.append(f"Heading/COG Skid (Z={skid_z:.1f})")

        # --- TEST C: Involuntary Spin (ROT) ---
        if 'ROT' in pre.columns and 'ROT' in post.columns:
            pre_rot  = pre['ROT'].dropna().abs()
            post_rot = post['ROT'].dropna().abs()
            if not pre_rot.empty and not post_rot.empty:
                pre_rot_std = pre_rot.std()
                if pd.isna(pre_rot_std) or pre_rot_std == 0:
                    pre_rot_std = VARIANCE_FLOOR
                rot_z = (post_rot.max() - pre_rot.mean()) / pre_rot_std
                if rot_z > Z_SCORE_THRESHOLD:
                    trauma_flags.append(f"ROT Spike (Z={rot_z:.1f})")

    return len(trauma_flags) > 0, trauma_flags


def run_phase2_forensics():
    print("Loading Surgical Dataset for Phase 2 Verification...")
    if not os.path.exists(CLEAN_TELEMETRY_CSV):
        print(f"❌ Error: '{CLEAN_TELEMETRY_CSV}' not found. Please run Phase 1 first.")
        return

    try:
        candidates = pd.read_csv(CANDIDATES_CSV)
        candidates['Timestamp_A'] = pd.to_datetime(candidates['Timestamp_A'])

        df = pd.read_csv(CLEAN_TELEMETRY_CSV)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['MMSI'] = df['MMSI'].astype(str).str.strip()

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # -------------------------------------------------------------------------
    # PRE-INDEX BY MMSI
    # Instead of scanning the full dataframe on every candidate pair lookup,
    # build a dict of {mmsi -> vessel_df} once. All subsequent lookups are O(1)
    # dictionary access followed by a tiny timestamp filter on a per-vessel slice.
    # This reduces Phase 2 runtime from ~10 minutes to under a minute.
    # -------------------------------------------------------------------------
    print("Building MMSI index for fast lookups...")
    vessel_index = {
        mmsi: group.reset_index(drop=True)
        for mmsi, group in df.groupby('MMSI')
    }
    print(f"Indexed {len(vessel_index)} vessels.\n")

    confirmed_crashes = []

    print(f"Initiating Kinematic & Physics Forensics on {len(candidates)} Suspects...\n")
    print("-" * 80)

    for _, row in candidates.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a, mmsi_b = str(row['MMSI_A']), str(row['MMSI_B'])
        name_a, name_b = row['Name_A'], row['Name_B']

        intersected, intersect_msg = check_kinematic_intersection(
            vessel_index, mmsi_a, mmsi_b, t_impact
        )
        trauma_a, flags_a = check_physical_trauma(vessel_index, mmsi_a, t_impact)
        trauma_b, flags_b = check_physical_trauma(vessel_index, mmsi_b, t_impact)

        if intersected and trauma_a and trauma_b:
            confirmed_crashes.append(row)
            print(f"🚨 CONFIRMED COLLISION: {name_a} vs {name_b} at {t_impact}")
            print(f"   -> Kinematics: Paths crossed within +/- {intersect_msg} seconds")
            if trauma_a: print(f"   -> Trauma {name_a}: {', '.join(flags_a)}")
            if trauma_b: print(f"   -> Trauma {name_b}: {', '.join(flags_b)}")

            start_window = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
            end_window   = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)

            # Use vessel_index for the telemetry export too
            telemetry_a = vessel_index.get(mmsi_a, pd.DataFrame())
            telemetry_b = vessel_index.get(mmsi_b, pd.DataFrame())

            incident_telemetry = pd.concat([
                telemetry_a[telemetry_a['Timestamp'].between(start_window, end_window)],
                telemetry_b[telemetry_b['Timestamp'].between(start_window, end_window)],
            ])

            cols_to_keep = ['Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude',
                            'SOG', 'COG', 'Heading', 'ROT', 'Navigational status']
            existing_cols = [c for c in cols_to_keep if c in incident_telemetry.columns]
            incident_telemetry = incident_telemetry[existing_cols].sort_values(
                by=['MMSI', 'Timestamp']
            )

            safe_name_a = str(name_a).replace(' ', '_').replace('/', '')
            safe_name_b = str(name_b).replace(' ', '_').replace('/', '')
            filename = f"telemetry_{safe_name_a}_vs_{safe_name_b}.csv"

            incident_telemetry.to_csv(filename, index=False)
            print(f"   -> 💾 Exported +/- {TRAUMA_WINDOW_MINUTES} min telemetry to '{filename}'")
            print("-" * 80)
        else:
            reason = "Paths did not cross" if not intersected else "No physical trauma detected"
            if "Insufficient pings" in intersect_msg or "active" in intersect_msg:
                reason = intersect_msg
            print(f"✅ Cleared: {name_a} vs {name_b} ({reason})")

    if confirmed_crashes:
        final_df = pd.DataFrame(confirmed_crashes)
        final_df.to_csv("FINAL_investigation_results.csv", index=False)
        print(f"\nInvestigation Complete. {len(final_df)} confirmed collision(s) saved to FINAL_investigation_results.csv")
    else:
        print("\nInvestigation Complete. No collisions confirmed.")


if __name__ == "__main__":
    run_phase2_forensics()