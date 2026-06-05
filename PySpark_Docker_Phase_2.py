import pandas as pd
import numpy as np
from shapely.geometry import LineString
import warnings
import os

warnings.filterwarnings('ignore')

# Output directory: set via OUTPUT_DIR env variable in Docker, defaults to '.' locally
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '.')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CLEAN_TELEMETRY_CSV   = os.path.join(OUTPUT_DIR, "clean_suspect_telemetry.csv")
CANDIDATES_CSV        = os.path.join(OUTPUT_DIR, "suspected_collisions_list.csv")
TRAUMA_WINDOW_MINUTES = 5

# Fallback std used when pre-collision variance is undefined (fewer than 2 pings)
# or exactly zero (perfectly steady course). Without this floor, trivially small
# post-collision changes on a rock-steady vessel produce arbitrarily large Z-scores.
VARIANCE_FLOOR    = 0.5
Z_SCORE_THRESHOLD = 3.0

# Area ratio above which two vessels are considered asymmetric. When one vessel
# is more than 2x larger by area, only the smaller vessel is required to show
# dual-signal trauma — a large ship may not detectably feel a small vessel impact.
SIZE_RATIO_THRESHOLD = 2.0


# Vessel size helpers

def get_vessel_size(vessel_index, mmsi):
    # Returns (length, width, type_of_mobile) from the first non-null values
    # in the vessel's telemetry. Falls back to None for missing fields.
    df = vessel_index.get(mmsi, pd.DataFrame())
    if df.empty:
        return None, None, None

    length = None
    width  = None
    mobile = None

    if 'Length' in df.columns:
        vals = df['Length'].dropna()
        if not vals.empty:
            length = float(vals.iloc[0])

    if 'Width' in df.columns:
        vals = df['Width'].dropna()
        if not vals.empty:
            width = float(vals.iloc[0])

    if 'Type of mobile' in df.columns:
        vals = df['Type of mobile'].dropna()
        if not vals.empty:
            mobile = str(vals.iloc[0]).strip()

    return length, width, mobile


def is_asymmetric(vessel_index, mmsi_a, mmsi_b):
    # Determines size asymmetry between two vessels.
    # Priority: length+width area proxy > length only > Type of mobile (Class A/B).
    # Falls back to 'symmetric' if no data is available.
    len_a, wid_a, mob_a = get_vessel_size(vessel_index, mmsi_a)
    len_b, wid_b, mob_b = get_vessel_size(vessel_index, mmsi_b)

    if len_a and wid_a and len_b and wid_b and len_a > 0 and len_b > 0:
        area_a = len_a * wid_a
        area_b = len_b * wid_b
        if area_a > 0 and area_b > 0:
            ratio = max(area_a, area_b) / min(area_a, area_b)
            if ratio >= SIZE_RATIO_THRESHOLD:
                return 'A_larger' if area_a > area_b else 'B_larger'
        return 'symmetric'

    if len_a and len_b and len_a > 0 and len_b > 0:
        ratio = max(len_a, len_b) / min(len_a, len_b)
        if ratio >= SIZE_RATIO_THRESHOLD:
            return 'A_larger' if len_a > len_b else 'B_larger'
        return 'symmetric'

    if mob_a and mob_b:
        a_class_a = 'Class A' in mob_a
        b_class_a = 'Class A' in mob_b
        if a_class_a and not b_class_a:
            return 'A_larger'
        if b_class_a and not a_class_a:
            return 'B_larger'

    return 'symmetric'


def has_dual_signal(flags):
    # A real collision produces multiple simultaneous physical anomalies.
    # A single signal (skid alone, SOG drop alone) can result from normal
    # manoeuvring and is insufficient on its own.
    has_sog   = any(f.startswith('SOG Shock') for f in flags)
    has_skid  = any(f.startswith('Heading/COG Skid') for f in flags)
    has_rot   = any(f.startswith('ROT Spike') for f in flags)
    has_black = 'Catastrophic Blackout' in flags
    return sum([has_sog, has_skid, has_rot, has_black]) >= 2


def trauma_required(vessel_index, mmsi_a, mmsi_b, trauma_a, trauma_b, flags_a, flags_b):
    # Applies the trauma requirement accounting for vessel size asymmetry.
    # Symmetric vessels: both must show dual-signal trauma.
    # Asymmetric: only the smaller vessel must show dual-signal trauma.
    asymmetry = is_asymmetric(vessel_index, mmsi_a, mmsi_b)
    dual_a = has_dual_signal(flags_a)
    dual_b = has_dual_signal(flags_b)

    if asymmetry == 'A_larger':
        return dual_b
    elif asymmetry == 'B_larger':
        return dual_a
    else:
        return dual_a and dual_b


# Kinematic intersection check
#
# Attempts to construct track geometries from pings within a time window
# around the reported impact. Starts at ±30s and expands to ±300s, stopping
# at the first window where both vessels have at least 2 pings. Uses
# pre-indexed vessel data for O(1) lookup rather than scanning the full dataframe.

def check_kinematic_intersection(vessel_index, mmsi_a, mmsi_b, t_impact):
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


# Physical trauma detection
#
# Three independent Z-score tests on the pre/post collision windows (±5 min):
#
# SOG Shock — significant speed change. Uses abs() to catch both deceleration
#   and acceleration (KARIN HOEJ was shoved forward from 6.1 to 10.3 knots).
#
# Heading/COG Skid — divergence between magnetic heading and course over ground,
#   indicating involuntary yaw or loss of steerage. Heading 511 (AIS sentinel
#   for "unavailable") is excluded.
#
# ROT Spike — abnormal rate of turn. Requires corroboration from at least one
#   other signal since a sharp turn is normal maritime behaviour on its own.
#
# Catastrophic Blackout — vessel goes silent post-impact. Requires corroboration
#   since transponder failures and crew actions also cause silence.
#
# Post-window gap truncation: if post-collision pings show a gap >2 minutes,
# the window is truncated there. A vessel reappearing after a gap at a different
# speed is a transponder artefact, not a collision signature.

def check_physical_trauma(vessel_index, mmsi, t_impact):
    df = vessel_index.get(mmsi, pd.DataFrame())
    if df.empty:
        return False, []

    pre_start = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
    post_end  = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)

    pre  = df[df['Timestamp'].between(pre_start, t_impact)]
    post = df[df['Timestamp'].between(t_impact, post_end)].sort_values('Timestamp')

    trauma_flags = []

    if post.empty:
        trauma_flags.append("Catastrophic Blackout")

    if len(post) > 1:
        post = post.reset_index(drop=True)
        post_gaps = post['Timestamp'].diff().dt.total_seconds()
        big_gap_positions = post_gaps[post_gaps > 120].index
        if len(big_gap_positions) > 0:
            post = post.loc[:big_gap_positions[0] - 1]

    if not pre.empty:
        pre_sog_mean  = pre['SOG'].mean()
        post_sog_mean = post['SOG'].mean()

        if pd.notna(pre_sog_mean) and pd.notna(post_sog_mean):
            pre_sog_std = pre['SOG'].std()
            if pd.isna(pre_sog_std) or pre_sog_std == 0:
                pre_sog_std = VARIANCE_FLOOR
            speed_change_z = abs(pre_sog_mean - post_sog_mean) / pre_sog_std
            if speed_change_z > Z_SCORE_THRESHOLD:
                trauma_flags.append(f"SOG Shock (Z={speed_change_z:.1f})")

        if all(c in pre.columns for c in ['Heading', 'COG']) and \
           all(c in post.columns for c in ['Heading', 'COG']):
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

    if len(trauma_flags) == 1 and trauma_flags[0].startswith('ROT Spike'):
        trauma_flags = []

    if trauma_flags == ['Catastrophic Blackout']:
        trauma_flags = []

    return len(trauma_flags) > 0, trauma_flags


# Main forensics loop

def run_phase2_forensics():
    print("=" * 65)
    print("  AIS COLLISION DETECTION — FORENSIC VERIFICATION")
    print("=" * 65)

    if not os.path.exists(CLEAN_TELEMETRY_CSV) and not os.path.isdir(CLEAN_TELEMETRY_CSV):
        print(f"  Error: {CLEAN_TELEMETRY_CSV} not found. Run Phase 1 first.")
        return

    try:
        candidates = pd.read_csv(CANDIDATES_CSV)
        candidates['Timestamp_A'] = pd.to_datetime(candidates['Timestamp_A'])
        import glob as _glob
        _parts = _glob.glob(os.path.join(CLEAN_TELEMETRY_CSV, 'part-*.csv'))
        df = pd.read_csv(_parts[0] if _parts else CLEAN_TELEMETRY_CSV)
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.replace(tzinfo=None))
        df['MMSI'] = df['MMSI'].astype(str).str.strip()
    except Exception as e:
        print(f"  Error loading files: {e}")
        return

    # Pre-index by MMSI so each candidate lookup is O(1) dictionary access
    # followed by a timestamp filter on a small per-vessel slice, rather than
    # scanning the full dataframe. Reduces runtime from ~10 min to <30 sec.
    vessel_index = {
        mmsi: group.reset_index(drop=True)
        for mmsi, group in df.groupby('MMSI')
    }

    print(f"\n  Running forensic checks on {len(candidates)} candidates...\n")

    confirmed_crashes = []

    for _, row in candidates.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a, mmsi_b = str(row['MMSI_A']), str(row['MMSI_B'])
        name_a, name_b = row['Name_A'], row['Name_B']

        intersected, intersect_msg = check_kinematic_intersection(
            vessel_index, mmsi_a, mmsi_b, t_impact
        )
        trauma_a, flags_a = check_physical_trauma(vessel_index, mmsi_a, t_impact)
        trauma_b, flags_b = check_physical_trauma(vessel_index, mmsi_b, t_impact)

        confirmed = intersected and trauma_required(
            vessel_index, mmsi_a, mmsi_b, trauma_a, trauma_b, flags_a, flags_b
        )

        if confirmed:
            confirmed_crashes.append(row)

            start_window = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
            end_window   = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)

            incident_telemetry = pd.concat([
                vessel_index.get(mmsi_a, pd.DataFrame()),
                vessel_index.get(mmsi_b, pd.DataFrame()),
            ])
            incident_telemetry = incident_telemetry[
                incident_telemetry['Timestamp'].between(start_window, end_window)
            ]

            cols_to_keep = ['Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude',
                            'SOG', 'COG', 'Heading', 'ROT', 'Navigational status',
                            'Type of mobile', 'Ship type', 'Length', 'Width']
            existing_cols = [c for c in cols_to_keep if c in incident_telemetry.columns]
            incident_telemetry = incident_telemetry[existing_cols].sort_values(
                by=['MMSI', 'Timestamp']
            )

            safe_a = str(name_a).replace(' ', '_').replace('/', '')
            safe_b = str(name_b).replace(' ', '_').replace('/', '')
            filename = os.path.join(OUTPUT_DIR, f"telemetry_{safe_a}_vs_{safe_b}.csv")
            incident_telemetry.to_csv(filename, index=False)

    if confirmed_crashes:
        final_df = pd.DataFrame(confirmed_crashes)
        final_path = os.path.join(OUTPUT_DIR, "FINAL_investigation_results.csv")
        final_df.to_csv(final_path, index=False)

        print(f"\n  {len(final_df)} confirmed collision(s):\n")
        for _, row in final_df.iterrows():
            print(f"    {row['Name_A']} vs {row['Name_B']} — "
                  f"{row['Distance_Meters']:.1f}m apart @ {row['Timestamp_A']}")

        print(f"\n  Output files:")
        print(f"    {final_path}")
        for _, row in final_df.iterrows():
            safe_a = str(row['Name_A']).replace(' ', '_').replace('/', '')
            safe_b = str(row['Name_B']).replace(' ', '_').replace('/', '')
            print(f"    {os.path.join(OUTPUT_DIR, f'telemetry_{safe_a}_vs_{safe_b}.csv')}")
    else:
        print("\n  No collisions confirmed.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run_phase2_forensics()