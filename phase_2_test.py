import pandas as pd
import numpy as np
from shapely.geometry import LineString
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CLEAN_TELEMETRY_CSV = "clean_suspect_telemetry.csv"
CANDIDATES_CSV = "suspected_collisions_list.csv"
TRAUMA_WINDOW_MINUTES = 5


def check_kinematic_intersection(df, mmsi_a, mmsi_b, t_impact):
    windows = [(30, "30"), (60, "60"), (120, "120"), (300, "300")]
    for sec, label in windows:
        t_start = t_impact - pd.Timedelta(seconds=sec)
        t_end   = t_impact + pd.Timedelta(seconds=sec)
        track_a = df[(df['MMSI'] == mmsi_a) & (df['Timestamp'].between(t_start, t_end))].sort_values('Timestamp')
        track_b = df[(df['MMSI'] == mmsi_b) & (df['Timestamp'].between(t_start, t_end))].sort_values('Timestamp')
        if len(track_a) >= 2 and len(track_b) >= 2:
            line_a = LineString(zip(track_a['Longitude'], track_a['Latitude']))
            line_b = LineString(zip(track_b['Longitude'], track_b['Latitude']))
            if line_a.intersects(line_b):
                return True, label
            else:
                return False, f"Paths did not cross during active {label}s window"
    return False, "Insufficient pings to form paths"


def check_physical_trauma(df, mmsi, t_impact, variance_floor, sog_speed_guard):
    pre_start = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
    post_end  = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)

    pre  = df[(df['MMSI'] == mmsi) & (df['Timestamp'].between(pre_start, t_impact))]
    post = df[(df['MMSI'] == mmsi) & (df['Timestamp'].between(t_impact, post_end))]

    trauma_flags = []

    if post.empty:
        trauma_flags.append("Catastrophic Blackout")
        return True, trauma_flags

    if not pre.empty:
        # --- TEST A: SOG Shock ---
        pre_sog_mean  = pre['SOG'].mean()
        post_sog_mean = post['SOG'].mean()
        if pd.notna(pre_sog_mean) and pd.notna(post_sog_mean):
            pre_sog_std = pre['SOG'].std()
            if pd.isna(pre_sog_std) or pre_sog_std < variance_floor:
                pre_sog_std = variance_floor
            speed_drop_z = (pre_sog_mean - post_sog_mean) / pre_sog_std
            if speed_drop_z > 3.0:
                trauma_flags.append(f"SOG Shock (Z={speed_drop_z:.1f})")

        # --- TEST B: Heading/COG Skid ---
        if all(c in pre.columns for c in ['Heading', 'COG']) and \
           all(c in post.columns for c in ['Heading', 'COG']):
            # Apply SOG speed guard if configured — filters out low-speed pings
            # where COG becomes unreliable due to GPS drift
            if sog_speed_guard is not None:
                pre_hdg_valid  = pre[(pre['Heading'] != 511) & (pre['SOG'] > sog_speed_guard)]
                post_hdg_valid = post[(post['Heading'] != 511) & (post['SOG'] > sog_speed_guard)]
            else:
                pre_hdg_valid  = pre[pre['Heading'] != 511]
                post_hdg_valid = post[post['Heading'] != 511]

            if not pre_hdg_valid.empty and not post_hdg_valid.empty:
                pre_diff  = abs(pre_hdg_valid['Heading'] - pre_hdg_valid['COG']) % 360
                pre_skid  = np.minimum(pre_diff, 360 - pre_diff)
                pre_skid_std = pre_skid.std()
                if pd.isna(pre_skid_std) or pre_skid_std < variance_floor:
                    pre_skid_std = variance_floor
                post_diff = abs(post_hdg_valid['Heading'] - post_hdg_valid['COG']) % 360
                post_skid = np.minimum(post_diff, 360 - post_diff)
                skid_z = (post_skid.max() - pre_skid.mean()) / pre_skid_std
                if skid_z > 3.0:
                    trauma_flags.append(f"Heading/COG Skid (Z={skid_z:.1f})")

        # --- TEST C: ROT Spin ---
        if 'ROT' in pre.columns and 'ROT' in post.columns:
            pre_rot  = pre['ROT'].dropna().abs()
            post_rot = post['ROT'].dropna().abs()
            if not pre_rot.empty and not post_rot.empty:
                pre_rot_std = pre_rot.std()
                if pd.isna(pre_rot_std) or pre_rot_std < variance_floor:
                    pre_rot_std = variance_floor
                rot_z = (post_rot.max() - pre_rot.mean()) / pre_rot_std
                if rot_z > 3.0:
                    trauma_flags.append(f"ROT Spike (Z={rot_z:.1f})")

    return len(trauma_flags) > 0, trauma_flags


def run_with_config(df, candidates, label, variance_floor, sog_speed_guard):
    confirmed = []
    for _, row in candidates.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a, mmsi_b = str(row['MMSI_A']), str(row['MMSI_B'])
        name_a, name_b = row['Name_A'], row['Name_B']

        intersected, _ = check_kinematic_intersection(df, mmsi_a, mmsi_b, t_impact)
        trauma_a, flags_a = check_physical_trauma(df, mmsi_a, t_impact, variance_floor, sog_speed_guard)
        trauma_b, flags_b = check_physical_trauma(df, mmsi_b, t_impact, variance_floor, sog_speed_guard)

        if intersected and (trauma_a or trauma_b):
            confirmed.append({
                'MMSI_A': mmsi_a, 'MMSI_B': mmsi_b,
                'Name_A': name_a, 'Name_B': name_b,
                'Timestamp': t_impact,
                'Flags_A': ', '.join(flags_a) if flags_a else '—',
                'Flags_B': ', '.join(flags_b) if flags_b else '—',
            })

    guard_str = f"ON (> {sog_speed_guard} knots)" if sog_speed_guard is not None else "OFF"
    print(f"\n{'=' * 60}")
    print(f"CONFIG: {label}")
    print(f"  Variance floor:  {variance_floor}")
    print(f"  SOG speed guard: {guard_str}")
    print(f"  Confirmed collisions: {len(confirmed)}")
    for c in confirmed:
        print(f"  🚨 {c['Name_A']} vs {c['Name_B']} @ {c['Timestamp']}")
        print(f"     Flags A: {c['Flags_A']}")
        print(f"     Flags B: {c['Flags_B']}")
    print(f"{'=' * 60}")
    return set((c['MMSI_A'], c['MMSI_B']) for c in confirmed)


if __name__ == "__main__":
    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates['Timestamp_A'] = pd.to_datetime(candidates['Timestamp_A'])

    df = pd.read_csv(CLEAN_TELEMETRY_CSV)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['MMSI'] = df['MMSI'].astype(str).str.strip()

    # --- CONFIG 1: Original ---
    c1 = run_with_config(df, candidates,
        label="ORIGINAL (mixed floors, SOG guard ON @ 2.0)",
        variance_floor=0.5, sog_speed_guard=2.0)

    # --- CONFIG 2: Uniform floor, no SOG mean guard ---
    c2 = run_with_config(df, candidates,
        label="Uniform floor=0.5, SOG guard OFF",
        variance_floor=0.5, sog_speed_guard=None)

    # --- Diff ---
    print("\n--- DIFF: Original vs Proposed ---")
    gained = c2 - c1
    lost   = c1 - c2
    if not gained and not lost:
        print("✅ Identical results. Safe to remove SOG speed guard.")
    if gained:
        print(f"⚠️  New false positives introduced: {gained}")
    if lost:
        print(f"❌ Collisions lost: {lost}")