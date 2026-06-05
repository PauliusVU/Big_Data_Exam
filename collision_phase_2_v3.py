import zipfile
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
ZIP_PATH = "data/aisdk-2021-12.zip"
TARGET_CSV = "aisdk-2021-12-13.csv"
CANDIDATES_CSV = "suspected_collisions_list.csv"
TRAUMA_WINDOW_MINUTES = 5

def check_kinematic_intersection(df, mmsi_a, mmsi_b, t_impact):
    """
    Checks if the actual paths of the ships crossed within the specified window of the alert.
    Progressively scales from 30s -> 60s -> 2m (120s) -> 5m (300s) to combat sparse ping data.
    """
    # --- TIER 1: Primary 30-Second Window (+/- 30s) ---
    t_start_30 = t_impact - pd.Timedelta(seconds=30)
    t_end_30 = t_impact + pd.Timedelta(seconds=30)
    
    track_a_30 = df[(df['MMSI'] == mmsi_a) & (df['Timestamp'].between(t_start_30, t_end_30))]
    track_b_30 = df[(df['MMSI'] == mmsi_b) & (df['Timestamp'].between(t_start_30, t_end_30))]
    
    if len(track_a_30) >= 2 and len(track_b_30) >= 2:
        line_a = LineString(zip(track_a_30['Longitude'], track_a_30['Latitude']))
        line_b = LineString(zip(track_b_30['Longitude'], track_b_30['Latitude']))
        return line_a.intersects(line_b), "30"
    
    # --- TIER 2: 60-Second Fallback Window (+/- 60s) ---
    t_start_60 = t_impact - pd.Timedelta(seconds=60)
    t_end_60 = t_impact + pd.Timedelta(seconds=60)
    
    track_a_60 = df[(df['MMSI'] == mmsi_a) & (df['Timestamp'].between(t_start_60, t_end_60))]
    track_b_60 = df[(df['MMSI'] == mmsi_b) & (df['Timestamp'].between(t_start_60, t_end_60))]
    
    if len(track_a_60) >= 2 and len(track_b_60) >= 2:
        line_a = LineString(zip(track_a_60['Longitude'], track_a_60['Latitude']))
        line_b = LineString(zip(track_b_60['Longitude'], track_b_60['Latitude']))
        return line_a.intersects(line_b), "60"

    # --- TIER 3: 2-Minute Fallback Window (+/- 120s) ---
    t_start_120 = t_impact - pd.Timedelta(seconds=120)
    t_end_120 = t_impact + pd.Timedelta(seconds=120)
    
    track_a_120 = df[(df['MMSI'] == mmsi_a) & (df['Timestamp'].between(t_start_120, t_end_120))]
    track_b_120 = df[(df['MMSI'] == mmsi_b) & (df['Timestamp'].between(t_start_120, t_end_120))]
    
    if len(track_a_120) >= 2 and len(track_b_120) >= 2:
        line_a = LineString(zip(track_a_120['Longitude'], track_a_120['Latitude']))
        line_b = LineString(zip(track_b_120['Longitude'], track_b_120['Latitude']))
        return line_a.intersects(line_b), "120"

    # --- TIER 4: 5-Minute Maximum Fallback Window (+/- 300s) ---
    t_start_300 = t_impact - pd.Timedelta(seconds=300)
    t_end_300 = t_impact + pd.Timedelta(seconds=300)
    
    track_a_300 = df[(df['MMSI'] == mmsi_a) & (df['Timestamp'].between(t_start_300, t_end_300))]
    track_b_300 = df[(df['MMSI'] == mmsi_b) & (df['Timestamp'].between(t_start_300, t_end_300))]
    
    if len(track_a_300) >= 2 and len(track_b_300) >= 2:
        line_a = LineString(zip(track_a_300['Longitude'], track_a_300['Latitude']))
        line_b = LineString(zip(track_b_300['Longitude'], track_b_300['Latitude']))
        return line_a.intersects(line_b), "300"
    else:
        return False, "Insufficient pings in 5m window to form lines"

def check_physical_trauma(df, mmsi, t_impact):
    """Checks for catastrophic anomalies entirely via Statistical Z-Scores."""
    pre_start = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
    post_end = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
    
    pre = df[(df['MMSI'] == mmsi) & (df['Timestamp'].between(pre_start, t_impact))]
    post = df[(df['MMSI'] == mmsi) & (df['Timestamp'].between(t_impact, post_end))]
    
    trauma_flags = []
    
    # 1. The Blackout Check (Sinking / Complete Transponder Loss)
    if post.empty:
        trauma_flags.append("Catastrophic Blackout")
        return True, trauma_flags
        
    if not pre.empty:
        
        # --- TEST A: SOG Shock ---
        pre_sog_mean = pre['SOG'].mean()
        post_sog_mean = post['SOG'].mean()
        last_ping_sog = pre.iloc[-1]['SOG'] if 'SOG' in pre.columns and pd.notna(pre.iloc[-1]['SOG']) else pre_sog_mean
        
        if pd.notna(last_ping_sog) and last_ping_sog < 3.0:
            pass # Controlled approach bypass (e.g., rescue boat docking)
        elif pd.notna(pre_sog_mean) and pre_sog_mean > 2.0 and pd.notna(post_sog_mean):
            pre_sog_std = pre['SOG'].std()
            if pd.isna(pre_sog_std) or pre_sog_std < 0.5: pre_sog_std = 0.5
            if ((pre_sog_mean - post_sog_mean) / pre_sog_std) > 3.0:
                trauma_flags.append("SOG Shock (Statistical Anomaly)")
                
        # --- TEST B: Heading vs. COG Skid ---
        if 'Heading' in pre.columns and 'COG' in pre.columns and 'Heading' in post.columns and 'COG' in post.columns:
            pre_hdg_valid = pre[pre['Heading'] != 511]
            post_hdg_valid = post[post['Heading'] != 511]
            
            if not pre_hdg_valid.empty and not post_hdg_valid.empty:
                # 1. Calculate how much the ship naturally crabs/drifts prior to impact
                pre_diff = abs(pre_hdg_valid['Heading'] - pre_hdg_valid['COG']) % 360
                pre_skid = np.minimum(pre_diff, 360 - pre_diff)
                pre_skid_mean = pre_skid.mean()
                pre_skid_std = pre_skid.std()
                if pd.isna(pre_skid_std) or pre_skid_std < 2.0: pre_skid_std = 2.0
                
                # 2. Find the maximum drift angle after the impact
                post_diff = abs(post_hdg_valid['Heading'] - post_hdg_valid['COG']) % 360
                post_skid = np.minimum(post_diff, 360 - post_diff)
                post_skid_max = post_skid.max()
                
                # 3. Z-Score: Did it suddenly slide sideways significantly more than it was before?
                skid_z_score = (post_skid_max - pre_skid_mean) / pre_skid_std
                # Require a statistical spike AND a physical minimum of 15 degrees to filter out microscopic anomalies
                if skid_z_score > 3.0 and post_skid_max > 15.0:
                    trauma_flags.append("Heading/COG Skid (Statistical Anomaly)")
                    
        # --- TEST C: Involuntary Spin (ROT) ---
        if 'ROT' in pre.columns and 'ROT' in post.columns:
            pre_rot = pre['ROT'].dropna().abs()
            post_rot = post['ROT'].dropna().abs()
            
            if not pre_rot.empty and not post_rot.empty:
                # Calculate normal turning behavior
                pre_rot_mean = pre_rot.mean()
                pre_rot_std = pre_rot.std()
                if pd.isna(pre_rot_std) or pre_rot_std < 1.0: pre_rot_std = 1.0
                
                # Find maximum turn rate after impact
                post_rot_max = post_rot.max()
                
                rot_z_score = (post_rot_max - pre_rot_mean) / pre_rot_std
                
                if rot_z_score > 3.0 and post_rot_max > 5.0:
                    trauma_flags.append("ROT Spike (Statistical Anomaly)")
            
    has_trauma = len(trauma_flags) > 0
    return has_trauma, trauma_flags

def run_phase2_forensics():
    print("Loading datasets for Phase 2 Verification...")
    try:
        candidates = pd.read_csv(CANDIDATES_CSV)
        
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(TARGET_CSV) as f:
                df = pd.read_csv(f)
                
        if '# Timestamp' in df.columns:
            df.rename(columns={'# Timestamp': 'Timestamp'}, inplace=True)
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df['MMSI'] = df['MMSI'].astype(str).str.strip()
        candidates['Timestamp_A'] = pd.to_datetime(candidates['Timestamp_A'])
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    confirmed_crashes = []

    print(f"\nInitiating Kinematic & Physics Forensics on {len(candidates)} Suspects...\n")
    print("-" * 80)

    for idx, row in candidates.iterrows():
        t_impact = row['Timestamp_A']
        mmsi_a = str(row['MMSI_A'])
        mmsi_b = str(row['MMSI_B'])
        name_a = row['Name_A']
        name_b = row['Name_B']
        
        # --- 1. KINEMATIC INTERSECTION CHECK ---
        intersected, intersect_msg = check_kinematic_intersection(df, mmsi_a, mmsi_b, t_impact)
        
        # --- 2. PHYSICAL TRAUMA CHECK (+/- 5 MINUTES) ---
        trauma_a, flags_a = check_physical_trauma(df, mmsi_a, t_impact)
        trauma_b, flags_b = check_physical_trauma(df, mmsi_b, t_impact)
        
        # --- THE FINAL DECISION ENGINE ---
        if intersected and (trauma_a or trauma_b):
            confirmed_crashes.append(row)
            print(f"🚨 CONFIRMED COLLISION: {name_a} vs {name_b} at {t_impact}")
            print(f"   -> Kinematics: Paths physically crossed within +/- {intersect_msg} seconds")
            if trauma_a: print(f"   -> Trauma {name_a}: {', '.join(flags_a)}")
            if trauma_b: print(f"   -> Trauma {name_b}: {', '.join(flags_b)}")
            
            # =================================================================
            # EXPORT BLACK-BOX TELEMETRY FOR THE INCIDENT
            # =================================================================
            start_window = t_impact - pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
            end_window = t_impact + pd.Timedelta(minutes=TRAUMA_WINDOW_MINUTES)
            
            incident_telemetry = df[
                (df['MMSI'].isin([mmsi_a, mmsi_b])) & 
                (df['Timestamp'].between(start_window, end_window))
            ].copy()
            
            cols_to_keep = ['Timestamp', 'MMSI', 'Name', 'Latitude', 'Longitude', 'SOG', 'COG', 'Heading', 'ROT', 'Navigational status']
            existing_cols = [c for c in cols_to_keep if c in incident_telemetry.columns]
            incident_telemetry = incident_telemetry[existing_cols].sort_values(by='Timestamp')
            
            safe_name_a = str(name_a).replace(' ', '_').replace('/', '')
            safe_name_b = str(name_b).replace(' ', '_').replace('/', '')
            filename = f"telemetry_{safe_name_a}_vs_{safe_name_b}.csv"
            
            incident_telemetry.to_csv(filename, index=False)
            print(f"   -> 💾 Exported +/- 5 min raw telemetry to '{filename}'")
            print("-" * 80)
        else:
            reason = "Paths did not cross" if not intersected else "No physical trauma detected"
            if "Insufficient pings" in intersect_msg: reason = intersect_msg
            print(f"✅ Cleared: {name_a} vs {name_b} ({reason})")

    if confirmed_crashes:
        final_df = pd.DataFrame(confirmed_crashes)
        final_df.to_csv("FINAL_investigation_results.csv", index=False)
        print(f"\nInvestigation Complete. Saved {len(final_df)} confirmed disasters to FINAL_investigation_results.csv")
    else:
        print("\nInvestigation Complete. No fatal anomalies confirmed.")

if __name__ == "__main__":
    run_phase2_forensics()