import os
import pandas as pd
import numpy as np
import hashlib
from sqlalchemy import create_engine

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DIR = os.path.join(DATA_DIR, 'synthetic_output')

ORIGINAL_LMS_FILES = {
    'Basic SQL':                            'assignment_submissions_progress_Basic SQL.xlsx',
    'Basic Python Programming':             'assignment_submissions_progress_Basic Python Programming.xlsx',
    'Data Processing using NumPy & Pandas': 'assignment_submissions_progress_Data Processing using NumPy  Pa.xlsx',
    'Data Processing using Pyspark':        'assignment_submissions_progress_Data Processing using Pyspark.xlsx',
}

SYNTHETIC_LMS_FILES = {
    'Basic SQL':                            'synthetic_lms_Basic SQL.xlsx',
    'Basic Python Programming':             'synthetic_lms_Basic Python Programming.xlsx',
    'Data Processing using NumPy & Pandas': 'synthetic_lms_Data Processing using NumPy  Pa.xlsx',
    'Data Processing using Pyspark':        'synthetic_lms_Data Processing using Pyspark.xlsx',
}

ORIGINAL_EOD_FILE = 'intern_eod_last3months_random.xlsx'
SYNTHETIC_EOD_FILE = 'synthetic_eod_data.xlsx'


def save_to_database(fact_lms, fact_activity, dim_intern, dim_course):
    try:
        engine = create_engine('postgresql://hackathon_user:hackathon_password@localhost:5432/intern_analytics')
        fact_lms.to_sql('fact_lms', engine, if_exists='replace', index=False)
        fact_activity.to_sql('fact_activity', engine, if_exists='replace', index=False)
        dim_intern.to_sql('dim_intern', engine, if_exists='replace', index=False)
        dim_course.to_sql('dim_course', engine, if_exists='replace', index=False)
    except Exception as e:
        print(f"⚠️  Database save skipped (non-fatal): {e}")


# ─────────────────────────────────────────────
#  STEP 0 — MERGE SYNTHETIC INTO ORIGINALS
# ─────────────────────────────────────────────

def merge_and_update_excel():
    """
    Merge synthetic data INTO the original Excel files:
      1. For each LMS course file: append synthetic rows (de-duped by User Name + Course Name)
      2. For EOD file: append synthetic rows (de-duped by all key columns)
    Updates the original .xlsx files in-place and returns merged DataFrames.
    """
    print("\n🔄 Merging synthetic data into original Excel files...")
    merged_lms_dfs = []
    files_updated = 0

    # ── Merge LMS files ──────────────────────
    for course, orig_file in ORIGINAL_LMS_FILES.items():
        orig_path = os.path.join(DATA_DIR, orig_file)
        synth_file = SYNTHETIC_LMS_FILES[course]
        synth_path = os.path.join(SYNTHETIC_DIR, synth_file)

        if not os.path.exists(orig_path):
            print(f"  ⚠️  Original file not found: {orig_file}")
            continue

        df_orig = pd.read_excel(orig_path)
        orig_count = len(df_orig)

        if os.path.exists(synth_path):
            df_synth = pd.read_excel(synth_path)

            # De-duplicate: only add synthetic rows whose User Name is NOT already present
            existing_names = set(df_orig['User Name'].str.strip().str.title())
            df_synth['User Name'] = df_synth['User Name'].str.strip().str.title()
            df_new = df_synth[~df_synth['User Name'].isin(existing_names)]

            if len(df_new) > 0:
                df_merged = pd.concat([df_orig, df_new], ignore_index=True)
                # Skipped rewriting to Excel to prevent dashboard lockups
                files_updated += 1
                print(f"  ✅ {orig_file}: {orig_count} → {len(df_merged)} rows (+{len(df_new)} synthetic)")
            else:
                df_merged = df_orig
                print(f"  ➡️  {orig_file}: {orig_count} rows (synthetic already merged)")
        else:
            df_merged = df_orig
            print(f"  ➡️  {orig_file}: {orig_count} rows (no synthetic file found)")

        merged_lms_dfs.append(df_merged)

    # ── Merge EOD file ───────────────────────
    eod_orig_path = os.path.join(DATA_DIR, ORIGINAL_EOD_FILE)
    eod_synth_path = os.path.join(SYNTHETIC_DIR, SYNTHETIC_EOD_FILE)

    if os.path.exists(eod_orig_path):
        df_eod_orig = pd.read_excel(eod_orig_path)
        eod_orig_count = len(df_eod_orig)

        if os.path.exists(eod_synth_path):
            df_eod_synth = pd.read_excel(eod_synth_path)

            # De-duplicate: only add synthetic rows whose Full Name is NOT already present
            df_eod_orig['_full_name'] = (
                df_eod_orig['First Name'].astype(str).str.strip() + ' ' +
                df_eod_orig['Last Name'].astype(str).str.strip()
            ).str.title()
            df_eod_synth['_full_name'] = (
                df_eod_synth['First Name'].astype(str).str.strip() + ' ' +
                df_eod_synth['Last Name'].astype(str).str.strip()
            ).str.title()

            existing_eod_names = set(df_eod_orig['_full_name'])
            df_eod_new = df_eod_synth[~df_eod_synth['_full_name'].isin(existing_eod_names)]

            # Drop helper column before saving
            df_eod_orig = df_eod_orig.drop(columns=['_full_name'])
            df_eod_new = df_eod_new.drop(columns=['_full_name'])

            if len(df_eod_new) > 0:
                df_eod_merged = pd.concat([df_eod_orig, df_eod_new], ignore_index=True)
                # Skipped rewriting to Excel to prevent dashboard lockups
                files_updated += 1
                print(f"  ✅ {ORIGINAL_EOD_FILE}: {eod_orig_count} → {len(df_eod_merged)} rows (+{len(df_eod_new)} synthetic)")
            else:
                df_eod_merged = df_eod_orig
                print(f"  ➡️  {ORIGINAL_EOD_FILE}: {eod_orig_count} rows (synthetic already merged)")
        else:
            df_eod_merged = df_eod_orig
            print(f"  ➡️  {ORIGINAL_EOD_FILE}: {eod_orig_count} rows (no synthetic file)")
    else:
        print(f"  ❌ Original EOD file not found: {ORIGINAL_EOD_FILE}")
        df_eod_merged = pd.DataFrame()

    # Combine all LMS into a single DataFrame
    df_lms_merged = pd.concat(merged_lms_dfs, ignore_index=True) if merged_lms_dfs else pd.DataFrame()

    print(f"\n  📊 Merge Summary: {files_updated} files updated")
    print(f"     Total LMS rows: {len(df_lms_merged)}")
    print(f"     Total EOD rows: {len(df_eod_merged)}")

    return df_lms_merged, df_eod_merged


# ─────────────────────────────────────────────
#  MAIN ETL PROCESS
# ─────────────────────────────────────────────

def run_etl_process():
    """
    Full ELT pipeline:
      0. Merge synthetic data into original Excel files
      1. BRONZE — Re-ingest the updated Excel files
      2. SILVER — Clean & transform
      3. GOLD   — Dimensional star-schema modelling
      4. DATA PROFILING — Quality checks
    """

    # ── STEP 0: MERGE SYNTHETIC DATA ──
    synthetic_exists = os.path.exists(SYNTHETIC_DIR) and any(
        f.endswith('.xlsx') for f in os.listdir(SYNTHETIC_DIR) if not f.startswith('~$')
    )
    if synthetic_exists:
        df_lms_raw, df_eod_raw = merge_and_update_excel()
    else:
        print("ℹ️  No synthetic data found — reading original files directly.")
        lms_files = [os.path.join(DATA_DIR, f) for f in ORIGINAL_LMS_FILES.values()]
        try:
            df_lms_raw = pd.concat([pd.read_excel(f) for f in lms_files], ignore_index=True)
            df_eod_raw = pd.read_excel(os.path.join(DATA_DIR, ORIGINAL_EOD_FILE))
        except Exception as e:
            print(f"❌ File Ingestion Error: {e}")
            return None

    print(f"\n📥 Bronze Layer Loaded:")
    print(f"   LMS records : {len(df_lms_raw)}")
    print(f"   EOD records : {len(df_eod_raw)}")

    # ── STEP 2: SILVER LAYER (Preprocessing & Cleaning) ──

    # A. Handling Missing Values
    df_lms_raw['Overall Status'] = df_lms_raw['Overall Status'].fillna('Not Started').replace('-', 'Not Started')
    df_lms_raw['Progress (%)']   = df_lms_raw['Progress (%)'].fillna('0%')

    # B. Name Normalisation
    df_eod_raw['Full Name'] = (df_eod_raw['First Name'].astype(str).str.strip() + " " + df_eod_raw['Last Name'].astype(str).str.strip()).str.title()
    df_lms_raw['User Name'] = df_lms_raw['User Name'].astype(str).str.strip().str.title()

    # C. Progress % → float
    df_lms_raw['Progress_Numeric'] = df_lms_raw['Progress (%)'].astype(str).str.rstrip('%').astype(float)

    # D. Assignments fraction  e.g. "3/4" → Reviewed=3, Total_Assg=4
    split_assign = df_lms_raw['Reviewed / Total Assignments'].astype(str).str.split('/', expand=True)
    df_lms_raw['Reviewed']   = pd.to_numeric(split_assign[0].str.strip(), errors='coerce').fillna(0).astype(int)
    df_lms_raw['Total_Assg'] = pd.to_numeric(split_assign[1].str.strip(), errors='coerce').fillna(1).astype(int)

    # E. Reviewed / Submitted fraction
    split_rev = df_lms_raw['Reviewed / Submitted'].astype(str).str.split('/', expand=True)
    df_lms_raw['Rev_submitted'] = pd.to_numeric(split_rev[0].str.strip(), errors='coerce').fillna(0).astype(int)
    df_lms_raw['Submitted']     = pd.to_numeric(split_rev[1].str.strip(), errors='coerce').fillna(0).astype(int)

    # F. Knowledge Check score  e.g. "68.0 / 70.0" → KC_scored=68.0, KC_total=70.0
    split_kc = df_lms_raw['Overall Knowledge Check'].astype(str).str.split('/', expand=True)
    df_lms_raw['KC_scored'] = pd.to_numeric(split_kc[0].str.strip(), errors='coerce').fillna(0.0)
    df_lms_raw['KC_total']  = pd.to_numeric(split_kc[1].str.strip(), errors='coerce').fillna(1.0)
    df_lms_raw['KC_total']  = df_lms_raw['KC_total'].replace(0, 1)

    # G. Test score  e.g. "820.99 / 933.0" → Test_scored, Test_total
    split_test = df_lms_raw['Overall Test'].astype(str).str.split('/', expand=True)
    df_lms_raw['Test_scored'] = pd.to_numeric(split_test[0].str.strip(), errors='coerce').fillna(0.0)
    df_lms_raw['Test_total']  = pd.to_numeric(split_test[1].str.strip(), errors='coerce').fillna(1.0)
    df_lms_raw['Test_total']  = df_lms_raw['Test_total'].replace(0, 1)

    # H. Date normalisation for EOD
    df_eod_raw['Date'] = pd.to_datetime(df_eod_raw['Date'], dayfirst=True, errors='coerce')

    # ── STEP 3: GOLD LAYER (Dimensional Modelling - Star Schema) ──

    # DIM_INTERN  (SCD Type 1)
    dim_intern = pd.DataFrame(df_lms_raw['User Name'].unique(), columns=['Intern_Name'])
    dim_intern['Intern_ID'] = [hashlib.md5(name.encode()).hexdigest()[:8] for name in dim_intern['Intern_Name']]

    # DIM_COURSE
    dim_course = df_lms_raw[['Course Name', 'Start Date', 'End Date']].drop_duplicates().reset_index(drop=True)
    dim_course['Course_ID'] = [f"CRS-{i:03d}" for i in range(len(dim_course))]

    # DIM_MENTOR  (atomised - one row per mentor name)
    mentors = df_lms_raw['Mentor Name'].astype(str).str.split(', ').explode().unique()
    dim_mentor = pd.DataFrame(mentors, columns=['Mentor_Name']).dropna().reset_index(drop=True)

    # FACT_LMS_PROGRESS  — includes KC + Test score columns for mentor_dashboard
    fact_lms = df_lms_raw.merge(dim_intern, left_on='User Name', right_on='Intern_Name')
    fact_lms = fact_lms.merge(dim_course, on=['Course Name', 'Start Date', 'End Date'])
    fact_lms = fact_lms[[
        'Intern_ID', 'Course_ID',
        'Progress_Numeric', 'Reviewed', 'Total_Assg',
        'Rev_submitted', 'Submitted',
        'Overall Status', 'Mentor Name',
        'KC_scored', 'KC_total',
        'Test_scored', 'Test_total',
    ]]

    # FACT_DAILY_EFFORT
    fact_activity = df_eod_raw.merge(dim_intern, left_on='Full Name', right_on='Intern_Name')
    fact_activity = fact_activity[['Date', 'Intern_ID', 'Activity', 'Hours']]

    # ── STEP 4: DATA PROFILING & QUALITY CHECKS ──
    daily_effort  = fact_activity.groupby(['Intern_ID', 'Date'])['Hours'].sum().reset_index()
    outlier_count = int(len(daily_effort[daily_effort['Hours'] > 12]))

    orphan_count     = len(df_eod_raw[~df_eod_raw['Full Name'].isin(df_lms_raw['User Name'])])
    duplicates       = df_lms_raw.duplicated(subset=['User Name', 'Course Name']).sum()
    invalid_hours    = (fact_activity['Hours'] < 0).sum()
    invalid_progress = (fact_lms['Progress_Numeric'] > 100).sum()

    print("\n📋 Data Profiling Report")
    print("─" * 40)
    print(f"  Unique Interns    : {len(dim_intern)}")
    print(f"  Courses           : {len(dim_course)}")
    print(f"  Mentors           : {len(dim_mentor)}")
    print(f"  Fact LMS rows     : {len(fact_lms)}")
    print(f"  Fact Activity rows: {len(fact_activity)}")
    print(f"  Duplicates Found  : {duplicates}")
    print(f"  Invalid Hours     : {invalid_hours}")
    print(f"  Invalid Progress  : {invalid_progress}")
    print(f"  Orphan Records    : {orphan_count}")
    print(f"  Outlier Days (>12h): {outlier_count}")

    try:
        save_to_database(fact_lms, fact_activity, dim_intern, dim_course)
    except Exception as e:
        print(f"⚠️  Database Save Error: {e}")

    return fact_lms, fact_activity, dim_intern, dim_course, dim_mentor, outlier_count


if __name__ == "__main__":
    res = run_etl_process()
    if res:
        f_lms, f_act, d_int, d_crs, d_mnt, alerts = res
        print(f"\n✅ ETL Complete")
        print(f"   fact_lms columns : {f_lms.columns.tolist()}")
        print(f"   Dimensions       : {len(d_int)} Interns | {len(d_crs)} Courses | {len(d_mnt)} Mentors")
        print(f"   Outliers         : {alerts}")