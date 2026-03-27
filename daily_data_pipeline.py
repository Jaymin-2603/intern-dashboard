import os
import glob
import time
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
# Path to the folder containing your CSV files
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Database credentials (PostgreSQL used here per existing project setup)
DB_TYPE = 'postgresql'
DB_USER = 'hackathon_user'
DB_PASS = 'hackathon_password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'intern_analytics'

def get_db_engine():
    """Create and return a SQLAlchemy engine for PostgreSQL."""
    # Format: postgresql://username:password@host:port/database_name
    connection_string = f"{DB_TYPE}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def read_files_with_fallback(pattern, directory):
    """
    Reads files matching a pattern. 
    The prompt requested CSV, but handles fallback if files are actually Excel (.xlsx).
    """
    files = glob.glob(os.path.join(directory, pattern))
    dfs = []
    for f in files:
        try:
            if f.endswith('.csv'):
                dfs.append(pd.read_csv(f))
            elif f.endswith('.xlsx'):
                dfs.append(pd.read_excel(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def apply_database_indexes(engine):
    """Ensure database indexes exist on critical columns for performance."""
    with engine.begin() as conn:
        # Index on User Name for LMS submissions table
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_lms_user_name ON lms_submissions(\"User Name\");"))
            print("✅ Index on lms_submissions('User Name') confirmed.")
        except Exception as e:
            print(f"⚠️ Could not create index on lms_submissions: {e}")
            
        # Index on Date and User Name for EOD table
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eod_date ON eod_activity(\"Date\");"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_eod_user_name ON eod_activity(\"User Name\");"))
            print("✅ Index on eod_activity('Date', 'User Name') confirmed.")
        except Exception as e:
            print(f"⚠️ Could not create index on eod_activity: {e}")

def run_daily_pipeline():
    """Main ETL pipeline to extract, transform, and load data."""
    print(f"\n[{datetime.now()}] 🚀 Starting daily data pipeline...")
    engine = get_db_engine()
    
    # ─────────────────────────────────────────────
    #  1. SOURCE & PROCESSING: LMS SUBMISSIONS
    # ─────────────────────────────────────────────
    # Matches both .csv and .xlsx just in case
    df_lms = read_files_with_fallback('assignment_submissions*.*', DATA_DIR)
    
    if not df_lms.empty:
        print(f"📥 Loaded {len(df_lms)} rows of LMS data.")
        
        # A. Missing Values: 'Overall Test'
        if 'Overall Test' in df_lms.columns:
            # Impute with dummy fraction or 0 if missing. 
            df_lms['Overall Test'] = df_lms['Overall Test'].fillna('0 / 1')
            
        # B. Date Formatting: 'Start Date' & 'End Date' -> datetime objects
        if 'Start Date' in df_lms.columns:
            df_lms['Start Date'] = pd.to_datetime(df_lms['Start Date'], errors='coerce')
            
        if 'End Date' in df_lms.columns:
            df_lms['End Date'] = pd.to_datetime(df_lms['End Date'], errors='coerce')
        
        # Standardize User Name format for joining consistency
        if 'User Name' in df_lms.columns:
            df_lms['User Name'] = df_lms['User Name'].astype(str).str.strip().str.title()
            
        # C. Loading: Append to Database
        try:
            df_lms.to_sql('lms_submissions', engine, if_exists='append', index=False)
            print("✅ LMS data appended to database.")
        except Exception as e:
            print(f"❌ Failed to push LMS data: {e}")
    else:
        print("ℹ️ No assignment_submissions files found.")

    # ─────────────────────────────────────────────
    #  2. SOURCE & PROCESSING: INTERN EOD
    # ─────────────────────────────────────────────
    df_eod = read_files_with_fallback('intern_eod*.*', DATA_DIR)
    
    if not df_eod.empty:
        print(f"📥 Loaded {len(df_eod)} rows of EOD activity data.")
        
        # A. Date Formatting
        if 'Date' in df_eod.columns:
            df_eod['Date'] = pd.to_datetime(df_eod['Date'], errors='coerce')
            
        # B. Make sure 'User Name' exists in EOD (sometimes it's split First/Last)
        if 'User Name' not in df_eod.columns and 'First Name' in df_eod.columns and 'Last Name' in df_eod.columns:
            df_eod['User Name'] = (df_eod['First Name'].astype(str).str.strip() + " " + df_eod['Last Name'].astype(str).str.strip()).str.title()
            
        # C. Loading: Append to Database
        try:
            df_eod.to_sql('eod_activity', engine, if_exists='append', index=False)
            print("✅ EOD data appended to database.")
        except Exception as e:
            print(f"❌ Failed to push EOD data: {e}")
    else:
        print("ℹ️ No intern_eod files found.")
        
    # ─────────────────────────────────────────────
    #  3. OPTIMIZATION: CREATE INDEXES
    # ─────────────────────────────────────────────
    apply_database_indexes(engine)
    print(f"[{datetime.now()}] 🏁 Pipeline execution finished.")

def start_scheduler():
    """Wrap pipeline execution in an automated daily schedule."""
    import schedule  # Imported here so --run-now doesn't require it
    
    # Optional: Run immediately on startup once
    # run_daily_pipeline()
    
    # Schedule the job every day at a specific time, e.g., 02:00 AM
    schedule.every().day.at("02:00").do(run_daily_pipeline)
    print("⏳ Automated scheduler active. Waiting for next daily run... (Press Ctrl+C to exit)")
    
    # Keep script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Data Pipeline")
    parser.add_argument("--run-now", action="store_true", help="Run the pipeline once immediately")
    args = parser.parse_args()
    
    if args.run_now:
        run_daily_pipeline()
    else:
        start_scheduler()
