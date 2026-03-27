"""
Synthetic Data Generator for Kenexai Intern Analytics
=====================================================
Generates realistic synthetic intern data by learning distributions from
the existing dataset and scaling up to N interns.

Usage:
    python generate_synthetic_data.py                       # 200 interns, seed=42
    python generate_synthetic_data.py --num-interns 500     # 500 interns
    python generate_synthetic_data.py --validate            # run with built-in checks
"""

import argparse
import os
import sys
import warnings
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from faker import Faker

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
ORIGINAL_LMS_FILES = [
    'assignment_submissions_progress_Basic SQL.xlsx',
    'assignment_submissions_progress_Basic Python Programming.xlsx',
    'assignment_submissions_progress_Data Processing using NumPy  Pa.xlsx',
    'assignment_submissions_progress_Data Processing using Pyspark.xlsx',
]
ORIGINAL_EOD_FILE = 'intern_eod_last3months_random.xlsx'

COURSE_CONFIGS = {
    'Basic SQL': {
        'start_date': '01 Jan 2026',
        'end_date': '01 Feb 2026',
    },
    'Basic Python Programming': {
        'start_date': '01 Jan 2026',
        'end_date': '01 Feb 2026',
    },
    'Data Processing using NumPy & Pandas': {
        'start_date': '01 Feb 2026',
        'end_date': '01 Mar 2026',
    },
    'Data Processing using Pyspark': {
        'start_date': '01 Feb 2026',
        'end_date': '01 Mar 2026',
    },
}


# ─────────────────────────────────────────────
#  STEP 1 — LEARN DISTRIBUTIONS FROM ORIGINALS
# ─────────────────────────────────────────────

def load_originals(data_dir):
    """Load and return original LMS + EOD DataFrames."""
    lms_dfs = []
    for f in ORIGINAL_LMS_FILES:
        path = os.path.join(data_dir, f)
        lms_dfs.append(pd.read_excel(path))
    df_lms = pd.concat(lms_dfs, ignore_index=True)
    df_eod = pd.read_excel(os.path.join(data_dir, ORIGINAL_EOD_FILE))
    return df_lms, df_eod


def learn_distributions(df_lms, df_eod):
    """Extract statistical profiles from real data."""
    dist = {}

    # ── Progress distribution per course ──
    dist['progress_per_course'] = {}
    for course in df_lms['Course Name'].unique():
        progs = df_lms[df_lms['Course Name'] == course]['Progress (%)'].astype(str).str.rstrip('%')
        progs = pd.to_numeric(progs, errors='coerce').dropna().values
        dist['progress_per_course'][course] = {
            'mean': float(np.mean(progs)) if len(progs) else 50.0,
            'std': float(np.std(progs)) if len(progs) else 25.0,
            'values': progs.tolist(),
        }

    # ── Status distribution ──
    status_counts = df_lms['Overall Status'].fillna('Not Started').value_counts(normalize=True)
    dist['status_probs'] = status_counts.to_dict()

    # ── Assignment ratios ──
    split = df_lms['Reviewed / Total Assignments'].astype(str).str.split('/', expand=True)
    reviewed = pd.to_numeric(split[0].str.strip(), errors='coerce').fillna(0)
    total = pd.to_numeric(split[1].str.strip(), errors='coerce').fillna(1)
    ratios = (reviewed / total.replace(0, 1)).values
    dist['assg_ratio'] = {'mean': float(np.mean(ratios)), 'std': float(np.std(ratios))}
    dist['assg_total_values'] = sorted(total.unique().tolist())

    # ── Reviewed / Submitted ratios ──
    split_rs = df_lms['Reviewed / Submitted'].astype(str).str.split('/', expand=True)
    rev_sub = pd.to_numeric(split_rs[0].str.strip(), errors='coerce').fillna(0)
    submitted = pd.to_numeric(split_rs[1].str.strip(), errors='coerce').fillna(0)
    dist['rev_sub_ratio'] = {
        'mean': float(np.mean((rev_sub / submitted.replace(0, 1)).values)),
        'std': float(np.std((rev_sub / submitted.replace(0, 1)).values)),
    }

    # ── KC Scores ──
    kc_split = df_lms['Overall Knowledge Check'].astype(str).str.split('/', expand=True)
    kc_scored = pd.to_numeric(kc_split[0].str.strip(), errors='coerce').fillna(0)
    kc_total = pd.to_numeric(kc_split[1].str.strip(), errors='coerce').fillna(1)
    kc_pct = (kc_scored / kc_total.replace(0, 1)).values
    dist['kc_ratio'] = {'mean': float(np.mean(kc_pct)), 'std': float(np.std(kc_pct))}
    dist['kc_total_values'] = sorted(kc_total.unique().tolist())

    # ── Test Scores ──
    ts_split = df_lms['Overall Test'].astype(str).str.split('/', expand=True)
    ts_scored = pd.to_numeric(ts_split[0].str.strip(), errors='coerce').fillna(0)
    ts_total = pd.to_numeric(ts_split[1].str.strip(), errors='coerce').fillna(1)
    ts_pct = (ts_scored / ts_total.replace(0, 1)).values
    dist['test_ratio'] = {'mean': float(np.mean(ts_pct)), 'std': float(np.std(ts_pct))}
    dist['test_total_values'] = sorted(ts_total.unique().tolist())

    # ── Mentor groups per course ──
    dist['mentors_per_course'] = {}
    for course in df_lms['Course Name'].unique():
        mentors = df_lms[df_lms['Course Name'] == course]['Mentor Name'].dropna().unique().tolist()
        dist['mentors_per_course'][course] = mentors

    # ── EOD: Activity frequencies ──
    act_counts = df_eod['Activity'].value_counts(normalize=True)
    dist['activity_probs'] = act_counts.to_dict()
    dist['activities'] = act_counts.index.tolist()

    # ── EOD: Hours distribution ──
    hours = df_eod['Hours'].values
    dist['hours'] = {
        'mean': float(np.mean(hours)),
        'std': float(np.std(hours)),
        'min': float(np.min(hours)),
        'max': float(np.max(hours)),
    }

    # ── EOD: Records per intern distribution ──
    eod_names = (df_eod['First Name'].str.strip() + ' ' + df_eod['Last Name'].str.strip()).str.title()
    records_per_intern = eod_names.value_counts()
    dist['eod_records_per_intern'] = {
        'mean': float(records_per_intern.mean()),
        'std': float(records_per_intern.std()),
        'min': int(records_per_intern.min()),
        'max': int(records_per_intern.max()),
    }

    # ── EOD: Date range ──
    dates = pd.to_datetime(df_eod['Date'], dayfirst=True, errors='coerce').dropna()
    dist['date_range'] = {
        'start': dates.min(),
        'end': dates.max(),
    }

    # ── Completed Assignment patterns ──
    dist['completed_assg_values'] = df_lms['Completed Assignment'].dropna().unique().tolist()

    return dist


# ─────────────────────────────────────────────
#  STEP 2 — NAME GENERATION
# ─────────────────────────────────────────────

# Pool of common Indian first and last names for realistic generation
INDIAN_FIRST_NAMES = [
    'Aarav', 'Aditya', 'Akash', 'Amit', 'Ananya', 'Aniket', 'Ankit', 'Ankita',
    'Ansh', 'Archit', 'Arjun', 'Arnav', 'Arushi', 'Avni', 'Bhavya', 'Chirag',
    'Darsh', 'Darshan', 'Deep', 'Deepak', 'Dev', 'Devi', 'Dhairya', 'Dhaval',
    'Dhruv', 'Disha', 'Divya', 'Ekta', 'Garv', 'Gaurav', 'Harsh', 'Harshita',
    'Hetal', 'Himani', 'Hiren', 'Ishaan', 'Isha', 'Jagruti', 'Janvi', 'Jatin',
    'Jeel', 'Jignesh', 'Jigar', 'Jinay', 'Kajal', 'Karan', 'Kartik', 'Kavya',
    'Khushi', 'Kishan', 'Komal', 'Krishna', 'Krish', 'Kruti', 'Kunal', 'Laksh',
    'Lavanya', 'Madhav', 'Mahek', 'Mahir', 'Manav', 'Manish', 'Mansi', 'Mayur',
    'Meera', 'Megha', 'Milan', 'Mira', 'Mitali', 'Mohit', 'Mukund', 'Naman',
    'Nandini', 'Neel', 'Neha', 'Nikhil', 'Nikita', 'Nisha', 'Nishant', 'Ojas',
    'Om', 'Palak', 'Param', 'Parth', 'Pooja', 'Prachi', 'Pranav', 'Pranit',
    'Pratham', 'Priya', 'Priyanshi', 'Punit', 'Raj', 'Rajvi', 'Rashi', 'Ravi',
    'Riddhi', 'Rishi', 'Riya', 'Rohan', 'Rohit', 'Rushil', 'Rutvik', 'Saanvi',
    'Sahil', 'Sakshi', 'Samarth', 'Sanjay', 'Sanya', 'Sara', 'Sarthak', 'Saumya',
    'Shreya', 'Shruti', 'Siddharth', 'Simran', 'Smit', 'Sneha', 'Soham', 'Suhani',
    'Suraj', 'Swara', 'Tanay', 'Tanishka', 'Tanvi', 'Tisha', 'Tushar', 'Urvi',
    'Vaidik', 'Vansh', 'Varun', 'Ved', 'Vedant', 'Vidhi', 'Vikas', 'Viren',
    'Vishwa', 'Vivaan', 'Vivek', 'Yash', 'Yashika', 'Yogesh', 'Zeel',
]

INDIAN_LAST_NAMES = [
    'Acharya', 'Agrawal', 'Amin', 'Barot', 'Bhagat', 'Bhatt', 'Chauhan',
    'Chaudhary', 'Darji', 'Desai', 'Deshmukh', 'Dholakia', 'Dixit', 'Gajjar',
    'Ganatra', 'Gohil', 'Gupta', 'Iyer', 'Jadav', 'Jani', 'Jha', 'Joshi',
    'Kapoor', 'Khatri', 'Kothari', 'Kulkarni', 'Kumar', 'Lakhani', 'Makwana',
    'Malhotra', 'Mehta', 'Mishra', 'Modi', 'Nair', 'Nanavati', 'Pandey',
    'Pandya', 'Parekh', 'Parikh', 'Parmar', 'Patel', 'Pathak', 'Pillai',
    'Raghav', 'Rajput', 'Rana', 'Rathod', 'Raval', 'Rawat', 'Reddy',
    'Saini', 'Sanghavi', 'Saxena', 'Shah', 'Sharma', 'Shinde', 'Shukla',
    'Singh', 'Solanki', 'Soni', 'Srivastava', 'Suthar', 'Thakkar', 'Tiwari',
    'Trivedi', 'Vaghela', 'Verma', 'Vyas', 'Yadav',
]


def generate_unique_names(n, rng):
    """Generate n unique First Name + Last Name pairs."""
    names = set()
    first_pool = list(INDIAN_FIRST_NAMES)
    last_pool = list(INDIAN_LAST_NAMES)

    while len(names) < n:
        first = rng.choice(first_pool)
        last = rng.choice(last_pool)
        full = f"{first} {last}"
        names.add(full)

    names_list = sorted(names)
    rng.shuffle(names_list)
    return [(name.split()[0], name.split()[1]) for name in names_list[:n]]


# ─────────────────────────────────────────────
#  STEP 3 — SYNTHETIC LMS GENERATION
# ─────────────────────────────────────────────

def generate_intern_profile(rng):
    """
    Generate a latent 'ability' score [0, 1] per intern.
    This drives correlated progress, scores, and effort.
    """
    return np.clip(rng.beta(2.5, 2.0), 0.05, 0.99)


def generate_synthetic_lms(names, dist, rng):
    """
    Generate LMS records for all interns × all courses.
    Returns a dict of {course_name: DataFrame}.
    """
    courses = list(COURSE_CONFIGS.keys())
    course_dfs = {c: [] for c in courses}

    for first, last in names:
        full_name = f"{first} {last}"
        ability = generate_intern_profile(rng)

        for course in courses:
            cfg = COURSE_CONFIGS[course]
            prog_dist = dist['progress_per_course'].get(course, {'mean': 50, 'std': 25})

            # ── Progress (correlated with ability) ──
            base_progress = ability * 100
            noise = rng.normal(0, 10)
            progress = np.clip(base_progress + noise, 0, 100)
            progress = round(progress)

            # ── Status (driven by progress) ──
            if progress >= 90:
                status = rng.choice(['Completed', 'In Progress'], p=[0.8, 0.2])
            elif progress >= 40:
                status = rng.choice(['In Progress', 'Completed', 'Not started'], p=[0.75, 0.15, 0.10])
            elif progress > 0:
                status = rng.choice(['In Progress', 'Not started'], p=[0.65, 0.35])
            else:
                status = 'Not started'
                progress = 0

            # ── Assignments (correlated with progress) ──
            total_assg = int(rng.choice([3, 4, 5]))
            reviewed_ratio = np.clip(ability + rng.normal(0, 0.15), 0, 1)
            reviewed = int(round(reviewed_ratio * total_assg))
            reviewed = min(reviewed, total_assg)

            # Completed Assignment string
            completed_count = min(reviewed + int(rng.choice([0, 1])), total_assg)
            completed_assg_str = f"{completed_count}/{total_assg}"

            # Reviewed / Submitted
            submitted = min(completed_count + int(rng.choice([0, 0, 1])), total_assg)
            rev_submitted = min(reviewed, submitted)
            rev_sub_str = f"{rev_submitted}/{submitted}" if submitted > 0 else "0/0"

            # Reviewed / Total Assignments
            rev_total_str = f"{reviewed}/{total_assg}"

            # ── Knowledge Check (correlated with ability) ──
            kc_total = float(rng.choice([70.0, 100.0, 150.0, 200.0, 300.0, 470.0]))
            kc_pct = np.clip(ability + rng.normal(0, 0.12), 0, 1)
            kc_scored = round(kc_pct * kc_total, 2)
            kc_str = f"{kc_scored} / {kc_total}"

            # ── Test Score (correlated with ability) ──
            test_total = float(rng.choice([40.0, 50.0, 80.0, 100.0, 200.0, 933.0]))
            test_pct = np.clip(ability + rng.normal(0, 0.15), 0, 1)
            test_scored = round(test_pct * test_total, 2)
            test_str = f"{test_scored} / {test_total}"

            # ── Mentor (from real mentor groups for this course) ──
            mentor_options = dist['mentors_per_course'].get(course, ['Mentor A'])
            mentor_name = rng.choice(mentor_options) if mentor_options else 'Unknown Mentor'

            # ── Progress string ──
            progress_str = f"{progress}%"

            course_dfs[course].append({
                'User Name': full_name,
                'Course Name': course,
                'Start Date': cfg['start_date'],
                'End Date': cfg['end_date'],
                'Mentor Name': mentor_name,
                'Progress (%)': progress_str,
                'Completed Assignment': completed_assg_str,
                'Reviewed / Submitted': rev_sub_str,
                'Overall Knowledge Check': kc_str,
                'Overall Test': test_str,
                'Reviewed / Total Assignments': rev_total_str,
                'Overall Status': status,
            })

    return {c: pd.DataFrame(rows) for c, rows in course_dfs.items()}


# ─────────────────────────────────────────────
#  STEP 4 — SYNTHETIC EOD GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_eod(names, dist, lms_dfs, rng):
    """
    Generate daily EOD activity records for each intern.
    Higher-ability interns log more days and hours.
    """
    date_start = dist['date_range']['start']
    date_end = dist['date_range']['end']
    all_dates = pd.date_range(start=date_start, end=date_end, freq='D')

    # Build activity weights
    activities = dist['activities']
    act_probs = np.array([dist['activity_probs'][a] for a in activities])
    act_probs = act_probs / act_probs.sum()  # ensure sums to 1

    hours_mean = dist['hours']['mean']
    hours_std = dist['hours']['std']
    hours_min = dist['hours']['min']
    hours_max = dist['hours']['max']

    # Build a lookup of avg progress per intern from LMS
    intern_progress = {}
    for course, df in lms_dfs.items():
        for _, row in df.iterrows():
            name = row['User Name']
            prog = float(row['Progress (%)'].rstrip('%'))
            intern_progress.setdefault(name, []).append(prog)
    intern_avg_progress = {n: np.mean(ps) for n, ps in intern_progress.items()}

    records = []
    for first, last in names:
        full_name = f"{first} {last}"
        avg_prog = intern_avg_progress.get(full_name, 50) / 100.0  # 0-1 scale

        # Number of activity days (higher progress → more active days)
        base_records = dist['eod_records_per_intern']['mean']
        n_records = int(np.clip(
            rng.normal(base_records * (0.5 + 0.7 * avg_prog), base_records * 0.2),
            base_records * 0.3,
            base_records * 1.5,
        ))

        # Pick random active dates
        if n_records >= len(all_dates):
            active_dates = all_dates.tolist()
        else:
            active_date_indices = rng.choice(len(all_dates), size=n_records, replace=True)
            active_dates = [all_dates[i] for i in sorted(active_date_indices)]

        for dt in active_dates:
            # Each day: 1-3 activities
            n_activities = rng.choice([1, 1, 1, 2, 2, 3])
            day_activities = rng.choice(activities, size=n_activities, replace=False, p=act_probs)

            for activity in day_activities:
                hours = np.clip(
                    rng.normal(hours_mean * (0.6 + 0.6 * avg_prog), hours_std),
                    hours_min,
                    hours_max,
                )
                hours = round(hours, 1)

                records.append({
                    'Date': dt.strftime('%d/%m/%Y'),
                    'First Name': first,
                    'Last Name': last,
                    'Activity': activity,
                    'Hours': hours,
                })

    df_eod = pd.DataFrame(records)
    return df_eod


# ─────────────────────────────────────────────
#  STEP 5 — VALIDATION
# ─────────────────────────────────────────────

def validate_output(lms_dfs, df_eod, num_interns):
    """Run quality checks on generated data."""
    print("\n🔍 Running Validation Checks...")
    passed = 0
    failed = 0

    expected_lms_cols = [
        'User Name', 'Course Name', 'Start Date', 'End Date', 'Mentor Name',
        'Progress (%)', 'Completed Assignment', 'Reviewed / Submitted',
        'Overall Knowledge Check', 'Overall Test', 'Reviewed / Total Assignments',
        'Overall Status',
    ]
    expected_eod_cols = ['Date', 'First Name', 'Last Name', 'Activity', 'Hours']

    # ── Check 1: LMS Schema ──
    for course, df in lms_dfs.items():
        if list(df.columns) == expected_lms_cols:
            passed += 1
        else:
            print(f"  ❌ Schema mismatch for {course}")
            print(f"     Expected: {expected_lms_cols}")
            print(f"     Got:      {list(df.columns)}")
            failed += 1

    # ── Check 2: EOD Schema ──
    if list(df_eod.columns) == expected_eod_cols:
        passed += 1
    else:
        print(f"  ❌ EOD schema mismatch")
        failed += 1

    # ── Check 3: Row counts ──
    for course, df in lms_dfs.items():
        if len(df) == num_interns:
            passed += 1
        else:
            print(f"  ❌ {course}: expected {num_interns} rows, got {len(df)}")
            failed += 1

    # ── Check 4: Unique intern names ──
    all_names = set()
    for df in lms_dfs.values():
        all_names.update(df['User Name'].unique())
    if len(all_names) == num_interns:
        passed += 1
    else:
        print(f"  ❌ Expected {num_interns} unique names, got {len(all_names)}")
        failed += 1

    # ── Check 5: Hours in valid range ──
    if df_eod['Hours'].between(0.0, 10.0).all():
        passed += 1
    else:
        bad = df_eod[~df_eod['Hours'].between(0.0, 10.0)]
        print(f"  ❌ {len(bad)} EOD records with invalid hours")
        failed += 1

    # ── Check 6: Progress values ──
    for course, df in lms_dfs.items():
        progs = df['Progress (%)'].astype(str).str.rstrip('%').astype(float)
        if progs.between(0, 100).all():
            passed += 1
        else:
            print(f"  ❌ {course}: invalid progress values found")
            failed += 1

    # ── Check 7: Status values ──
    valid_statuses = {'Completed', 'In Progress', 'In progress', 'Not started', 'Not Started'}
    for course, df in lms_dfs.items():
        if set(df['Overall Status'].unique()).issubset(valid_statuses):
            passed += 1
        else:
            invalid = set(df['Overall Status'].unique()) - valid_statuses
            print(f"  ❌ {course}: invalid statuses: {invalid}")
            failed += 1

    # ── Check 8: Logical consistency (Completed → high progress) ──
    consistency_ok = True
    for course, df in lms_dfs.items():
        completed = df[df['Overall Status'] == 'Completed']
        if not completed.empty:
            progs = completed['Progress (%)'].astype(str).str.rstrip('%').astype(float)
            if progs.min() < 30:
                print(f"  ⚠️  {course}: 'Completed' intern with {progs.min()}% progress")
                consistency_ok = False
    if consistency_ok:
        passed += 1
    else:
        failed += 1

    # ── Check 9: EOD records exist for all interns ──
    eod_names = set((df_eod['First Name'] + ' ' + df_eod['Last Name']).unique())
    if len(eod_names) == num_interns:
        passed += 1
    else:
        print(f"  ❌ EOD has {len(eod_names)} unique interns, expected {num_interns}")
        failed += 1

    total = passed + failed
    print(f"\n  ✅ Passed: {passed}/{total}")
    if failed:
        print(f"  ❌ Failed: {failed}/{total}")
    else:
        print("  🎉 All checks passed!")

    return failed == 0


# ─────────────────────────────────────────────
#  STEP 6 — EXPORT
# ─────────────────────────────────────────────

def export_data(lms_dfs, df_eod, output_dir):
    """Save generated data to Excel files."""
    os.makedirs(output_dir, exist_ok=True)

    # Export per-course LMS files
    course_file_map = {
        'Basic SQL': 'synthetic_lms_Basic SQL.xlsx',
        'Basic Python Programming': 'synthetic_lms_Basic Python Programming.xlsx',
        'Data Processing using NumPy & Pandas': 'synthetic_lms_Data Processing using NumPy  Pa.xlsx',
        'Data Processing using Pyspark': 'synthetic_lms_Data Processing using Pyspark.xlsx',
    }

    for course, df in lms_dfs.items():
        fname = course_file_map.get(course, f"synthetic_lms_{course}.xlsx")
        path = os.path.join(output_dir, fname)
        df.to_excel(path, index=False, engine='openpyxl')
        print(f"  📄 {fname} ({len(df)} rows)")

    # Export EOD
    eod_path = os.path.join(output_dir, 'synthetic_eod_data.xlsx')
    df_eod.to_excel(eod_path, index=False, engine='openpyxl')
    print(f"  📄 synthetic_eod_data.xlsx ({len(df_eod)} rows)")

    # Export combined LMS
    combined_lms = pd.concat(lms_dfs.values(), ignore_index=True)
    combined_path = os.path.join(output_dir, 'synthetic_combined_lms.xlsx')
    combined_lms.to_excel(combined_path, index=False, engine='openpyxl')
    print(f"  📄 synthetic_combined_lms.xlsx ({len(combined_lms)} rows)")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic intern data from existing dataset distributions'
    )
    parser.add_argument('--num-interns', type=int, default=200,
                        help='Number of synthetic interns to generate (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./synthetic_output',
                        help='Output directory for generated files (default: ./synthetic_output)')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory containing original data files (default: .)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks on the generated data')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.num_interns

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Synthetic Data Generator — Kenexai          ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Interns : {n:<6}                             ║")
    print(f"║  Seed    : {args.seed:<6}                             ║")
    print(f"║  Output  : {args.output_dir:<34}║")
    print(f"╚══════════════════════════════════════════════╝")

    # Step 1: Load originals
    print("\n📥 Loading original data...")
    df_lms, df_eod = load_originals(args.data_dir)
    print(f"   LMS: {len(df_lms)} rows | EOD: {len(df_eod)} rows")

    # Step 2: Learn distributions
    print("\n📊 Learning distributions from real data...")
    dist = learn_distributions(df_lms, df_eod)
    print(f"   Courses:    {len(dist['progress_per_course'])}")
    print(f"   Activities: {len(dist['activities'])}")
    print(f"   Hours:      μ={dist['hours']['mean']:.2f}, σ={dist['hours']['std']:.2f}")

    # Step 3: Generate names
    print(f"\n👤 Generating {n} unique intern names...")
    names = generate_unique_names(n, rng)
    print(f"   Sample: {names[0][0]} {names[0][1]}, {names[1][0]} {names[1][1]}, ...")

    # Step 4: Generate LMS
    print(f"\n📚 Generating LMS records ({n} interns × {len(COURSE_CONFIGS)} courses)...")
    lms_dfs = generate_synthetic_lms(names, dist, rng)
    total_lms = sum(len(df) for df in lms_dfs.values())
    print(f"   Total LMS records: {total_lms}")

    # Step 5: Generate EOD
    print(f"\n⏱️  Generating EOD activity records...")
    df_eod_syn = generate_synthetic_eod(names, dist, lms_dfs, rng)
    print(f"   Total EOD records: {len(df_eod_syn)}")

    # Step 6: Export
    print(f"\n💾 Exporting to {args.output_dir}/")
    export_data(lms_dfs, df_eod_syn, args.output_dir)

    # Step 7: Validate (optional)
    if args.validate:
        all_ok = validate_output(lms_dfs, df_eod_syn, n)
        if not all_ok:
            sys.exit(1)

    # Summary statistics
    print("\n" + "═" * 48)
    print("📈  Generation Summary")
    print("═" * 48)
    print(f"  Interns generated  : {n}")
    print(f"  LMS records        : {total_lms}")
    print(f"  EOD records        : {len(df_eod_syn)}")
    print(f"  Courses covered    : {len(COURSE_CONFIGS)}")
    print(f"  Activities covered : {len(dist['activities'])}")

    # Show status distribution
    all_lms = pd.concat(lms_dfs.values())
    status_dist = all_lms['Overall Status'].value_counts()
    print(f"\n  Status Distribution:")
    for status, count in status_dist.items():
        pct = count / len(all_lms) * 100
        print(f"    {status:15s} : {count:4d} ({pct:.1f}%)")

    # Show progress statistics
    progs = all_lms['Progress (%)'].astype(str).str.rstrip('%').astype(float)
    print(f"\n  Progress Statistics:")
    print(f"    Mean   : {progs.mean():.1f}%")
    print(f"    Median : {progs.median():.1f}%")
    print(f"    Std    : {progs.std():.1f}%")

    # Hours statistics
    print(f"\n  EOD Hours Statistics:")
    print(f"    Mean   : {df_eod_syn['Hours'].mean():.2f}")
    print(f"    Std    : {df_eod_syn['Hours'].std():.2f}")
    print(f"    Range  : {df_eod_syn['Hours'].min():.1f} – {df_eod_syn['Hours'].max():.1f}")

    print(f"\n✅ Done! Files saved to: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
