import pandas as pd

def build_star_schema(lms_df, eod_df):
    # --- DIM_INTERN ---
    dim_intern = pd.DataFrame(lms_df['User Name'].unique(), columns=['Intern_Name'])
    dim_intern['Intern_ID'] = range(1001, 1001 + len(dim_intern))

    # --- DIM_COURSE ---
    dim_course = lms_df[['Course Name', 'Start Date', 'End Date']].drop_duplicates()
    dim_course['Course_ID'] = range(501, 501 + len(dim_course))

    # --- FACT_EFFORT (From EOD Data) ---
    # Joining with Dim_Intern to get IDs
    fact_effort = eod_df.merge(dim_intern, left_on='Full Name', right_on='Intern_Name')
    fact_effort = fact_effort[['Date', 'Intern_ID', 'Activity', 'Hours']]

    # --- FACT_PROGRESS (From LMS Data) ---
    fact_progress = lms_df.merge(dim_intern, left_on='User Name', right_on='Intern_Name')
    fact_progress = fact_progress.merge(dim_course, on=['Course Name', 'Start Date', 'End Date'])
    
    # Final Fact Table Structure
    fact_progress = fact_progress[[
        'Intern_ID', 'Course_ID', 'Progress_Numeric', 
        'Reviewed', 'Total_Assg', 'Overall Status'
    ]]

    return dim_intern, dim_course, fact_effort, fact_progress

# This structure allows mentors to run lightning-fast queries 
# like: "Show me all Interns (Dim_Intern) in SQL (Dim_Course) 
# who have more than 50 hours logged (Fact_Effort)."