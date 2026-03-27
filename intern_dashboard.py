import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ACTIVITY_EMOJI = {
    'PySpark Session':          '⚡',
    'Data Engineering Course':  '🏗️',
    'NumPy Practice':           '🔢',
    'Advanced SQL Practice':    '🗄️',
    'Power BI Dashboard Work':  '📊',
    'PySpark LMS Learning':     '📚',
    'Pandas Exam Preparation':  '🐼',
    'PL/SQL Concepts':          '🔍',
    'Project Research':         '🔬',
    'Pandas Practice':          '🐼',
    'Spark Architecture Study': '🏛️',
    'SQL Revision':             '🔄',
}

ACTIVITY_COLOR = {
    'PySpark Session':          '#207F8D',
    'Data Engineering Course':  '#2ECC8E',
    'NumPy Practice':           '#F4A623',
    'Advanced SQL Practice':    '#E8593C',
    'Power BI Dashboard Work':  '#A855F7',
    'PySpark LMS Learning':     '#06B6D4',
    'Pandas Exam Preparation':  '#F97316',
    'PL/SQL Concepts':          '#EF4444',
    'Project Research':         '#8B5CF6',
    'Pandas Practice':          '#FB923C',
    'Spark Architecture Study': '#10B981',
    'SQL Revision':             '#3B82F6',
}

STATUS_COLOR = {
    'Completed':   ('#D1FAE5', '#065F46', '✅'),
    'In Progress': ('#FEF3C7', '#92400E', '🔄'),
    'Not started': ('#FEE2E2', '#991B1B', '⏳'),
    'Not Started': ('#FEE2E2', '#991B1B', '⏳'),
}

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Nunito, sans-serif', color='#1a3c40'),
)

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_cohort_avg_progress(fact_lms, dim_course):
    merged = fact_lms.merge(dim_course, on='Course_ID')
    return merged.groupby('Course Name')['Progress_Numeric'].mean().to_dict()

def get_cohort_total_hours(fact_activity):
    return fact_activity.groupby('Intern_ID')['Hours'].sum()

def get_intern_rank(intern_id, cohort_hours_series):
    sorted_series = cohort_hours_series.sort_values(ascending=False).reset_index()
    if intern_id not in sorted_series['Intern_ID'].values:
        return "N/A"
    rank = sorted_series[sorted_series['Intern_ID'] == intern_id].index[0] + 1
    total = len(sorted_series)
    return f"#{rank} of {total}"

def get_skill_intensity_score(intern_lms, intern_hours, cohort_hours_series):
    avg_progress = intern_lms['Progress_Numeric'].mean() / 100
    my_hours = intern_hours
    cohort_median = cohort_hours_series.median()
    hours_ratio = min(my_hours / cohort_median, 1.5) / 1.5
    assg_ratio = (intern_lms['Reviewed'].sum() / max(intern_lms['Total_Assg'].sum(), 1))
    score = (avg_progress * 0.5) + (hours_ratio * 0.3) + (assg_ratio * 0.2)
    return round(score * 100, 1)

def get_cohort_skill_intensity(fact_lms, cohort_hours_series):
    if cohort_hours_series.empty or fact_lms.empty:
        return 50.0
    
    # 1. Avg Progress per intern
    avg_prog = fact_lms.groupby('Intern_ID')['Progress_Numeric'].mean() / 100
    
    # 2. Hours ratio
    cohort_median = cohort_hours_series.median()
    if cohort_median == 0:
        cohort_median = 1
    hours_ratio = (cohort_hours_series / cohort_median).clip(upper=1.5) / 1.5
    
    # 3. Assg ratio
    assg_sums = fact_lms.groupby('Intern_ID')[['Reviewed', 'Total_Assg']].sum()
    assg_sums['Total_Assg'] = assg_sums['Total_Assg'].replace(0, 1)
    assg_ratio = assg_sums['Reviewed'] / assg_sums['Total_Assg']
    
    # Merge and compute
    df = pd.DataFrame({'avg_prog': avg_prog, 'hours_ratio': hours_ratio, 'assg_ratio': assg_ratio}).dropna()
    if df.empty:
        return 50.0
        
    scores = (df['avg_prog'] * 0.5) + (df['hours_ratio'] * 0.3) + (df['assg_ratio'] * 0.2)
    return round(scores.mean() * 100, 1)

def get_top_activity_this_week(intern_activity):
    last_date = intern_activity['Date'].max()
    week_ago = last_date - timedelta(days=7)
    recent = intern_activity[intern_activity['Date'] >= week_ago]
    if recent.empty:
        return None, 0
    top = recent.groupby('Activity')['Hours'].sum().idxmax()
    hrs = recent.groupby('Activity')['Hours'].sum().max()
    return top, round(hrs, 1)

def prediction_badge(intern_lms, intern_hours, cohort_hours_series):
    all_progress = intern_lms['Progress_Numeric'].tolist()
    above_70 = all(p >= 70 for p in all_progress)
    above_median = intern_hours >= cohort_hours_series.median()
    completed_count = (intern_lms['Overall Status'] == 'Completed').sum()
    if above_70 and above_median:
        return "🎯 On track to complete all courses!", "success"
    elif completed_count == len(intern_lms):
        return "🏆 All courses completed! Star intern!", "success"
    elif intern_hours < cohort_hours_series.quantile(0.25):
        return "⚠️ Low effort detected — time to push harder!", "error"
    else:
        return "📈 Progressing steadily. Keep the momentum!", "info"

# ─────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────
def build_radar_chart(intern_lms, cohort_avg, dim_course, intern_name):
    course_map = dict(zip(dim_course['Course_ID'], dim_course['Course Name']))
    courses = list(cohort_avg.keys())
    short_names = [c.replace('Data Processing using ', '').replace(' & ', '/') for c in courses]

    intern_vals = []
    for course in courses:
        cid_row = dim_course[dim_course['Course Name'] == course]
        if cid_row.empty:
            intern_vals.append(0)
            continue
        cid = cid_row['Course_ID'].values[0]
        row = intern_lms[intern_lms['Course_ID'] == cid]
        intern_vals.append(row['Progress_Numeric'].values[0] if not row.empty else 0)

    cohort_vals = [cohort_avg.get(c, 0) for c in courses]

    # Close the polygon
    intern_vals += [intern_vals[0]]
    cohort_vals += [cohort_vals[0]]
    short_names += [short_names[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cohort_vals, theta=short_names, fill='toself',
        name='Cohort Avg',
        fillcolor='rgba(136,136,136,0.15)',
        line=dict(color='#888', width=2, dash='dash'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=intern_vals, theta=short_names, fill='toself',
        name=intern_name.split()[0],
        fillcolor='rgba(32,127,141,0.3)',
        line=dict(color='#207F8D', width=3),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Nunito, sans-serif', color='#1a3c40'),
        polar=dict(
            bgcolor='rgba(202,221,224,0.2)',
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10), gridcolor='rgba(0,0,0,0.1)'),
            angularaxis=dict(tickfont=dict(size=11, color='#1a3c40')),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        height=380,
        margin=dict(t=20, b=60, l=40, r=40),
    )
    return fig


def build_stacked_area(intern_activity):
    daily = intern_activity.groupby(['Date', 'Activity'])['Hours'].sum().reset_index()
    pivot = daily.pivot(index='Date', columns='Activity', values='Hours').fillna(0).reset_index()
    activities = [c for c in pivot.columns if c != 'Date']

    fig = go.Figure()
    for act in activities:
        color = ACTIVITY_COLOR.get(act, '#207F8D')
        fig.add_trace(go.Scatter(
            x=pivot['Date'], y=pivot[act],
            name=f"{ACTIVITY_EMOJI.get(act, '•')} {act}",
            stackgroup='one',
            mode='lines',
            line=dict(width=0.5, color=color),
            fillcolor=color.replace('#', 'rgba(').rstrip(')') if False else color,
            hovertemplate='%{y:.1f} hrs<extra>' + act + '</extra>',
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=320,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', title='Hours'),
        legend=dict(orientation='h', font=dict(size=10), yanchor='top', y=-0.25, xanchor='center', x=0.5),
        hovermode='x unified',
    )
    return fig


def build_streak_heatmap(intern_activity):
    daily_hrs = intern_activity.groupby('Date')['Hours'].sum().reset_index()
    daily_hrs['week'] = daily_hrs['Date'].dt.isocalendar().week.astype(int)
    daily_hrs['dow'] = daily_hrs['Date'].dt.dayofweek  # 0=Mon

    min_week = int(daily_hrs['week'].min())
    max_week = int(daily_hrs['week'].max())
    weeks = list(range(min_week, max_week + 1))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    z = np.zeros((7, len(weeks)))
    for _, row in daily_hrs.iterrows():
        wi = weeks.index(int(row['week']))
        z[int(row['dow']), wi] = row['Hours']

    # Build week labels
    week_labels = [f"W{w}" for w in weeks]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=week_labels,
        y=days,
        colorscale=[
            [0.0, '#E8F4F6'],
            [0.01, '#B3D9DF'],
            [0.3, '#5BB5C4'],
            [0.6, '#207F8D'],
            [1.0, '#0F3F47'],
        ],
        showscale=True,
        colorbar=dict(title='Hours', thickness=12, len=0.8),
        hovertemplate='Week %{x}, %{y}<br>Hours: %{z:.1f}<extra></extra>',
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=220,
        xaxis=dict(side='top', tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
        margin=dict(t=30, b=10, l=50, r=60),
    )
    return fig


def build_benchmark_bar(intern_activity, fact_activity, dim_intern, intern_id):
    all_acts = fact_activity.groupby(['Intern_ID', 'Activity'])['Hours'].sum().reset_index()
    n_interns = all_acts['Intern_ID'].nunique()
    cohort_avg = all_acts.groupby('Activity')['Hours'].sum() / n_interns

    my_acts = intern_activity.groupby('Activity')['Hours'].sum()
    all_activities = cohort_avg.index.tolist()

    my_vals = [my_acts.get(a, 0) for a in all_activities]
    coh_vals = [cohort_avg.get(a, 0) for a in all_activities]

    # Sort by my hours descending
    order = sorted(range(len(all_activities)), key=lambda i: my_vals[i], reverse=True)
    sorted_acts = [all_activities[i] for i in order]
    sorted_me   = [my_vals[i] for i in order]
    sorted_coh  = [coh_vals[i] for i in order]
    labels = [f"{ACTIVITY_EMOJI.get(a,'•')} {a}" for a in sorted_acts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=sorted_me, name='Me', orientation='h',
        marker_color='#207F8D',
        hovertemplate='%{x:.1f} hrs<extra>Me</extra>',
    ))
    fig.add_trace(go.Bar(
        y=labels, x=sorted_coh, name='Cohort Avg', orientation='h',
        marker_color='rgba(136,136,136,0.35)',
        hovertemplate='%{x:.1f} hrs<extra>Cohort Avg</extra>',
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode='overlay',
        height=380,
        xaxis=dict(title='Total Hours', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=20, b=20, l=220, r=20),
    )
    return fig


def build_score_gauge(score, cohort_avg_score):
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=score,
        delta={'reference': cohort_avg_score, 'valueformat': '.1f',
               'increasing': {'color': '#207F8D'}, 'decreasing': {'color': '#E8593C'}},
        number={'suffix': '', 'font': {'size': 36, 'color': '#0F3F47', 'family': 'Nunito'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#1a3c40'},
            'bar': {'color': '#207F8D', 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],  'color': 'rgba(232,89,60,0.15)'},
                {'range': [40, 70], 'color': 'rgba(244,166,35,0.15)'},
                {'range': [70, 100],'color': 'rgba(32,127,141,0.15)'},
            ],
            'threshold': {
                'line': {'color': '#888', 'width': 2},
                'thickness': 0.75,
                'value': cohort_avg_score,
            },
        },
        title={'text': 'Skill Intensity Score', 'font': {'size': 13, 'color': '#5a7d82', 'family': 'Nunito'}},
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=220,
        margin=dict(t=30, b=0, l=20, r=20),
    )
    return fig


def build_course_progress_line(intern_lms, dim_course):
    merged = intern_lms.merge(dim_course, on='Course_ID')
    merged = merged.sort_values('Progress_Numeric', ascending=False)
    colors = ['#207F8D' if s == 'Completed' else '#F4A623' if s == 'In Progress' else '#E8593C'
              for s in merged['Overall Status']]
    short = merged['Course Name'].str.replace('Data Processing using ', '').str.replace(' & ', '/')

    fig = go.Figure(go.Bar(
        x=merged['Progress_Numeric'],
        y=short,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.0f}%" for v in merged['Progress_Numeric']],
        textposition='inside',
        hovertemplate='%{y}<br>Progress: %{x:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=200,
        xaxis=dict(range=[0, 105], showgrid=False, title=''),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(t=10, b=10, l=170, r=20),
    )
    return fig


# ─────────────────────────────────────────────
#  MOTIVATIONAL BANNER
# ─────────────────────────────────────────────
def show_banner(intern_lms, intern_hours, cohort_hours_series):
    avg_prog = intern_lms['Progress_Numeric'].mean()
    not_started = intern_lms[intern_lms['Overall Status'].isin(['Not started', 'Not Started'])]
    cohort_median = cohort_hours_series.median()

    if avg_prog >= 80:
        st.success(f"🚀 **Top Tier Alert!** Your average progress is **{avg_prog:.0f}%** — you're outpacing most of your cohort. Keep crushing it!")
    elif not not_started.empty:
        courses = ', '.join(not_started.merge(st.session_state.get('dim_course_cache', pd.DataFrame()), on='Course_ID', how='left')['Course Name'].fillna('a course').tolist()) if False else "one or more courses"
        st.warning(f"⚡ **Heads Up!** You haven't started **{len(not_started)} course(s)** yet. Every day counts — time to dive in!")
    elif intern_hours < cohort_median:
        gap = round(cohort_median - intern_hours, 1)
        st.warning(f"📈 **Almost There!** You're **{gap} hrs** below the cohort median. A little extra daily effort will move your rank significantly!")
    else:
        st.info(f"✅ **On Track!** You've logged **{intern_hours:.0f} hrs** and your effort is above the cohort median. Consistency is your superpower!")


# ─────────────────────────────────────────────
#  COURSE SCOREBOARD
# ─────────────────────────────────────────────
def show_course_scoreboard(intern_lms, dim_course):
    merged = intern_lms.merge(dim_course, on='Course_ID').sort_values('Progress_Numeric', ascending=False)
    for _, row in merged.iterrows():
        status = row['Overall Status']
        bg, fg, icon = STATUS_COLOR.get(status, ('#F3F4F6', '#374151', '❓'))
        prog = row['Progress_Numeric']
        assg = f"{row['Reviewed']}/{row['Total_Assg']}"
        short_name = row['Course Name'].replace('Data Processing using ', '').replace(' & Pandas', '/Pandas')

        col_name, col_bar, col_badge, col_assg = st.columns([3, 4, 2, 1])
        with col_name:
            st.markdown(f"<p style='margin:0;padding:8px 0;font-weight:700;font-size:0.88rem;color:#0f3f47'>{short_name}</p>", unsafe_allow_html=True)
        with col_bar:
            st.progress(int(prog) / 100)
        with col_badge:
            st.markdown(
                f"<span style='background:{bg};color:{fg};padding:3px 10px;border-radius:20px;"
                f"font-size:0.78rem;font-weight:700;white-space:nowrap'>{icon} {status}</span>",
                unsafe_allow_html=True
            )
        with col_assg:
            st.markdown(f"<p style='text-align:center;margin:0;padding:8px 0;font-size:0.88rem;color:#5a7d82'>{assg}</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:2px 0;opacity:0.15'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────
def show_intern_dashboard(fact_lms, fact_activity, dim_intern, dim_course):

    # ── Sidebar intern selector ───────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Intern View")
    intern_names = sorted(dim_intern['Intern_Name'].tolist())
    selected_intern = st.sidebar.selectbox("Select Your Name", intern_names, key="intern_select")

    focus_mode = st.sidebar.toggle("🎯 Focus Mode (hide comparisons)", value=False, key="focus_mode")

    intern_id = dim_intern[dim_intern['Intern_Name'] == selected_intern]['Intern_ID'].values[0]
    intern_lms = fact_lms[fact_lms['Intern_ID'] == intern_id].copy()
    intern_activity = fact_activity[fact_activity['Intern_ID'] == intern_id].copy()

    # ── Pre-compute cohort stats ──────────────
    cohort_hours = get_cohort_total_hours(fact_activity)
    my_total_hours = cohort_hours.get(intern_id, 0)
    rank_str = get_intern_rank(intern_id, cohort_hours)
    cohort_avg_progress = get_cohort_avg_progress(fact_lms, dim_course)
    skill_score = get_skill_intensity_score(intern_lms, my_total_hours, cohort_hours)
    cohort_skill_score = get_cohort_skill_intensity(fact_lms, cohort_hours)
    top_activity, top_hrs = get_top_activity_this_week(intern_activity)
    badge_text, badge_type = prediction_badge(intern_lms, my_total_hours, cohort_hours)

    # ── Page Header ───────────────────────────
    first_name = selected_intern.split()[0]
    completed_count = int((intern_lms['Overall Status'] == 'Completed').sum())
    total_courses = len(intern_lms)

    st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(32,127,141,0.15) 0%,rgba(202,221,224,0.4) 100%);
             border-radius:16px;padding:20px 28px;margin-bottom:16px;
             border-left:4px solid #207F8D'>
            <h1 style='margin:0;font-size:1.9rem;color:#0f3f47'>
                👋 Welcome back, <span style='color:#207F8D'>{first_name}</span>!
            </h1>
            <p style='margin:6px 0 0;color:#5a7d82;font-size:0.95rem'>
                Your personal learning intelligence dashboard &nbsp;·&nbsp; 
                {completed_count}/{total_courses} courses completed &nbsp;·&nbsp;
                Rank: <b style='color:#207F8D'>{rank_str}</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Motivational Banner ───────────────────
    show_banner(intern_lms, my_total_hours, cohort_hours)

    # ── Prediction + Top Activity row ─────────
    pb_col, ta_col = st.columns([3, 2])
    with pb_col:
        if badge_type == "success":
            st.success(badge_text)
        elif badge_type == "error":
            st.error(badge_text)
        else:
            st.info(badge_text)
    with ta_col:
        if top_activity:
            emoji = ACTIVITY_EMOJI.get(top_activity, '🔥')
            st.markdown(f"""
                <div style='background:rgba(32,127,141,0.1);border-radius:12px;padding:12px 16px;
                     border:1px solid rgba(32,127,141,0.25)'>
                    <p style='margin:0;font-size:0.75rem;color:#5a7d82;text-transform:uppercase;letter-spacing:0.5px'>
                        🔥 Top Activity This Week
                    </p>
                    <p style='margin:4px 0 0;font-size:1rem;font-weight:800;color:#0f3f47'>
                        {emoji} {top_activity}
                    </p>
                    <p style='margin:2px 0 0;font-size:0.85rem;color:#207F8D'>{top_hrs} hrs logged</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📅 Activity Timeline", "🏆 Benchmarks", "🤖 AI Study Assistant"])

    # ══════════════════════════════════════════
    #  TAB 1 — OVERVIEW
    # ══════════════════════════════════════════
    with tab1:

        # KPI strip
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("⏱️ Total Hours", f"{my_total_hours:.0f} hrs")
        with k2:
            st.metric("📗 Completed", f"{completed_count}/{total_courses}")
        with k3:
            total_reviewed = int(intern_lms['Reviewed'].sum())
            total_assg = int(intern_lms['Total_Assg'].sum())
            st.metric("📝 Assignments", f"{total_reviewed}/{total_assg}")
        with k4:
            st.metric("🏅 Cohort Rank", rank_str)
        with k5:
            avg_p = intern_lms['Progress_Numeric'].mean()
            cohort_ap = np.mean(list(cohort_avg_progress.values()))
            st.metric("📈 Avg Progress", f"{avg_p:.0f}%", delta=f"{avg_p - cohort_ap:+.1f}% vs cohort")

        st.markdown("<br>", unsafe_allow_html=True)

        # Radar + Gauge side by side
        r_col, g_col = st.columns([3, 2])
        with r_col:
            st.markdown("#### 🎯 Learning Journey — vs Cohort")
            if not focus_mode:
                st.plotly_chart(build_radar_chart(intern_lms, cohort_avg_progress, dim_course, selected_intern),
                                use_container_width=True)
            else:
                fig_solo = build_course_progress_line(intern_lms, dim_course)
                st.plotly_chart(fig_solo, use_container_width=True)

        with g_col:
            st.markdown("#### ⚡ Skill Intensity Score")
            st.plotly_chart(build_score_gauge(skill_score, cohort_skill_score),
                            use_container_width=True)
            st.markdown(f"""
                <div style='text-align:center;padding:8px'>
                    <p style='margin:0;font-size:0.78rem;color:#5a7d82'>
                        Weighted score: 50% progress · 30% effort · 20% assignments<br>
                        <b>Grey marker = cohort average ({cohort_skill_score:.1f})</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Course Scoreboard
        st.markdown("#### 📋 Course Scoreboard")
        hdr1, hdr2, hdr3, hdr4 = st.columns([3, 4, 2, 1])
        hdr1.markdown("<p style='font-size:0.78rem;color:#5a7d82;font-weight:700;margin:0'>COURSE</p>", unsafe_allow_html=True)
        hdr2.markdown("<p style='font-size:0.78rem;color:#5a7d82;font-weight:700;margin:0'>PROGRESS</p>", unsafe_allow_html=True)
        hdr3.markdown("<p style='font-size:0.78rem;color:#5a7d82;font-weight:700;margin:0'>STATUS</p>", unsafe_allow_html=True)
        hdr4.markdown("<p style='font-size:0.78rem;color:#5a7d82;font-weight:700;margin:0;text-align:center'>ASSG</p>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0 8px;opacity:0.25'>", unsafe_allow_html=True)
        show_course_scoreboard(intern_lms, dim_course)

    # ══════════════════════════════════════════
    #  TAB 2 — ACTIVITY TIMELINE
    # ══════════════════════════════════════════
    with tab2:

        if intern_activity.empty:
            st.warning("No activity logs found for this intern.")
        else:
            area_col, stat_col = st.columns([4, 1])
            with area_col:
                st.markdown("#### 🧬 Activity DNA — Daily Effort Breakdown")
                st.plotly_chart(build_stacked_area(intern_activity), use_container_width=True)

            with stat_col:
                st.markdown("#### 📊 Distribution")
                act_summary = intern_activity.groupby('Activity')['Hours'].sum().sort_values(ascending=False)
                total_h = act_summary.sum()
                for act, hrs in act_summary.items():
                    pct = hrs / total_h * 100
                    emoji = ACTIVITY_EMOJI.get(act, '•')
                    st.markdown(f"""
                        <div style='margin-bottom:8px'>
                            <p style='margin:0;font-size:0.78rem;color:#1a3c40;font-weight:600'>{emoji} {act[:18]}</p>
                            <p style='margin:0;font-size:0.72rem;color:#5a7d82'>{hrs:.0f} hrs · {pct:.0f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📆 Study Streak Calendar — Jan to Mar 2026")
            st.markdown("<p style='font-size:0.85rem;color:#5a7d82;margin-top:-8px'>Darker = more hours logged that day</p>", unsafe_allow_html=True)
            st.plotly_chart(build_streak_heatmap(intern_activity), use_container_width=True)

            # Streak stats
            s1, s2, s3, s4 = st.columns(4)
            daily_hrs = intern_activity.groupby('Date')['Hours'].sum()
            active_days = (daily_hrs > 0).sum()
            max_day = daily_hrs.max()
            avg_day = daily_hrs.mean()

            # Longest streak calc
            all_dates = pd.date_range(intern_activity['Date'].min(), intern_activity['Date'].max())
            active_set = set(daily_hrs[daily_hrs > 0].index)
            longest_streak = cur = 0
            for d in all_dates:
                if d in active_set:
                    cur += 1
                    longest_streak = max(longest_streak, cur)
                else:
                    cur = 0

            s1.metric("📅 Active Days", f"{active_days}")
            s2.metric("🔥 Longest Streak", f"{longest_streak} days")
            s3.metric("⚡ Best Day", f"{max_day:.1f} hrs")
            s4.metric("📊 Daily Avg", f"{avg_day:.1f} hrs")

    # ══════════════════════════════════════════
    #  TAB 3 — BENCHMARKS
    # ══════════════════════════════════════════
    with tab3:

        if focus_mode:
            st.info("🎯 Focus Mode is ON — cohort comparisons are hidden. Turn it off in the sidebar to see benchmarks.")
        else:
            st.markdown("#### 📊 How Do I Compare? — Activity Hours vs Cohort")
            st.markdown("<p style='font-size:0.85rem;color:#5a7d82;margin-top:-8px'>Your hours (teal) overlaid on cohort average (grey). Same activity, two bars.</p>", unsafe_allow_html=True)
            st.plotly_chart(
                build_benchmark_bar(intern_activity, fact_activity, dim_intern, intern_id),
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Course-level comparison table
            st.markdown("#### 📚 Course Progress vs Cohort Average")
            merged = intern_lms.merge(dim_course, on='Course_ID')
            cmp_rows = []
            for _, row in merged.iterrows():
                c = row['Course Name']
                my_p = row['Progress_Numeric']
                coh_p = cohort_avg_progress.get(c, 0)
                delta = my_p - coh_p
                arrow = "🟢 +" if delta >= 0 else "🔴 "
                cmp_rows.append({
                    'Course': c.replace('Data Processing using ', ''),
                    'My Progress': f"{my_p:.0f}%",
                    'Cohort Avg': f"{coh_p:.0f}%",
                    'Gap': f"{arrow}{abs(delta):.1f}%",
                    'Status': row['Overall Status'],
                })
            cmp_df = pd.DataFrame(cmp_rows)
            st.dataframe(
                cmp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Course': st.column_config.TextColumn('Course'),
                    'My Progress': st.column_config.TextColumn('My Progress'),
                    'Cohort Avg': st.column_config.TextColumn('Cohort Avg'),
                    'Gap': st.column_config.TextColumn('vs Cohort'),
                    'Status': st.column_config.TextColumn('Status'),
                }
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Hours leaderboard (anonymized)
            st.markdown("#### 🏆 Hours Leaderboard (Anonymized)")
            st.markdown("<p style='font-size:0.85rem;color:#5a7d82;margin-top:-8px'>Your position among all active interns. Other names hidden.</p>", unsafe_allow_html=True)

            sorted_ids = cohort_hours.sort_values(ascending=False).reset_index()
            sorted_ids.columns = ['Intern_ID', 'Total_Hours']
            my_rank_int = sorted_ids[sorted_ids['Intern_ID'] == intern_id].index[0] + 1 if intern_id in sorted_ids['Intern_ID'].values else 0

            lb_data = []
            for i, row in sorted_ids.iterrows():
                rank_num = i + 1
                if row['Intern_ID'] == intern_id:
                    name_display = f"⭐ {selected_intern} (You)"
                    row_style = True
                else:
                    name_display = f"Intern #{rank_num}" if rank_num != my_rank_int else selected_intern
                    row_style = False
                lb_data.append({
                    'Rank': f"#{rank_num}",
                    'Name': name_display,
                    'Total Hours': f"{row['Total_Hours']:.0f} hrs",
                    'Bar': row['Total_Hours'],
                })
            lb_df = pd.DataFrame(lb_data)
            st.dataframe(
                lb_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Rank': st.column_config.TextColumn('Rank', width='small'),
                    'Name': st.column_config.TextColumn('Intern'),
                    'Total Hours': st.column_config.TextColumn('Hours'),
                    'Bar': st.column_config.ProgressColumn('Effort Bar', min_value=0, max_value=float(sorted_ids['Total_Hours'].max()), format='%.0f'),
                }
            )

    # ══════════════════════════════════════════
    #  TAB 4 — STUDY ASSISTANT (GenAI)
    # ══════════════════════════════════════════
    with tab4:
        st.markdown("#### 🤖 AI Study Assistant")
        st.markdown("<p style='font-size:0.85rem;color:#5a7d82;margin-top:-8px'>"
                    "Ask questions about your courses, request coding tips, or get a motivation boost!</p>", 
                    unsafe_allow_html=True)
        
        # Initialize chat history in session state specific to intern
        chat_key = f"chat_history_{intern_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [
                {"role": "assistant", "content": f"Hi {selected_intern}! How can I help you with your learning journey today?"}
            ]

        # Display chat messages
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state[chat_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message to state and display
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Generate AI response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # In a real app, pass the syllabus or current course as context
                        response = genai_service.get_study_assistant_response(prompt, selected_intern)
                        st.markdown(response)
            
            # Add assistant response to state
            st.session_state[chat_key].append({"role": "assistant", "content": response})