import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import ml_models
import genai_service

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Nunito, sans-serif', color='#1a3c40'),
)

STATUS_CFG = {
    'Completed':   ('#D1FAE5', '#065F46', '✅'),
    'In Progress': ('#FEF3C7', '#92400E', '🔄'),
    'In progress': ('#FEF3C7', '#92400E', '🔄'),
    'Not started': ('#FEE2E2', '#991B1B', '⏳'),
    'Not Started': ('#FEE2E2', '#991B1B', '⏳'),
}

RISK_COLOR = {
    'High':   ('#FEE2E2', '#991B1B', '🔴'),
    'Medium': ('#FEF3C7', '#92400E', '🟡'),
    'Low':    ('#D1FAE5', '#065F46', '🟢'),
}

COURSE_SHORT = {
    'Basic Python Programming':              'Python',
    'Basic SQL':                             'SQL',
    'Data Processing using NumPy & Pandas':  'NumPy/Pandas',
    'Data Processing using Pyspark':         'PySpark',
}

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


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_mentor_intern_ids(mentor_name, fact_lms):
    return fact_lms[fact_lms['Mentor Name'].str.contains(mentor_name, na=False)]['Intern_ID'].unique()


def get_mentor_courses(mentor_name, fact_lms, dim_course):
    """Return list of course names that this mentor's interns are enrolled in."""
    intern_ids = get_mentor_intern_ids(mentor_name, fact_lms)
    course_ids = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)]['Course_ID'].unique()
    courses = dim_course[dim_course['Course_ID'].isin(course_ids)]['Course Name'].unique().tolist()
    return sorted(courses)


def compute_dropout_risk(intern_ids, fact_lms, fact_activity, dim_intern):
    """
    Risk Score (0–100) = weighted combination of:
      40% avg progress deficit (below 50% = max risk)
      30% effort trend (hours declining week-over-week)
      20% assignment completion lag
      10% inactivity recency
    Returns DataFrame with Intern_ID, Intern_Name, Risk_Score, Risk_Level, risk components
    """
    if len(intern_ids) == 0:
        return pd.DataFrame()
        
    max_date = fact_activity['Date'].max()
    
    # Filter datasets once
    f_lms = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)]
    f_act = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)]
    
    # 1. Component 1: Progress deficit (0–40)
    avg_prog = f_lms.groupby('Intern_ID')['Progress_Numeric'].mean().fillna(0)
    prog_risk = ((50 - avg_prog) / 50).clip(lower=0) * 40

    # 2. Component 2: Effort trend (0–30)
    d14 = max_date - timedelta(days=14)
    d28 = max_date - timedelta(days=28)
    recent_act = f_act[f_act['Date'] >= d14].groupby('Intern_ID')['Hours'].sum()
    prev_act = f_act[(f_act['Date'] >= d28) & (f_act['Date'] < d14)].groupby('Intern_ID')['Hours'].sum()
    
    effort_df = pd.DataFrame({'recent': recent_act, 'prev': prev_act}).fillna(0)
    
    def calc_effort_risk(row):
        w_recent, w_prev = row['recent'], row['prev']
        if w_prev > 0:
            trend_pct = (w_recent - w_prev) / w_prev
            return max(0, -trend_pct) * 30
        return 15 if w_recent > 0 else 30
        
    effort_risk = effort_df.apply(calc_effort_risk, axis=1) if not effort_df.empty else pd.Series(30, index=intern_ids)
    
    # 3. Component 3: Assignment lag (0–20)
    assg_sums = f_lms.groupby('Intern_ID')[['Total_Assg', 'Reviewed']].sum()
    lag_ratio = (1 - (assg_sums['Reviewed'] / assg_sums['Total_Assg'].clip(lower=1))).clip(lower=0)
    assg_risk = lag_ratio * 20

    # 4. Component 4: Recency (0–10)
    if not f_act.empty:
        last_act = f_act.groupby('Intern_ID')['Date'].max()
        days_since = (max_date - last_act).dt.days
        recency_risk = (days_since * 1.5).clip(upper=10)
    else:
        recency_risk = pd.Series(10, index=intern_ids)

    # Combine all components
    df = pd.DataFrame({
        'Avg_Progress': avg_prog,
        'Prog_Risk': prog_risk,
        'Effort_Risk': effort_risk,
        'Assg_Lag': assg_risk,
        'Recency_Risk': recency_risk
    }).reindex(intern_ids).fillna({
        'Avg_Progress': 0, 'Prog_Risk': 40, 'Effort_Risk': 30, 'Assg_Lag': 20, 'Recency_Risk': 10
    })
    
    df['Risk_Score'] = (df['Prog_Risk'] + df['Effort_Risk'] + df['Assg_Lag'] + df['Recency_Risk']).round(1)
    
    def get_level(s):
        return 'High' if s >= 50 else ('Medium' if s >= 25 else 'Low')
        
    df['Risk_Level'] = df['Risk_Score'].apply(get_level)
    
    # Join Names and finalize
    df = df.reset_index().rename(columns={'index': 'Intern_ID'})
    df = df.merge(dim_intern[['Intern_ID', 'Intern_Name']], on='Intern_ID', how='left')
    df['Intern_Name'] = df['Intern_Name'].fillna('Unknown')
    
    # Apply rounding
    for col in ['Avg_Progress', 'Effort_Risk', 'Assg_Lag', 'Recency_Risk', 'Prog_Risk']:
        df[col] = df[col].round(1)
        
    return df.sort_values('Risk_Score', ascending=False).reset_index(drop=True)


def compute_weekly_trend(fact_activity, intern_ids):
    iact = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)].copy()
    if iact.empty:
        return pd.DataFrame()
    iact['week'] = iact['Date'].dt.isocalendar().week.astype(int)
    pivot = iact.groupby(['Intern_ID', 'week'])['Hours'].sum().unstack(fill_value=0)
    return pivot


def pending_reviews_df(fact_lms, intern_ids, dim_intern, dim_course):
    sub = fact_lms[
        (fact_lms['Intern_ID'].isin(intern_ids)) &
        (fact_lms['Reviewed'] < fact_lms['Total_Assg'])
    ].copy()
    sub = sub.merge(dim_intern, on='Intern_ID').merge(dim_course, on='Course_ID')
    sub['Pending'] = sub['Total_Assg'] - sub['Reviewed']
    sub['Course_Short'] = sub['Course Name'].map(COURSE_SHORT).fillna(sub['Course Name'])
    return sub[['Intern_Name', 'Course_Short', 'Reviewed', 'Total_Assg', 'Pending']].sort_values('Pending', ascending=False)


def _ensure_score_cols(fact_lms):
    df = fact_lms.copy()
    for col, default in [('KC_scored', 0.0), ('KC_total', 1.0),
                         ('Test_scored', 0.0), ('Test_total', 1.0)]:
        if col not in df.columns:
            df[col] = default
    return df


def score_summary(fact_lms, intern_ids, dim_course):
    sub = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)].copy()
    sub = sub.merge(dim_course, on='Course_ID')

    if 'KC_pct' in sub.columns:
        pass
    elif 'KC_scored' in sub.columns and 'KC_total' in sub.columns:
        sub['KC_pct'] = (sub['KC_scored'] / sub['KC_total'].replace(0, 1) * 100).round(1)
    elif 'Overall Knowledge Check' in sub.columns:
        kc = sub['Overall Knowledge Check'].astype(str).str.split('/', expand=True)
        ks = pd.to_numeric(kc[0].str.strip(), errors='coerce').fillna(0)
        kt = pd.to_numeric(kc[1].str.strip(), errors='coerce').fillna(1)
        sub['KC_pct'] = (ks / kt.replace(0, 1) * 100).round(1)
    else:
        sub['KC_pct'] = 0.0

    if 'Test_pct' in sub.columns:
        pass
    elif 'Test_scored' in sub.columns and 'Test_total' in sub.columns:
        sub['Test_pct'] = (sub['Test_scored'] / sub['Test_total'].replace(0, 1) * 100).round(1)
    elif 'Overall Test' in sub.columns:
        ts = sub['Overall Test'].astype(str).str.split('/', expand=True)
        tsc = pd.to_numeric(ts[0].str.strip(), errors='coerce').fillna(0)
        ttt = pd.to_numeric(ts[1].str.strip(), errors='coerce').fillna(1)
        sub['Test_pct'] = (tsc / ttt.replace(0, 1) * 100).round(1)
    else:
        sub['Test_pct'] = 0.0

    return sub


# ─────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────

def chart_cohort_progress_heatmap(fact_lms, intern_ids, dim_intern, dim_course):
    sub = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)].copy()
    sub = sub.merge(dim_intern, on='Intern_ID').merge(dim_course, on='Course_ID')
    sub['Course_Short'] = sub['Course Name'].map(COURSE_SHORT).fillna(sub['Course Name'])
    pivot = sub.pivot_table(index='Intern_Name', columns='Course_Short',
                            values='Progress_Numeric', aggfunc='mean').fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  '#FEE2E2'],
            [0.3,  '#FEF3C7'],
            [0.6,  '#D1FAE5'],
            [1.0,  '#065F46'],
        ],
        zmin=0, zmax=100,
        text=[[f'{v:.0f}%' for v in row] for row in pivot.values],
        texttemplate='%{text}',
        textfont=dict(size=11, color='#1a3c40'),
        colorbar=dict(title='%', thickness=12, len=0.8),
        hovertemplate='%{y}<br>%{x}: %{z:.1f}%<extra></extra>',
        xgap=3, ygap=3,
    ))
    n = len(pivot.index)
    fig.update_layout(
        **CHART_BASE,
        height=max(300, n * 26 + 80),
        margin=dict(t=10, b=10, l=160, r=60),
        xaxis=dict(side='top', tickfont=dict(size=12, color='#0f3f47')),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
    )
    return fig


def chart_risk_scatter(risk_df):
    color_map = {'High': '#E8593C', 'Medium': '#F4A623', 'Low': '#207F8D'}
    fig = go.Figure()
    for level in ['High', 'Medium', 'Low']:
        df = risk_df[risk_df['Risk_Level'] == level]
        if df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df['Avg_Progress'],
            y=df['Risk_Score'],
            mode='markers+text',
            name=level,
            text=df['Intern_Name'].str.split().str[0],
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=df['Assg_Lag'] * 3 + 12,
                color=color_map[level],
                opacity=0.85,
                line=dict(width=1.5, color='white'),
            ),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Progress: %{x:.0f}%<br>'
                'Risk Score: %{y:.1f}<br>'
                'Assg Lag Risk: %{customdata[1]:.1f}<extra></extra>'
            ),
            customdata=df[['Intern_Name', 'Assg_Lag']].values,
        ))
    fig.add_hline(y=50, line_dash='dash', line_color='#E8593C', opacity=0.5,
                  annotation_text='High Risk threshold', annotation_position='right')
    fig.add_hline(y=25, line_dash='dot', line_color='#F4A623', opacity=0.5,
                  annotation_text='Medium threshold', annotation_position='right')
    fig.update_layout(
        **CHART_BASE,
        height=380,
        margin=dict(t=20, b=40, l=50, r=120),
        xaxis=dict(title='Average Progress (%)', range=[-5, 105], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(title='Risk Score', range=[-5, 105], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
        legend=dict(orientation='h', y=-0.15, xanchor='center', x=0.5),
    )
    return fig


def chart_weekly_effort(fact_activity, intern_ids, dim_intern):
    iact = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)].copy()
    if iact.empty:
        return go.Figure()
    iact['week'] = iact['Date'].dt.isocalendar().week.astype(int)
    iact = iact.merge(dim_intern, on='Intern_ID')
    weekly = iact.groupby(['week', 'Intern_Name'])['Hours'].sum().reset_index()

    palette = px.colors.qualitative.Safe
    interns = weekly['Intern_Name'].unique()
    fig = go.Figure()
    for i, intern in enumerate(interns):
        df = weekly[weekly['Intern_Name'] == intern]
        fig.add_trace(go.Bar(
            x=df['week'], y=df['Hours'],
            name=intern.split()[0],
            marker_color=palette[i % len(palette)],
            hovertemplate=f'<b>{intern}</b><br>Week %{{x}}: %{{y:.1f}} hrs<extra></extra>',
        ))
    total_weekly = iact.groupby('week')['Hours'].sum().reset_index()
    fig.add_trace(go.Scatter(
        x=total_weekly['week'], y=total_weekly['Hours'],
        mode='lines+markers', name='Total',
        line=dict(color='#0f3f47', width=2.5, dash='dot'),
        marker=dict(size=6),
        yaxis='y2',
        hovertemplate='Total W%{x}: %{y:.0f} hrs<extra></extra>',
    ))
    fig.update_layout(
        **CHART_BASE,
        barmode='stack',
        height=340,
        margin=dict(t=20, b=20, l=50, r=60),
        xaxis=dict(title='Week', tickmode='linear', dtick=1),
        yaxis=dict(title='Hours (per intern)', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        yaxis2=dict(title='Total Hours', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', font=dict(size=9), y=-0.25, xanchor='center', x=0.5),
    )
    return fig


def chart_score_distribution(score_df):
    fig = go.Figure()
    for col, label, color in [
        ('KC_pct',   'Knowledge Check %', '#207F8D'),
        ('Test_pct', 'Test Score %',       '#F4A623'),
    ]:
        valid = score_df[score_df[col] > 0][col]
        if valid.empty:
            continue
        fig.add_trace(go.Box(
            y=valid, name=label,
            marker_color=color,
            boxmean='sd',
            line=dict(width=2),
            opacity=0.8,
            hovertemplate='%{y:.1f}%<extra>' + label + '</extra>',
        ))
    fig.update_layout(
        **CHART_BASE,
        height=300,
        margin=dict(t=20, b=20, l=50, r=20),
        yaxis=dict(title='Score %', range=[0, 105], showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        legend=dict(orientation='h', y=-0.2, xanchor='center', x=0.5),
    )
    return fig


def chart_activity_breakdown(fact_activity, intern_ids):
    iact = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)]
    if iact.empty:
        return go.Figure()
    agg = iact.groupby('Activity')['Hours'].sum().sort_values(ascending=False).reset_index()
    agg['Label'] = agg['Activity'].map(lambda a: f"{ACTIVITY_EMOJI.get(a,'•')} {a}")
    total = agg['Hours'].sum()
    fig = go.Figure(go.Pie(
        labels=agg['Label'], values=agg['Hours'],
        hole=0.55,
        textinfo='percent',
        textposition='inside',
        marker=dict(
            colors=px.colors.qualitative.Safe,
            line=dict(color='white', width=2),
        ),
        hovertemplate='%{label}<br>%{value:.0f} hrs (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        **CHART_BASE,
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(font=dict(size=10), orientation='v', x=1.0),
        annotations=[dict(
            text=f'<b>{total:.0f}</b><br>hrs',
            x=0.5, y=0.5, font_size=16,
            font=dict(color='#0f3f47', family='Nunito'),
            showarrow=False,
        )],
    )
    return fig


def chart_individual_progress_bars(fact_lms, intern_ids, dim_intern, dim_course):
    sub = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)].copy()
    sub = sub.merge(dim_intern, on='Intern_ID').merge(dim_course, on='Course_ID')
    sub['Course_Short'] = sub['Course Name'].map(COURSE_SHORT).fillna(sub['Course Name'])
    courses = list(COURSE_SHORT.values())
    palette = {'Python': '#207F8D', 'SQL': '#F4A623', 'NumPy/Pandas': '#2ECC8E', 'PySpark': '#E8593C'}
    fig = go.Figure()
    for course in courses:
        df = sub[sub['Course_Short'] == course]
        if df.empty:
            continue
        fig.add_trace(go.Bar(
            x=df['Intern_Name'].str.split().str[0],
            y=df['Progress_Numeric'],
            name=course,
            marker_color=palette.get(course, '#888'),
            hovertemplate='<b>%{x}</b><br>' + course + ': %{y:.0f}%<extra></extra>',
        ))
    fig.update_layout(
        **CHART_BASE,
        barmode='group',
        height=340,
        margin=dict(t=20, b=60, l=40, r=20),
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(title='Progress %', range=[0, 110], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
        legend=dict(orientation='h', y=-0.35, xanchor='center', x=0.5),
    )
    return fig


def chart_effort_vs_progress(fact_lms, fact_activity, intern_ids, dim_intern):
    hrs = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)].groupby('Intern_ID')['Hours'].sum()
    prog = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)].groupby('Intern_ID')['Progress_Numeric'].mean()
    df = pd.DataFrame({'hours': hrs, 'progress': prog}).dropna().reset_index()
    df = df.merge(dim_intern, on='Intern_ID')

    fig = go.Figure(go.Scatter(
        x=df['hours'], y=df['progress'],
        mode='markers+text',
        text=df['Intern_Name'].str.split().str[0],
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(
            size=14,
            color=df['progress'],
            colorscale=[[0, '#E8593C'], [0.5, '#F4A623'], [1, '#207F8D']],
            showscale=True,
            colorbar=dict(title='Prog %', thickness=10, len=0.6),
            line=dict(width=1.5, color='white'),
        ),
        hovertemplate='<b>%{text}</b><br>Hours: %{x:.0f}<br>Progress: %{y:.1f}%<extra></extra>',
    ))
    if len(df) > 2:
        z = np.polyfit(df['hours'], df['progress'], 1)
        x_line = np.linspace(df['hours'].min(), df['hours'].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_line, y=np.poly1d(z)(x_line),
            mode='lines', name='Trend',
            line=dict(color='rgba(0,0,0,0.25)', width=1.5, dash='dash'),
            showlegend=False,
        ))
    fig.update_layout(
        **CHART_BASE,
        height=340,
        margin=dict(t=20, b=40, l=50, r=60),
        xaxis=dict(title='Total Hours Logged', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(title='Avg Progress (%)', range=[-5, 105], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
    )
    return fig


def chart_cohort_weekly_line(fact_activity, intern_ids):
    iact = fact_activity[fact_activity['Intern_ID'].isin(intern_ids)].copy()
    if iact.empty:
        return go.Figure()
    iact['week'] = iact['Date'].dt.isocalendar().week.astype(int)
    weekly = iact.groupby('week')['Hours'].sum().reset_index()
    avg_line = weekly['Hours'].mean()

    fig = go.Figure()
    fig.add_hrect(y0=avg_line * 0.9, y1=avg_line * 1.1,
                  fillcolor='rgba(32,127,141,0.07)', line_width=0,
                  annotation_text='Target band', annotation_position='top left')
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['Hours'],
        mode='lines+markers',
        line=dict(color='#207F8D', width=3),
        marker=dict(size=8, color='white', line=dict(color='#207F8D', width=2.5)),
        fill='tozeroy',
        fillcolor='rgba(32,127,141,0.1)',
        hovertemplate='Week %{x}: %{y:.0f} hrs<extra></extra>',
    ))
    fig.add_hline(y=avg_line, line_dash='dot', line_color='#888',
                  annotation_text=f'Avg: {avg_line:.0f} hrs', annotation_position='right')
    fig.update_layout(
        **CHART_BASE,
        height=240,
        margin=dict(t=20, b=20, l=50, r=80),
        xaxis=dict(title='Week', tickmode='linear', dtick=1),
        yaxis=dict(title='Total Hours', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
    )
    return fig


def chart_risk_breakdown_bar(risk_df):
    df = risk_df.head(12).sort_values('Risk_Score')
    fig = go.Figure()
    components = [
        ('Prog_Risk',    'Progress Deficit', '#E8593C'),
        ('Effort_Risk',  'Effort Decline',   '#F4A623'),
        ('Assg_Lag',     'Assignment Lag',   '#A855F7'),
        ('Recency_Risk', 'Inactivity',       '#94A3B8'),
    ]
    for col, label, color in components:
        fig.add_trace(go.Bar(
            y=df['Intern_Name'].str.split().str[0],
            x=df[col],
            name=label,
            orientation='h',
            marker_color=color,
            hovertemplate=f'<b>%{{y}}</b><br>{label}: %{{x:.1f}}<extra></extra>',
        ))
    fig.update_layout(
        **CHART_BASE,
        barmode='stack',
        height=360,
        margin=dict(t=10, b=20, l=100, r=20),
        xaxis=dict(title='Risk Score', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation='h', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────
#  NEW ── BY-COURSE VIEW CHARTS
# ─────────────────────────────────────────────

def chart_course_intern_progress(fact_lms, intern_ids, dim_intern, course_name, dim_course):
    """
    For a single selected course: horizontal progress bars per intern,
    coloured by how far along they are.
    """
    cid_rows = dim_course[dim_course['Course Name'] == course_name]
    if cid_rows.empty:
        return go.Figure()
    cid = cid_rows['Course_ID'].values[0]

    sub = fact_lms[
        (fact_lms['Intern_ID'].isin(intern_ids)) &
        (fact_lms['Course_ID'] == cid)
    ].copy()
    sub = sub.merge(dim_intern, on='Intern_ID')
    sub = sub.sort_values('Progress_Numeric', ascending=True)

    colors = sub['Progress_Numeric'].apply(
        lambda p: '#2ECC8E' if p >= 70 else ('#F4A623' if p >= 40 else '#E8593C')
    )

    fig = go.Figure(go.Bar(
        x=sub['Progress_Numeric'],
        y=sub['Intern_Name'],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.0f}%" for p in sub['Progress_Numeric']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Progress: %{x:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        **CHART_BASE,
        height=max(280, len(sub) * 32 + 60),
        margin=dict(t=10, b=10, l=160, r=60),
        xaxis=dict(title='Progress (%)', range=[0, 115], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def chart_course_assignment_status(fact_lms, intern_ids, dim_intern, course_name, dim_course):
    """Grouped bar: Reviewed vs Total Assignments per intern for one course."""
    cid_rows = dim_course[dim_course['Course Name'] == course_name]
    if cid_rows.empty:
        return go.Figure()
    cid = cid_rows['Course_ID'].values[0]

    sub = fact_lms[
        (fact_lms['Intern_ID'].isin(intern_ids)) &
        (fact_lms['Course_ID'] == cid)
    ].copy()
    sub = sub.merge(dim_intern, on='Intern_ID')
    sub['Pending'] = sub['Total_Assg'] - sub['Reviewed']

    names = sub['Intern_Name'].str.split().str[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Reviewed',  x=names, y=sub['Reviewed'],  marker_color='#207F8D'))
    fig.add_trace(go.Bar(name='Pending',   x=names, y=sub['Pending'],   marker_color='#F4A623'))
    fig.update_layout(
        **CHART_BASE,
        barmode='stack',
        height=280,
        margin=dict(t=10, b=20, l=40, r=20),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title='Assignments', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
        legend=dict(orientation='h', y=-0.3, xanchor='center', x=0.5),
    )
    return fig


def chart_course_kc_test(fact_lms, intern_ids, dim_intern, course_name, dim_course):
    """Grouped bar: KC % vs Test % per intern for one course."""
    cid_rows = dim_course[dim_course['Course Name'] == course_name]
    if cid_rows.empty:
        return go.Figure()
    cid = cid_rows['Course_ID'].values[0]

    sub = fact_lms[
        (fact_lms['Intern_ID'].isin(intern_ids)) &
        (fact_lms['Course_ID'] == cid)
    ].copy()
    sub = sub.merge(dim_intern, on='Intern_ID')

    # Compute KC and Test percentages
    sub['KC_pct']   = (sub['KC_scored']   / sub['KC_total'].replace(0, 1)   * 100).round(1)
    sub['Test_pct'] = (sub['Test_scored'] / sub['Test_total'].replace(0, 1) * 100).round(1)

    names = sub['Intern_Name'].str.split().str[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='KC Score %',   x=names, y=sub['KC_pct'],   marker_color='#207F8D'))
    fig.add_trace(go.Bar(name='Test Score %', x=names, y=sub['Test_pct'], marker_color='#F4A623'))
    fig.update_layout(
        **CHART_BASE,
        barmode='group',
        height=280,
        margin=dict(t=10, b=20, l=40, r=20),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title='Score %', range=[0, 110], showgrid=True,
                   gridcolor='rgba(0,0,0,0.07)'),
        legend=dict(orientation='h', y=-0.3, xanchor='center', x=0.5),
    )
    return fig


# ─────────────────────────────────────────────
#  NEW ── BY-INTERN VIEW CHARTS
# ─────────────────────────────────────────────

def chart_intern_activity_pie(fact_activity, intern_id, start_dt, end_dt):
    """
    Pie/Donut: total working hours split by activity for a single intern
    within the selected date range.
    """
    iact = fact_activity[
        (fact_activity['Intern_ID'] == intern_id) &
        (fact_activity['Date'] >= start_dt) &
        (fact_activity['Date'] <= end_dt)
    ]
    if iact.empty:
        return go.Figure(), 0.0

    agg = iact.groupby('Activity')['Hours'].sum().sort_values(ascending=False).reset_index()
    agg['Label'] = agg['Activity'].map(lambda a: f"{ACTIVITY_EMOJI.get(a, '•')} {a}")
    total_hrs = agg['Hours'].sum()

    fig = go.Figure(go.Pie(
        labels=agg['Label'],
        values=agg['Hours'],
        hole=0.58,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Safe,
            line=dict(color='white', width=2),
        ),
        hovertemplate='%{label}<br><b>%{value:.1f} hrs</b> (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        **CHART_BASE,
        height=380,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        annotations=[dict(
            text=f'<b>{total_hrs:.0f}</b><br>hrs',
            x=0.5, y=0.5, font_size=18,
            font=dict(color='#0f3f47', family='Nunito'),
            showarrow=False,
        )],
    )
    return fig, float(total_hrs)


def chart_intern_daily_hours(fact_activity, intern_id, start_dt, end_dt):
    """Bar chart of daily hours for the intern within the date window."""
    iact = fact_activity[
        (fact_activity['Intern_ID'] == intern_id) &
        (fact_activity['Date'] >= start_dt) &
        (fact_activity['Date'] <= end_dt)
    ].copy()
    if iact.empty:
        return go.Figure()

    daily = iact.groupby('Date')['Hours'].sum().reset_index()
    avg_h = daily['Hours'].mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily['Date'], y=daily['Hours'],
        marker_color='#207F8D',
        marker_opacity=0.8,
        hovertemplate='%{x|%d %b}: %{y:.1f} hrs<extra></extra>',
    ))
    fig.add_hline(y=avg_h, line_dash='dot', line_color='#E8593C', opacity=0.7,
                  annotation_text=f'Avg: {avg_h:.1f}h', annotation_position='right')
    fig.update_layout(
        **CHART_BASE,
        height=220,
        margin=dict(t=20, b=20, l=40, r=80),
        xaxis=dict(title='Date', tickformat='%d %b', tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title='Hours', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
    )
    return fig


def chart_intern_weekly_hours(fact_activity, intern_id, start_dt, end_dt):
    """Line + area: weekly hours trend for one intern."""
    iact = fact_activity[
        (fact_activity['Intern_ID'] == intern_id) &
        (fact_activity['Date'] >= start_dt) &
        (fact_activity['Date'] <= end_dt)
    ].copy()
    if iact.empty:
        return go.Figure()

    iact['week'] = iact['Date'].dt.isocalendar().week.astype(int)
    wk = iact.groupby('week')['Hours'].sum().reset_index()

    fig = go.Figure(go.Scatter(
        x=wk['week'], y=wk['Hours'],
        mode='lines+markers',
        line=dict(color='#207F8D', width=3),
        marker=dict(size=8, color='white', line=dict(color='#207F8D', width=2.5)),
        fill='tozeroy',
        fillcolor='rgba(32,127,141,0.12)',
        hovertemplate='Week %{x}: %{y:.0f} hrs<extra></extra>',
    ))
    fig.update_layout(
        **CHART_BASE,
        height=200,
        margin=dict(t=10, b=20, l=40, r=20),
        xaxis=dict(title='Week', tickmode='linear', dtick=1, tickfont=dict(size=10)),
        yaxis=dict(title='Hours', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
    )
    return fig


# ─────────────────────────────────────────────
#  SECTION: BY-COURSE PANEL
# ─────────────────────────────────────────────

def render_by_course_panel(
    fact_lms, fact_activity, dim_intern, dim_course,
    intern_ids, selected_course, mentor_act
):
    """
    Full dashboard view scoped to one course:
    progress bars, assignment table, KC/Test scores, status distribution.
    """
    short = COURSE_SHORT.get(selected_course, selected_course)

    # ── Find the Course_ID ────────────────────
    cid_rows = dim_course[dim_course['Course Name'] == selected_course]
    if cid_rows.empty:
        st.warning(f"No dimension entry found for course: {selected_course}")
        return
    cid = cid_rows['Course_ID'].values[0]

    sub = fact_lms[
        (fact_lms['Intern_ID'].isin(intern_ids)) &
        (fact_lms['Course_ID'] == cid)
    ].copy()

    if sub.empty:
        st.info(f"No data found for course **{selected_course}** under this mentor.")
        return

    sub = sub.merge(dim_intern, on='Intern_ID')
    sub['KC_pct']   = (sub['KC_scored']   / sub['KC_total'].replace(0, 1)   * 100).round(1)
    sub['Test_pct'] = (sub['Test_scored'] / sub['Test_total'].replace(0, 1) * 100).round(1)
    sub['Pending']  = sub['Total_Assg'] - sub['Reviewed']

    # ── KPIs ─────────────────────────────────
    n_interns   = len(sub)
    avg_prog    = sub['Progress_Numeric'].mean()
    completed   = (sub['Overall Status'] == 'Completed').sum()
    in_prog     = sub['Overall Status'].isin(['In Progress', 'In progress']).sum()
    not_started = sub['Overall Status'].isin(['Not started', 'Not Started']).sum()
    total_rev   = sub['Reviewed'].sum()
    total_assg  = sub['Total_Assg'].sum()
    avg_kc      = sub[sub['KC_pct'] > 0]['KC_pct'].mean() if (sub['KC_pct'] > 0).any() else 0
    avg_test    = sub[sub['Test_pct'] > 0]['Test_pct'].mean() if (sub['Test_pct'] > 0).any() else 0

    # Course header
    st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(32,127,141,0.18) 0%,
             rgba(202,221,224,0.45) 100%);border-radius:14px;padding:16px 22px;
             margin-bottom:14px;border-left:4px solid #207F8D'>
            <h3 style='margin:0;color:#0f3f47'>📚 {selected_course}</h3>
            <p style='margin:4px 0 0;color:#5a7d82;font-size:0.87rem'>
                {n_interns} interns enrolled &nbsp;·&nbsp;
                Avg Progress: <b style='color:#207F8D'>{avg_prog:.0f}%</b> &nbsp;·&nbsp;
                Assignments: <b>{total_rev}/{total_assg}</b> reviewed
            </p>
        </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("👥 Enrolled",      n_interns)
    k2.metric("📈 Avg Progress",  f"{avg_prog:.0f}%")
    k3.metric("✅ Completed",     completed)
    k4.metric("🔄 In Progress",   in_prog)
    k5.metric("🎯 Avg KC Score",  f"{avg_kc:.0f}%" if avg_kc else "N/A")
    k6.metric("📋 Avg Test Score", f"{avg_test:.0f}%" if avg_test else "N/A")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs for course-specific views ───────
    ct1, ct2, ct3, ct4 = st.tabs([
        "📊 Progress & Status",
        "📝 Assignments",
        "🎓 Knowledge & Test Scores",
        "📋 Full Table",
    ])

    with ct1:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### 📊 Intern Progress Bars")
            st.plotly_chart(
                chart_course_intern_progress(fact_lms, intern_ids, dim_intern, selected_course, dim_course),
                use_container_width=True
            )
        with c2:
            st.markdown("#### 🥧 Status Distribution")
            status_counts = sub['Overall Status'].value_counts()
            status_colors_map = {
                'Completed':   '#2ECC8E',
                'In Progress': '#F4A623',
                'In progress': '#F4A623',
                'Not started': '#E8593C',
                'Not Started': '#E8593C',
            }
            fig_pie = go.Figure(go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.55,
                marker=dict(
                    colors=[status_colors_map.get(s, '#888') for s in status_counts.index],
                    line=dict(color='white', width=2),
                ),
                textinfo='label+value',
                hovertemplate='%{label}: %{value} interns<extra></extra>',
            ))
            fig_pie.update_layout(
                **CHART_BASE,
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                annotations=[dict(
                    text=f'<b>{n_interns}</b><br>total',
                    x=0.5, y=0.5, font_size=15,
                    font=dict(color='#0f3f47', family='Nunito'),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### 📅 Activity Distribution")
            st.plotly_chart(
                chart_activity_breakdown(fact_activity, intern_ids),
                use_container_width=True
            )

    with ct2:
        st.markdown("#### 📝 Assignment Review Status")
        st.plotly_chart(
            chart_course_assignment_status(fact_lms, intern_ids, dim_intern, selected_course, dim_course),
            use_container_width=True
        )
        st.divider()
        st.markdown("#### 📋 Assignment Detail Table")
        assg_table = sub[['Intern_Name', 'Reviewed', 'Total_Assg', 'Pending', 'Overall Status']].copy()
        assg_table['Completion %'] = (assg_table['Reviewed'] / assg_table['Total_Assg'].replace(0, 1) * 100).round(0).astype(int)
        st.dataframe(
            assg_table.sort_values('Pending', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                'Intern_Name':   st.column_config.TextColumn('Intern'),
                'Reviewed':      st.column_config.NumberColumn('Reviewed'),
                'Total_Assg':    st.column_config.NumberColumn('Total'),
                'Pending':       st.column_config.ProgressColumn(
                    'Pending', min_value=0,
                    max_value=int(assg_table['Total_Assg'].max()),
                    format='%d'
                ),
                'Overall Status': st.column_config.TextColumn('Status'),
                'Completion %':  st.column_config.ProgressColumn(
                    'Completion %', min_value=0, max_value=100, format='%d%%'
                ),
            }
        )

    with ct3:
        st.markdown("#### 🎓 Knowledge Check & Test Scores")
        st.plotly_chart(
            chart_course_kc_test(fact_lms, intern_ids, dim_intern, selected_course, dim_course),
            use_container_width=True
        )
        st.divider()
        st.markdown("#### 📋 Score Summary")
        score_tbl = sub[['Intern_Name', 'KC_pct', 'Test_pct', 'Progress_Numeric', 'Overall Status']].copy()
        score_tbl.columns = ['Intern', 'KC Score %', 'Test Score %', 'Progress %', 'Status']
        score_tbl['KC Score %']   = score_tbl['KC Score %'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        score_tbl['Test Score %'] = score_tbl['Test Score %'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        score_tbl['Progress %']   = score_tbl['Progress %'].map(lambda x: f"{x:.0f}%")
        st.dataframe(score_tbl, use_container_width=True, hide_index=True)

    with ct4:
        st.markdown("#### 📋 Full Intern × Course Data")
        full_tbl = sub[[
            'Intern_Name', 'Progress_Numeric', 'Overall Status',
            'Reviewed', 'Total_Assg', 'Pending', 'KC_pct', 'Test_pct'
        ]].copy()
        full_tbl.columns = [
            'Intern', 'Progress %', 'Status',
            'Reviewed', 'Total Assg', 'Pending', 'KC %', 'Test %'
        ]
        full_tbl['Progress %'] = full_tbl['Progress %'].map(lambda x: f"{x:.0f}%")
        full_tbl['KC %']       = full_tbl['KC %'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        full_tbl['Test %']     = full_tbl['Test %'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        st.dataframe(full_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  SECTION: BY-INTERN PANEL
# ─────────────────────────────────────────────

def render_by_intern_panel(
    fact_lms, fact_activity, dim_intern, dim_course,
    intern_ids, selected_intern_name, start_dt, end_dt
):
    """
    Full dashboard view scoped to one intern over a custom date range.
    Shows: per-course progress, activity pie chart split by hours, daily/weekly bars.
    """
    intern_row = dim_intern[dim_intern['Intern_Name'] == selected_intern_name]
    if intern_row.empty:
        st.warning("Intern not found in dimension table.")
        return
    intern_id = intern_row['Intern_ID'].values[0]

    # ── Filter activity to date window ────────
    iact = fact_activity[
        (fact_activity['Intern_ID'] == intern_id) &
        (fact_activity['Date'] >= start_dt) &
        (fact_activity['Date'] <= end_dt)
    ].copy()

    # ── LMS data for this intern ──────────────
    ilms = fact_lms[fact_lms['Intern_ID'] == intern_id].copy()
    ilms = ilms.merge(dim_course, on='Course_ID')
    ilms['Course_Short'] = ilms['Course Name'].map(COURSE_SHORT).fillna(ilms['Course Name'])
    ilms['KC_pct']   = (ilms['KC_scored']   / ilms['KC_total'].replace(0, 1)   * 100).round(1)
    ilms['Test_pct'] = (ilms['Test_scored'] / ilms['Test_total'].replace(0, 1) * 100).round(1)

    total_hrs   = iact['Hours'].sum() if not iact.empty else 0
    avg_prog    = ilms['Progress_Numeric'].mean() if not ilms.empty else 0
    total_rev   = ilms['Reviewed'].sum()
    total_assg  = ilms['Total_Assg'].sum()
    completed   = (ilms['Overall Status'] == 'Completed').sum()
    date_days   = (end_dt - start_dt).days + 1

    # ── Intern Header ─────────────────────────
    initials = ''.join([w[0].upper() for w in selected_intern_name.split()][:2])
    st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(32,127,141,0.18) 0%,
             rgba(202,221,224,0.45) 100%);border-radius:14px;padding:16px 22px;
             margin-bottom:14px;border-left:4px solid #207F8D;
             display:flex;align-items:center;gap:16px'>
            <div style='width:52px;height:52px;border-radius:50%;
                 background:linear-gradient(135deg,#207F8D,#0f3f47);
                 display:flex;align-items:center;justify-content:center;
                 font-size:1.2rem;font-weight:800;color:#fff;flex-shrink:0'>
                {initials}
            </div>
            <div>
                <h3 style='margin:0;color:#0f3f47'>{selected_intern_name}</h3>
                <p style='margin:4px 0 0;color:#5a7d82;font-size:0.87rem'>
                    {start_dt.strftime('%d %b %Y')} → {end_dt.strftime('%d %b %Y')}
                    &nbsp;·&nbsp; <b>{date_days}</b> days
                    &nbsp;·&nbsp; <b style='color:#207F8D'>{total_hrs:.0f} hrs</b> logged
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("⏱️ Total Hours",    f"{total_hrs:.0f} hrs")
    k2.metric("📈 Avg Progress",   f"{avg_prog:.0f}%")
    k3.metric("✅ Completed",      f"{completed}/{len(ilms)} courses")
    k4.metric("📝 Assignments",    f"{total_rev}/{total_assg}")
    k5.metric("🎯 Avg KC Score",
              f"{ilms[ilms['KC_pct']>0]['KC_pct'].mean():.0f}%" if (ilms['KC_pct'] > 0).any() else "N/A")
    k6.metric("📋 Avg Test Score",
              f"{ilms[ilms['Test_pct']>0]['Test_pct'].mean():.0f}%" if (ilms['Test_pct'] > 0).any() else "N/A")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main two-column layout ────────────────
    pie_col, right_col = st.columns([2, 3])

    with pie_col:
        st.markdown(f"#### 🥧 Time Distribution ({start_dt.strftime('%d %b')} – {end_dt.strftime('%d %b %Y')})")
        if iact.empty:
            st.info("No activity logs in the selected date range.")
        else:
            fig_pie, total_h = chart_intern_activity_pie(fact_activity, intern_id, start_dt, end_dt)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Activity breakdown table below pie
            agg_act = iact.groupby('Activity')['Hours'].sum().sort_values(ascending=False).reset_index()
            agg_act['%'] = (agg_act['Hours'] / agg_act['Hours'].sum() * 100).round(1)
            agg_act['Hours'] = agg_act['Hours'].map(lambda h: f"{h:.1f} hrs")
            agg_act['%']     = agg_act['%'].map(lambda p: f"{p:.1f}%")
            agg_act['Activity'] = agg_act['Activity'].map(
                lambda a: f"{ACTIVITY_EMOJI.get(a, '•')} {a}"
            )
            st.dataframe(
                agg_act.rename(columns={'Activity': 'Activity', 'Hours': 'Hours', '%': 'Share'}),
                use_container_width=True,
                hide_index=True,
            )

    with right_col:
        st.markdown("#### 📊 Course-wise Progress")
        for _, row in ilms.iterrows():
            bg, fg, icon = STATUS_CFG.get(row['Overall Status'], ('#F3F4F6', '#374151', '❓'))
            bar_color = '#2ECC8E' if row['Progress_Numeric'] >= 70 else (
                '#F4A623' if row['Progress_Numeric'] >= 40 else '#E8593C'
            )
            kc_str   = f"KC: {row['KC_pct']:.0f}%"   if row['KC_pct']   > 0 else "KC: —"
            test_str = f"Test: {row['Test_pct']:.0f}%" if row['Test_pct'] > 0 else "Test: —"
            st.markdown(f"""
                <div style='background:rgba(255,255,255,0.55);border-radius:12px;
                     padding:12px 16px;margin-bottom:10px;
                     border-left:4px solid {bar_color}'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <b style='color:#0f3f47;font-size:0.92rem'>{row['Course_Short']}</b>
                        <span style='background:{bg};color:{fg};padding:2px 10px;
                              border-radius:20px;font-size:0.75rem;font-weight:700'>
                            {icon} {row['Overall Status']}
                        </span>
                    </div>
                    <div style='margin-top:8px;background:rgba(32,127,141,0.1);
                         border-radius:20px;height:8px;overflow:hidden'>
                        <div style='width:{row["Progress_Numeric"]:.0f}%;height:100%;
                             background:{bar_color};border-radius:20px'></div>
                    </div>
                    <div style='margin-top:6px;font-size:0.82rem;color:#5a7d82'>
                        Progress: <b style='color:{bar_color}'>{row["Progress_Numeric"]:.0f}%</b>
                        &nbsp;·&nbsp; {row["Reviewed"]}/{row["Total_Assg"]} assignments
                        &nbsp;·&nbsp; {kc_str} &nbsp;·&nbsp; {test_str}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Daily hours chart
        if not iact.empty:
            st.markdown("#### 📅 Daily Hours")
            st.plotly_chart(
                chart_intern_daily_hours(fact_activity, intern_id, start_dt, end_dt),
                use_container_width=True
            )
            st.markdown("#### 📈 Weekly Hours Trend")
            st.plotly_chart(
                chart_intern_weekly_hours(fact_activity, intern_id, start_dt, end_dt),
                use_container_width=True
            )


# ─────────────────────────────────────────────
#  SECTION: INTERN DEEP DIVE (unchanged)
# ─────────────────────────────────────────────

def render_intern_deepdive(intern_name, intern_id, fact_lms, fact_activity, dim_course):
    ilms  = fact_lms[fact_lms['Intern_ID'] == intern_id].copy()
    iact  = fact_activity[fact_activity['Intern_ID'] == intern_id].copy()

    if ilms.empty:
        st.warning("No LMS data for this intern.")
        return

    ilms = ilms.merge(dim_course, on='Course_ID')
    ilms['Course_Short'] = ilms['Course Name'].map(COURSE_SHORT).fillna(ilms['Course Name'])
    if 'KC_pct' not in ilms.columns:
        if 'KC_scored' in ilms.columns and 'KC_total' in ilms.columns:
            ilms['KC_pct'] = (ilms['KC_scored'] / ilms['KC_total'].replace(0, 1) * 100).round(1)
        elif 'Overall Knowledge Check' in ilms.columns:
            kc = ilms['Overall Knowledge Check'].astype(str).str.split('/', expand=True)
            ks = pd.to_numeric(kc[0].str.strip(), errors='coerce').fillna(0)
            kt = pd.to_numeric(kc[1].str.strip(), errors='coerce').fillna(1)
            ilms['KC_pct'] = (ks / kt.replace(0, 1) * 100).round(1)
        else:
            ilms['KC_pct'] = 0.0
    if 'Test_pct' not in ilms.columns:
        if 'Test_scored' in ilms.columns and 'Test_total' in ilms.columns:
            ilms['Test_pct'] = (ilms['Test_scored'] / ilms['Test_total'].replace(0, 1) * 100).round(1)
        elif 'Overall Test' in ilms.columns:
            ts = ilms['Overall Test'].astype(str).str.split('/', expand=True)
            tsc = pd.to_numeric(ts[0].str.strip(), errors='coerce').fillna(0)
            ttt = pd.to_numeric(ts[1].str.strip(), errors='coerce').fillna(1)
            ilms['Test_pct'] = (tsc / ttt.replace(0, 1) * 100).round(1)
        else:
            ilms['Test_pct'] = 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Progress",  f"{ilms['Progress_Numeric'].mean():.0f}%")
    k2.metric("Total Hours",   f"{iact['Hours'].sum():.0f} hrs" if not iact.empty else "N/A")
    k3.metric("Assignments",   f"{int(ilms['Reviewed'].sum())}/{int(ilms['Total_Assg'].sum())}")
    k4.metric("Completed",     f"{(ilms['Overall Status']=='Completed').sum()}/{len(ilms)} courses")
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Course-wise Performance**")
        for _, row in ilms.iterrows():
            bg, fg, icon = STATUS_CFG.get(row['Overall Status'], ('#F3F4F6', '#374151', '❓'))
            kc_str   = f"KC: {row['KC_pct']:.0f}%"   if row['KC_pct']   > 0 else "KC: —"
            test_str = f"Test: {row['Test_pct']:.0f}%" if row['Test_pct'] > 0 else "Test: —"
            st.markdown(f"""
                <div style='background:rgba(255,255,255,0.5);border-radius:10px;
                     padding:10px 14px;margin-bottom:8px;border-left:4px solid #207F8D'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <b style='color:#0f3f47;font-size:0.9rem'>{row['Course_Short']}</b>
                        <span style='background:{bg};color:{fg};padding:2px 10px;
                              border-radius:20px;font-size:0.75rem;font-weight:700'>
                            {icon} {row['Overall Status']}
                        </span>
                    </div>
                    <div style='margin-top:6px;font-size:0.82rem;color:#5a7d82'>
                        Progress: <b>{row['Progress_Numeric']:.0f}%</b> &nbsp;·&nbsp;
                        {row['Reviewed']}/{row['Total_Assg']} assignments &nbsp;·&nbsp;
                        {kc_str} &nbsp;·&nbsp; {test_str}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with right:
        if not iact.empty:
            st.markdown("**Activity Distribution (All Time)**")
            agg = iact.groupby('Activity')['Hours'].sum().sort_values(ascending=False).reset_index()
            agg['Label'] = agg['Activity'].map(lambda a: f"{ACTIVITY_EMOJI.get(a, '•')} {a}")
            fig = go.Figure(go.Bar(
                x=agg['Hours'], y=agg['Label'], orientation='h',
                marker_color='#207F8D',
                text=[f"{h:.0f}h" for h in agg['Hours']],
                textposition='outside',
            ))
            fig.update_layout(
                **CHART_BASE,
                height=280,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(showgrid=False, title=''),
                yaxis=dict(tickfont=dict(size=10), autorange='reversed'),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Weekly Hours Trend**")
            iact2 = iact.copy()
            iact2['week'] = iact2['Date'].dt.isocalendar().week.astype(int)
            wk = iact2.groupby('week')['Hours'].sum().reset_index()
            fig2 = go.Figure(go.Scatter(
                x=wk['week'], y=wk['Hours'],
                mode='lines+markers',
                line=dict(color='#207F8D', width=2),
                marker=dict(size=7),
                fill='tozeroy', fillcolor='rgba(32,127,141,0.1)',
            ))
            fig2.update_layout(
                **CHART_BASE, height=140,
                margin=dict(t=5, b=20, l=40, r=10),
                xaxis=dict(tickmode='linear', dtick=1, title='Week', tickfont=dict(size=9)),
                yaxis=dict(title='Hrs', tickfont=dict(size=9)),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No activity logs for this intern.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ✨ AI-Powered Intervention Strategy")
    st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                "Generate a custom mentorship action plan based on this intern's current metrics and risk profile.</p>",
                unsafe_allow_html=True)
                
    # We will simulate risk level calculation for this profile view directly for the prompt
    avg_p = ilms['Progress_Numeric'].mean()
    hrs = iact['Hours'].sum() if not iact.empty else 0
    risk_level = "Medium"
    if avg_p < 40: risk_level = "High"
    elif avg_p > 70: risk_level = "Low"
    
    if st.button("Generate Intervention Plan", key=f"gen_plan_{intern_id}"):
        with st.spinner("Analyzing profile & generating strategy..."):
            metrics = {'Avg Progress': f"{avg_p:.0f}%", 'Hours Logged': f"{hrs:.0f} hrs"}
            plan = genai_service.generate_intervention_strategy(intern_name, risk_level, metrics)
            st.success("Plan Generated Successfully!")
            st.markdown(f"<div style='background-color:#F8FAFC;padding:16px;border-radius:8px;border:1px solid #E2E8F0'>{plan}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION: ACTION ITEMS
# ─────────────────────────────────────────────

def render_action_items(risk_df, pending_df):
    high_risk = risk_df[risk_df['Risk_Level'] == 'High']
    med_risk  = risk_df[risk_df['Risk_Level'] == 'Medium']
    total_pending = pending_df['Pending'].sum() if not pending_df.empty else 0

    cards = []
    if len(high_risk) > 0:
        names = ', '.join(high_risk['Intern_Name'].str.split().str[0].tolist())
        cards.append(('🚨 Urgent: At-Risk Interns',
                       f"{len(high_risk)} intern(s) need immediate check-in: **{names}**",
                       '#FEE2E2', '#991B1B'))
    if total_pending > 5:
        cards.append(('📝 Review Backlog',
                       f"**{int(total_pending)} assignments** are awaiting your review across {len(pending_df)} intern-course combinations.",
                       '#FEF3C7', '#92400E'))
    if len(med_risk) > 2:
        cards.append(('⚡ Medium Risk Watch',
                       f"{len(med_risk)} interns are trending downward in effort. Consider a group check-in.",
                       '#FFF7ED', '#9A3412'))
    if not cards:
        cards.append(('✅ All Clear!',
                       'No urgent action items. Your cohort is performing well!',
                       '#D1FAE5', '#065F46'))

    cols = st.columns(len(cards))
    for col, (title, body, bg, fg) in zip(cols, cards):
        col.markdown(f"""
            <div style='background:{bg};border-radius:14px;padding:16px 18px;height:110px;
                 border-left:4px solid {fg}'>
                <p style='margin:0;font-weight:800;font-size:0.88rem;color:{fg}'>{title}</p>
                <p style='margin:6px 0 0;font-size:0.82rem;color:{fg};opacity:0.9'>{body}</p>
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def show_mentor_dashboard(fact_lms, fact_activity, dim_intern, dim_course, dim_mentor):

    # ── Ensure KC / Test score columns exist ─
    fact_lms = _ensure_score_cols(fact_lms)

    # ── Sidebar ───────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧑‍🏫 Mentor View")

    mentor_name = st.sidebar.selectbox(
        "Select Mentor", sorted(dim_mentor['Mentor_Name'].tolist()), key="mentor_sel"
    )

    # ── View Mode selector ───────────────────
    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio(
        "📌 View Mode",
        ["🏠 Cohort Overview", "📚 By Course", "👤 By Intern"],
        key="mentor_view_mode",
    )

    # ── Date range (shown in all modes) ──────
    st.sidebar.markdown("---")
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(fact_activity['Date'].min().date(), fact_activity['Date'].max().date()),
        key="mentor_date"
    )
    start_dt = pd.Timestamp(date_range[0]) if len(date_range) > 0 else fact_activity['Date'].min()
    end_dt   = pd.Timestamp(date_range[1]) if len(date_range) > 1 else fact_activity['Date'].max()

    # ── Course selector (shown in By Course mode) ──
    selected_course = None
    if view_mode == "📚 By Course":
        st.sidebar.markdown("---")
        mentor_courses = get_mentor_courses(mentor_name, fact_lms, dim_course)
        if mentor_courses:
            selected_course = st.sidebar.selectbox(
                "📖 Select Course",
                mentor_courses,
                key="mentor_course_sel"
            )
        else:
            st.sidebar.warning("No courses found for this mentor.")

    # ── Intern selector (shown in By Intern mode) ──
    selected_intern = None
    if view_mode == "👤 By Intern":
        st.sidebar.markdown("---")
        intern_ids_all = get_mentor_intern_ids(mentor_name, fact_lms)
        intern_names_all = (
            dim_intern[dim_intern['Intern_ID'].isin(intern_ids_all)]['Intern_Name']
            .sort_values()
            .tolist()
        )
        if intern_names_all:
            selected_intern = st.sidebar.selectbox(
                "👤 Select Intern",
                intern_names_all,
                key="mentor_intern_sel"
            )
        else:
            st.sidebar.warning("No interns found for this mentor.")

    # ── Filter activity by date ───────────────
    fact_activity_f = fact_activity[
        (fact_activity['Date'] >= start_dt) & (fact_activity['Date'] <= end_dt)
    ].copy()

    # ── Core derivations ─────────────────────
    intern_ids   = get_mentor_intern_ids(mentor_name, fact_lms)
    n_interns    = len(intern_ids)
    risk_df      = compute_dropout_risk(intern_ids, fact_lms, fact_activity_f, dim_intern)
    pending_df   = pending_reviews_df(fact_lms, intern_ids, dim_intern, dim_course)
    score_df     = score_summary(fact_lms, intern_ids, dim_course)
    mentor_lms   = fact_lms[fact_lms['Intern_ID'].isin(intern_ids)]
    mentor_act   = fact_activity_f[fact_activity_f['Intern_ID'].isin(intern_ids)]

    completed_ct  = (mentor_lms['Overall Status'] == 'Completed').sum()
    inprog_ct     = (mentor_lms['Overall Status'] == 'In Progress').sum()
    notstart_ct   = (mentor_lms['Overall Status'].isin(['Not started', 'Not Started'])).sum()
    avg_progress  = mentor_lms['Progress_Numeric'].mean() if not mentor_lms.empty else 0
    total_hrs     = mentor_act['Hours'].sum()
    high_risk_ct  = (risk_df['Risk_Level'] == 'High').sum()
    total_pending = int(pending_df['Pending'].sum()) if not pending_df.empty else 0

    # ══════════════════════════════════════════
    #  ROUTE BY VIEW MODE
    # ══════════════════════════════════════════

    # ── BY COURSE VIEW ────────────────────────
    if view_mode == "📚 By Course":
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(32,127,141,0.18) 0%,
                 rgba(202,221,224,0.45) 100%);border-radius:16px;padding:18px 28px;
                 margin-bottom:16px;border-left:4px solid #207F8D'>
                <h1 style='margin:0;font-size:1.7rem;color:#0f3f47'>
                    📚 <span style='color:#207F8D'>{mentor_name}</span> — Course View
                </h1>
                <p style='margin:6px 0 0;color:#5a7d82;font-size:0.9rem'>
                    Viewing: <b>{selected_course or '—'}</b>
                    &nbsp;·&nbsp; {n_interns} interns in cohort
                    &nbsp;·&nbsp; Cohort avg: <b style='color:#207F8D'>{avg_progress:.0f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

        if selected_course:
            render_by_course_panel(
                fact_lms, fact_activity_f, dim_intern, dim_course,
                intern_ids, selected_course, mentor_act
            )
        return

    # ── BY INTERN VIEW ────────────────────────
    if view_mode == "👤 By Intern":
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(32,127,141,0.18) 0%,
                 rgba(202,221,224,0.45) 100%);border-radius:16px;padding:18px 28px;
                 margin-bottom:16px;border-left:4px solid #207F8D'>
                <h1 style='margin:0;font-size:1.7rem;color:#0f3f47'>
                    👤 <span style='color:#207F8D'>{mentor_name}</span> — Intern View
                </h1>
                <p style='margin:6px 0 0;color:#5a7d82;font-size:0.9rem'>
                    Intern: <b>{selected_intern or '—'}</b>
                    &nbsp;·&nbsp; {start_dt.strftime('%d %b %Y')} → {end_dt.strftime('%d %b %Y')}
                </p>
            </div>
        """, unsafe_allow_html=True)

        if selected_intern:
            render_by_intern_panel(
                fact_lms, fact_activity_f, dim_intern, dim_course,
                intern_ids, selected_intern, start_dt, end_dt
            )
        return

    # ── COHORT OVERVIEW (default) ─────────────
    st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(32,127,141,0.18) 0%,
             rgba(202,221,224,0.45) 100%);border-radius:16px;padding:20px 28px;
             margin-bottom:16px;border-left:4px solid #207F8D'>
            <h1 style='margin:0;font-size:1.85rem;color:#0f3f47'>
                🧑‍🏫 <span style='color:#207F8D'>{mentor_name}</span> — Mentor Command Centre
            </h1>
            <p style='margin:6px 0 0;color:#5a7d82;font-size:0.92rem'>
                Managing <b>{n_interns} interns</b> &nbsp;·&nbsp;
                <b style='color:#E8593C'>{high_risk_ct} at-risk</b> &nbsp;·&nbsp;
                <b style='color:#F4A623'>{total_pending} pending reviews</b> &nbsp;·&nbsp;
                Cohort avg progress: <b style='color:#207F8D'>{avg_progress:.0f}%</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    render_action_items(risk_df, pending_df)
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("👥 Interns",        n_interns)
    k2.metric("📈 Avg Progress",   f"{avg_progress:.0f}%")
    k3.metric("✅ Completed",       completed_ct)
    k4.metric("🔄 In Progress",     inprog_ct)
    k5.metric("⏳ Not Started",     notstart_ct)
    k6.metric("⏱️ Total Hours",    f"{total_hrs:.0f}")
    st.markdown("<br>", unsafe_allow_html=True)

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Ask Data (GenAI)",
        "📋 Cohort Overview",
        "🚨 Risk Intelligence",
        "📊 Performance Deep-Dive",
        "⏱️ Effort Analytics",
        "👤 Intern Profiles",
        "🤖 AI/ML Insights",
    ])

    # ══════════════════════════════════════════
    #  TAB 0 — ASK DATA (GenAI)
    # ══════════════════════════════════════════
    with tab0:
        st.markdown("#### 💬 Ask Data (NLP Query)")
        st.markdown("<p style='font-size:0.85rem;color:#5a7d82;margin-top:-8px'>"
                    "Ask questions about your cohort in plain english (e.g. 'Who are the worst performing interns?', 'What is the average progress?').</p>", 
                    unsafe_allow_html=True)
        
        # Initialize chat history specific to mentor
        chat_key_m = f"chat_history_mentor_{mentor_name}"
        if chat_key_m not in st.session_state:
            st.session_state[chat_key_m] = [
                {"role": "assistant", "content": f"Hello {mentor_name}. I have loaded the data for your {n_interns} interns. What would you like to know?"}
            ]

        # Display chat messages
        chat_container_m = st.container(height=350)
        with chat_container_m:
            for message in st.session_state[chat_key_m]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about your data...", key="mentor_data_chat"):
            st.session_state[chat_key_m].append({"role": "user", "content": prompt})
            with chat_container_m:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container_m:
                with st.chat_message("assistant"):
                    with st.spinner("Querying data..."):
                        # In full implementation, pass fact_lms/fact_activity directly to pandasAI here
                        response = genai_service.query_dataframe(mentor_lms.merge(dim_intern, on='Intern_ID'), prompt)
                        st.markdown(response)
            
            st.session_state[chat_key_m].append({"role": "assistant", "content": response})

    # ══════════════════════════════════════════
    #  TAB 1 — COHORT OVERVIEW
    # ══════════════════════════════════════════
    with tab1:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### 🗺️ Progress Heatmap — All Interns × All Courses")
            st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                        "Green = on track · Yellow = lagging · Red = not started</p>",
                        unsafe_allow_html=True)
            with st.container(height=500):
                st.plotly_chart(
                    chart_cohort_progress_heatmap(fact_lms, intern_ids, dim_intern, dim_course),
                    use_container_width=True
                )
        with right:
            st.markdown("#### 🥧 Cohort Activity Distribution")
            st.plotly_chart(chart_activity_breakdown(fact_activity_f, intern_ids),
                            use_container_width=True)
            st.markdown("#### 📅 Weekly Cohort Effort Trend")
            st.plotly_chart(chart_cohort_weekly_line(fact_activity_f, intern_ids),
                            use_container_width=True)

        st.divider()
        st.markdown("#### 📊 Status Distribution by Course")
        sub = mentor_lms.merge(dim_course, on='Course_ID')
        sub['Course_Short'] = sub['Course Name'].map(COURSE_SHORT).fillna(sub['Course Name'])
        status_pivot = sub.groupby(['Course_Short', 'Overall Status']).size().unstack(fill_value=0)
        status_colors = {
            'Completed':   '#207F8D',
            'In Progress': '#F4A623',
            'In progress': '#F4A623',
            'Not started': '#E8593C',
            'Not Started': '#E8593C',
        }
        fig_status = go.Figure()
        for status in status_pivot.columns:
            fig_status.add_trace(go.Bar(
                name=status,
                x=status_pivot.index,
                y=status_pivot[status],
                marker_color=status_colors.get(status, '#888'),
                text=status_pivot[status],
                textposition='inside',
                hovertemplate=f'{status}: %{{y}}<extra></extra>',
            ))
        fig_status.update_layout(
            **CHART_BASE,
            barmode='stack', height=260,
            margin=dict(t=10, b=20, l=40, r=20),
            xaxis=dict(tickfont=dict(size=12)),
            yaxis=dict(title='# Interns', showgrid=True, gridcolor='rgba(0,0,0,0.07)'),
            legend=dict(orientation='h', y=-0.3, xanchor='center', x=0.5),
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # ══════════════════════════════════════════
    #  TAB 2 — RISK INTELLIGENCE
    # ══════════════════════════════════════════
    with tab2:
        st.markdown("#### 🎯 Dropout Risk Radar")
        st.markdown(
            "<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
            "Risk score = progress deficit (40%) + effort decline (30%) + "
            "assignment lag (20%) + inactivity (10%). "
            "Bubble size = assignment lag severity.</p>",
            unsafe_allow_html=True
        )
        scatter_col, bar_col = st.columns([3, 2])
        with scatter_col:
            st.plotly_chart(chart_risk_scatter(risk_df), use_container_width=True)
        with bar_col:
            st.markdown("#### 🔬 Risk Breakdown")
            st.plotly_chart(chart_risk_breakdown_bar(risk_df), use_container_width=True)

        st.divider()
        st.markdown("#### 📋 Full Risk Register")
        display_risk = risk_df.copy()
        display_risk['Risk'] = display_risk['Risk_Level'].map(
            lambda l: f"{RISK_COLOR[l][2]} {l}"
        )
        display_risk['Avg Progress'] = display_risk['Avg_Progress'].map(lambda x: f"{x:.0f}%")
        display_risk['Risk Score']   = display_risk['Risk_Score'].map(lambda x: f"{x:.1f}/100")
        st.dataframe(
            display_risk[['Intern_Name', 'Risk', 'Risk Score', 'Avg Progress']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Intern_Name': st.column_config.TextColumn('Intern'),
                'Risk':        st.column_config.TextColumn('Risk Level'),
                'Risk Score':  st.column_config.TextColumn('Score'),
                'Avg Progress':st.column_config.TextColumn('Avg Progress'),
            }
        )
        st.divider()
        st.markdown("#### 📝 Assignment Review Queue")
        if pending_df.empty:
            st.success("✅ All assignments are reviewed!")
        else:
            st.dataframe(
                pending_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Intern_Name':  st.column_config.TextColumn('Intern'),
                    'Course_Short': st.column_config.TextColumn('Course'),
                    'Reviewed':     st.column_config.NumberColumn('Reviewed'),
                    'Total_Assg':   st.column_config.NumberColumn('Total'),
                    'Pending':      st.column_config.ProgressColumn(
                        'Pending', min_value=0,
                        max_value=int(pending_df['Total_Assg'].max()),
                        format='%d'
                    ),
                }
            )

    # ══════════════════════════════════════════
    #  TAB 3 — PERFORMANCE DEEP-DIVE
    # ══════════════════════════════════════════
    with tab3:
        st.markdown("#### 📊 Individual Progress by Course")
        st.plotly_chart(
            chart_individual_progress_bars(fact_lms, intern_ids, dim_intern, dim_course),
            use_container_width=True
        )
        st.divider()

        score_left, score_right = st.columns([2, 3])
        with score_left:
            st.markdown("#### 🎓 Score Distribution")
            st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                        "Box plots show spread of KC % and Test % scores across all interns.</p>",
                        unsafe_allow_html=True)
            st.plotly_chart(chart_score_distribution(score_df), use_container_width=True)

        with score_right:
            st.markdown("#### 🔗 Effort vs Progress Correlation")
            st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                        "Each dot = one intern. Colour = progress %. Trendline shows overall correlation.</p>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                chart_effort_vs_progress(fact_lms, fact_activity_f, intern_ids, dim_intern),
                use_container_width=True
            )
        st.divider()

        st.markdown("#### 📋 Full Score Summary Table")
        score_table = score_df.merge(dim_intern, on='Intern_ID')
        score_table['Course_Short'] = score_table['Course Name'].map(COURSE_SHORT).fillna(score_table['Course Name'])
        score_table['KC_pct']   = score_table['KC_pct'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        score_table['Test_pct'] = score_table['Test_pct'].map(lambda x: f"{x:.0f}%" if x > 0 else '—')
        score_table['Progress'] = score_table['Progress_Numeric'].map(lambda x: f"{x:.0f}%")
        st.dataframe(
            score_table[['Intern_Name', 'Course_Short', 'Progress', 'Overall Status', 'KC_pct', 'Test_pct']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Intern_Name':    st.column_config.TextColumn('Intern'),
                'Course_Short':   st.column_config.TextColumn('Course'),
                'Progress':       st.column_config.TextColumn('Progress'),
                'Overall Status': st.column_config.TextColumn('Status'),
                'KC_pct':         st.column_config.TextColumn('KC Score'),
                'Test_pct':       st.column_config.TextColumn('Test Score'),
            }
        )

    # ══════════════════════════════════════════
    #  TAB 4 — EFFORT ANALYTICS
    # ══════════════════════════════════════════
    with tab4:
        st.markdown("#### 📈 Weekly Effort per Intern (Stacked)")
        st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                    "Stack shows each intern's contribution. Dotted line = total cohort hours (right axis).</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_weekly_effort(fact_activity_f, intern_ids, dim_intern),
                        use_container_width=True)
        st.divider()

        st.markdown("#### 🏆 Effort Leaderboard")
        hrs_by_intern = mentor_act.groupby('Intern_ID')['Hours'].sum().reset_index()
        hrs_by_intern = hrs_by_intern.merge(dim_intern, on='Intern_ID').sort_values('Hours', ascending=False)
        hrs_by_intern['Rank'] = [f"#{i+1}" for i in range(len(hrs_by_intern))]
        hrs_by_intern['Hours_disp'] = hrs_by_intern['Hours'].map(lambda x: f"{x:.0f} hrs")
        max_hrs = hrs_by_intern['Hours'].max() if not hrs_by_intern.empty else 1
        st.dataframe(
            hrs_by_intern[['Rank', 'Intern_Name', 'Hours_disp', 'Hours']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'Rank':        st.column_config.TextColumn('Rank', width='small'),
                'Intern_Name': st.column_config.TextColumn('Intern'),
                'Hours_disp':  st.column_config.TextColumn('Total Hours'),
                'Hours':       st.column_config.ProgressColumn(
                    'Effort Bar', min_value=0, max_value=float(max_hrs), format='%.0f'
                ),
            }
        )
        st.divider()

        st.markdown("#### 🗓️ Intern Activity Calendar (Weekly Hours)")
        iact_cal = mentor_act.copy()
        if not iact_cal.empty:
            iact_cal['week'] = iact_cal['Date'].dt.isocalendar().week.astype(int)
            iact_cal = iact_cal.merge(dim_intern, on='Intern_ID')
            cal_pivot = iact_cal.groupby(['Intern_Name', 'week'])['Hours'].sum().unstack(fill_value=0)
            fig_cal = go.Figure(go.Heatmap(
                z=cal_pivot.values,
                x=[f"W{w}" for w in cal_pivot.columns],
                y=[n.split()[0] for n in cal_pivot.index],
                colorscale=[[0, '#F1EFE8'], [0.3, '#9FE1CB'], [0.7, '#207F8D'], [1, '#0F3F47']],
                colorbar=dict(title='Hrs', thickness=12, len=0.7),
                hovertemplate='%{y}<br>%{x}: %{z:.1f} hrs<extra></extra>',
                xgap=3, ygap=3,
            ))
            fig_cal.update_layout(
                **CHART_BASE,
                height=max(280, len(cal_pivot) * 24 + 80),
                margin=dict(t=30, b=10, l=100, r=60),
                xaxis=dict(side='top', tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
            )
            st.plotly_chart(fig_cal, use_container_width=True)

    # ══════════════════════════════════════════
    #  TAB 5 — INTERN PROFILES
    # ══════════════════════════════════════════
    with tab5:
        st.markdown("#### 👤 Individual Intern Deep-Dive")
        st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                    "Click any intern to expand their full performance profile.</p>",
                    unsafe_allow_html=True)

        risk_order       = risk_df.set_index('Intern_ID')['Risk_Level'].to_dict()
        risk_score_order = risk_df.set_index('Intern_ID')['Risk_Score'].to_dict()

        intern_names_sorted = dim_intern[dim_intern['Intern_ID'].isin(intern_ids)].copy()
        intern_names_sorted['Risk_Score'] = intern_names_sorted['Intern_ID'].map(
            lambda x: risk_score_order.get(x, 0)
        )
        intern_names_sorted = intern_names_sorted.sort_values('Risk_Score', ascending=False)

        for _, row in intern_names_sorted.iterrows():
            iid   = row['Intern_ID']
            iname = row['Intern_Name']
            level = risk_order.get(iid, 'Low')
            score = risk_score_order.get(iid, 0)
            bg, fg, icon = RISK_COLOR.get(level, ('#F1F5F9', '#374151', '•'))

            avg_p     = fact_lms[fact_lms['Intern_ID'] == iid]['Progress_Numeric'].mean()
            avg_p_str = f"{avg_p:.0f}%" if not np.isnan(avg_p) else "—"
            hrs_str   = f"{fact_activity_f[fact_activity_f['Intern_ID']==iid]['Hours'].sum():.0f} hrs"

            with st.expander(
                f"{icon} {iname}  —  Risk: {level} ({score:.0f})  ·  "
                f"Avg Progress: {avg_p_str}  ·  Hours: {hrs_str}",
                expanded=(level == 'High')
            ):
                render_intern_deepdive(iname, iid, fact_lms, fact_activity_f, dim_course)

    # ══════════════════════════════════════════
    #  TAB 6 — AI/ML INSIGHTS
    # ══════════════════════════════════════════
    with tab6:
        st.markdown("#### 🤖 Machine Learning Insights")
        st.markdown("<p style='font-size:0.83rem;color:#5a7d82;margin-top:-8px'>"
                    "Advanced analytics powered by K-Means Clustering and Random Forest Models.</p>",
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 👥 Intern Clusters (Learner Personas)")
            cluster_df = ml_models.get_intern_clusters(fact_lms, fact_activity_f)
            if not cluster_df.empty:
                cluster_df = cluster_df.merge(dim_intern, on='Intern_ID')
                fig_cluster = px.scatter(
                    cluster_df, x="Hours", y="Progress_Numeric", color="Persona",
                    hover_name="Intern_Name", size_max=15, 
                    title="K-Means Clusters (Hours vs Progress)",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_cluster.update_layout(**CHART_BASE, height=350, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.info("Not enough data to form clusters.")
                
        with col2:
            st.markdown("##### 📈 Test Score Prediction (Random Forest)")
            test_df = ml_models.predict_test_scores(fact_lms, fact_activity_f)
            if not test_df.empty:
                test_df = test_df.merge(dim_intern, on='Intern_ID')
                fig_test = px.scatter(
                    test_df, x="Test_pct", y="Predicted_Test_pct",
                    hover_name="Intern_Name", color="Score_Diff",
                    title="Actual vs Predicted Final Test Scores",
                    color_continuous_scale="RdYlGn_r"
                )
                fig_test.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect Prediction', line=dict(color='gray', dash='dash')))
                fig_test.update_layout(**CHART_BASE, height=350, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_test, use_container_width=True)
            else:
                st.info("Not enough data to predict scores.")

        st.divider()
        st.markdown("##### 🚨 Dropout Risk Classification (Random Forest)")
        risk_clf_df = ml_models.predict_dropout_risk(fact_lms, fact_activity_f)
        if not risk_clf_df.empty:
            risk_clf_df = risk_clf_df.merge(dim_intern, on='Intern_ID')
            risk_clf_df = risk_clf_df.sort_values(by='Risk_Probability', ascending=False)
            
            risk_clf_df['Risk_Probability'] = risk_clf_df['Risk_Probability'].apply(lambda x: f"{x:.1%}")
            risk_clf_df['Risk_Flag'] = risk_clf_df['Predicted_Risk_Class'].apply(lambda x: "High Risk 🔴" if x == 1 else "Safe 🟢")
            
            st.dataframe(
                risk_clf_df[['Intern_Name', 'Risk_Flag', 'Risk_Probability', 'Progress_Numeric', 'Hours']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Intern_Name': st.column_config.TextColumn('Intern Name'),
                    'Risk_Flag': st.column_config.TextColumn('ML Risk Flag'),
                    'Risk_Probability': st.column_config.TextColumn('Risk Probability'),
                    'Progress_Numeric': st.column_config.NumberColumn('Avg Progress %', format="%.1f"),
                    'Hours': st.column_config.NumberColumn('Total Hours', format="%.1f")
                }
            )
        else:
            st.info("Not enough data to predict dropout risks.")