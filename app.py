import streamlit as st
import pandas as pd
from warehouse_etl import run_etl_process

# ─── MUST be the very first Streamlit call ───────────────────────────────────
st.set_page_config(
    page_title="Intern Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS: sign-in page + global theme ────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
/* ── Hide default Streamlit chrome on login ── */
[data-testid="stSidebar"]          { display: none; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stDecoration"]       { display: none; }

/* ── Full-page login backdrop ── */
.login-backdrop {
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(32,127,141,0.22) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(32,127,141,0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 60% 80%, rgba(15,63,71,0.18) 0%, transparent 50%),
        #0b2b30;
    overflow: hidden;
}

/* animated grid lines */
.login-backdrop::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(32,127,141,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32,127,141,0.07) 1px, transparent 1px);
    background-size: 48px 48px;
    animation: gridShift 20s linear infinite;
}
@keyframes gridShift {
    0%   { background-position: 0 0,   0 0; }
    100% { background-position: 48px 48px, 48px 48px; }
}

/* floating orbs */
.orb {
    position: absolute; border-radius: 50%;
    filter: blur(80px); pointer-events: none; opacity: 0.18;
    animation: floatOrb 12s ease-in-out infinite;
}
.orb-1 { width:480px; height:480px; background:#207F8D; top:-10%;  left:-8%;
          animation-duration: 14s; }
.orb-2 { width:360px; height:360px; background:#0f3f47; top:50%;   right:-5%;
          animation-duration: 18s; animation-delay: -6s; }
.orb-3 { width:280px; height:280px; background:#2ECC8E; bottom:-8%; left:35%;
          animation-duration: 16s; animation-delay: -3s; }
@keyframes floatOrb {
    0%,100% { transform: translate(0,0)   scale(1); }
    33%     { transform: translate(20px,-30px) scale(1.05); }
    66%     { transform: translate(-15px,20px) scale(0.95); }
}

/* ── Login card ── */
[data-testid="column"]:has(.login-card-marker) > div {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 28px;
    padding: 52px 56px 48px;
    width: 100%;
    margin: 0 auto;
    box-shadow:
        0 32px 80px rgba(0,0,0,0.5),
        0 0 0 1px rgba(32,127,141,0.2) inset;
    animation: cardReveal 0.7s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes cardReveal {
    from { opacity:0; transform: translateY(32px) scale(0.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

/* Strong CSS override for Streamlit Radio text */
div.stRadio > div[role="radiogroup"] label > div, 
div.stRadio > div[role="radiogroup"] label p,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] div,
[data-testid="stRadio"] span {
    color: #ffffff !important;
}

/* logo mark */
.logo-mark {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #207F8D 0%, #2ECC8E 100%);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 24px;
    font-size: 28px;
    box-shadow: 0 8px 24px rgba(32,127,141,0.4);
    animation: logoPulse 3s ease-in-out infinite;
}
@keyframes logoPulse {
    0%,100% { box-shadow: 0 8px 24px rgba(32,127,141,0.4); }
    50%     { box-shadow: 0 8px 40px rgba(32,127,141,0.7); }
}

.login-title {
    font-family: 'Nunito', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff !important;
    text-align: center;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.login-subtitle {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.7) !important;
    text-align: center;
    margin: 0 0 36px;
    letter-spacing: 0.3px;
}

/* role tabs */
.role-tabs {
    display: flex;
    gap: 8px;
    background: rgba(0,0,0,0.25);
    border-radius: 14px;
    padding: 5px;
    margin-bottom: 28px;
}
.role-tab {
    flex: 1; text-align: center;
    padding: 10px 0;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    color: rgba(255,255,255,0.5);
    border: none; background: transparent;
    font-family: 'Nunito', sans-serif;
}
.role-tab.active {
    background: linear-gradient(135deg, #207F8D, #0f5a63);
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(32,127,141,0.45);
}

/* field labels */
.field-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

/* Streamlit selectbox dark override */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #fff !important;
}
[data-testid="stSelectbox"] label { display: none; }

/* Enter button */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #207F8D 0%, #2ECC8E 100%) !important;
    color: #fff !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 14px 0 !important;
    border-radius: 14px !important;
    border: none !important;
    cursor: pointer !important;
    margin-top: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(32,127,141,0.4) !important;
    letter-spacing: 0.4px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(32,127,141,0.6) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* error flash */
.login-error {
    background: rgba(232,89,60,0.15);
    border: 1px solid rgba(232,89,60,0.4);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #ff9d84;
    margin-bottom: 16px;
    text-align: center;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }

/* divider */
.login-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 28px 0 20px;
}

/* footer */
.login-footer {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5) !important;
    text-align: center;
    margin-top: 24px;
}

/* stat pills at bottom of card */
.stat-row {
    display: flex; gap: 10px; justify-content: center; margin-top: 24px;
}
.stat-pill {
    background: rgba(32,127,141,0.18);
    border: 1px solid rgba(32,127,141,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    color: rgba(255,255,255,0.6);
    font-weight: 600;
}
.stat-pill b { color: #5dd9e8; }

/* ── After login: restore sidebar ── */
.sidebar-visible [data-testid="stSidebar"] { display: block !important; }

/* ── Logout button style ── */
.logout-btn > button {
    background: rgba(232,89,60,0.15) !important;
    border: 1px solid rgba(232,89,60,0.35) !important;
    color: #ff9d84 !important;
    font-size: 0.8rem !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.logout-btn > button:hover {
    background: rgba(232,89,60,0.28) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = None          # "mentor" | "intern"
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "login_role_tab" not in st.session_state:
    st.session_state.login_role_tab = "Mentor"
if "login_error" not in st.session_state:
    st.session_state.login_error = ""


# ─────────────────────────────────────────────
#  LOAD DATA (cached so sign-in is instant)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading from Star Schema Database...", ttl=300)
def load_data():
    try:
        import pandas as pd
        from sqlalchemy import create_engine
        
        engine = create_engine('postgresql://hackathon_user:hackathon_password@localhost:5432/intern_analytics')
        
        # Load tables directly from Postgres (blazingly fast)
        fact_lms = pd.read_sql_table('fact_lms', engine)
        fact_activity = pd.read_sql_table('fact_activity', engine)
        dim_intern = pd.read_sql_table('dim_intern', engine)
        dim_course = pd.read_sql_table('dim_course', engine)
        
        # Deduplicate to prevent join multiplication
        dim_intern = dim_intern.drop_duplicates(subset=['Intern_ID'], keep='last')
        
        # Reconstruct missing DB components
        mentors = fact_lms['Mentor Name'].astype(str).str.split(', ').explode().unique()
        dim_mentor = pd.DataFrame(mentors, columns=['Mentor_Name']).dropna().reset_index(drop=True)
        dim_mentor = dim_mentor.drop_duplicates()
        
        daily_eff = fact_activity.groupby(['Intern_ID', 'Date'])['Hours'].sum().reset_index()
        outlier_count = int(len(daily_eff[daily_eff['Hours'] > 12]))
        
        return fact_lms, fact_activity, dim_intern, dim_course, dim_mentor, outlier_count
        
    except Exception as e:
        print(f"⚠️ DB load failed ({e}), falling back to flat-file ETL...")
        return run_etl_process()


# ─────────────────────────────────────────────
#  SIGN-IN PAGE
# ─────────────────────────────────────────────
def render_login(mentor_list, intern_list):
    # Backdrop + orbs (rendered outside the card columns)
    st.markdown("""
        <div class="login-backdrop">
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="orb orb-3"></div>
        </div>
    """, unsafe_allow_html=True)

    # Centre the card using columns
    _, card_col, _ = st.columns([1, 2, 1])

    with card_col:
        st.markdown('<div class="login-card-marker"></div>', unsafe_allow_html=True)

        # Logo + title
        st.markdown("""
            <div class="logo-mark">🧠</div>
            <h1 class="login-title">Intern Intelligence</h1>
            <p class="login-subtitle">Powered by Star Schema · Gold Layer Analytics</p>
        """, unsafe_allow_html=True)

        # Role selector tabs (visual only — actual logic via radio)
        role_tab = st.radio(
            "Select your role",
            ["🧑‍🏫  Mentor", "👤  Intern"],
            horizontal=True,
            key="role_radio",
            label_visibility="collapsed",
        )
        is_mentor = role_tab.startswith("🧑")

        st.markdown("<br>", unsafe_allow_html=True)

        # Error message
        if st.session_state.login_error:
            st.markdown(
                f'<div class="login-error">⚠️ {st.session_state.login_error}</div>',
                unsafe_allow_html=True,
            )

        # Name selector
        if is_mentor:
            st.markdown('<p class="field-label">Mentor Name</p>', unsafe_allow_html=True)
            selected = st.selectbox(
                "Mentor Name",
                ["— Select your name —"] + mentor_list,
                key="login_mentor_select",
                label_visibility="collapsed",
            )
        else:
            st.markdown('<p class="field-label">Intern Name</p>', unsafe_allow_html=True)
            selected = st.selectbox(
                "Intern Name",
                ["— Select your name —"] + intern_list,
                key="login_intern_select",
                label_visibility="collapsed",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Enter button
        if st.button("Enter Dashboard  →", use_container_width=True, key="login_btn"):
            if selected.startswith("—"):
                st.session_state.login_error = "Please select your name to continue."
                st.rerun()
            else:
                st.session_state.authenticated = True
                st.session_state.role = "mentor" if is_mentor else "intern"
                st.session_state.user_name = selected
                st.session_state.login_error = ""
                st.rerun()

        # Divider + stats
        st.markdown('<hr class="login-divider">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="stat-row">
                <span class="stat-pill"><b>{len(intern_list)}</b> interns</span>
                <span class="stat-pill"><b>{len(mentor_list)}</b> mentors</span>
                <span class="stat-pill"><b>4</b> courses</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="login-footer">Kenexai Hackathon 2k26 · CHARUSAT · Data &amp; AI Challenge</p>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DASHBOARD SHELL (after login)
# ─────────────────────────────────────────────
def render_dashboard(fact_lms, fact_activity, dim_intern, dim_course, dim_mentor, quality_alerts):
    # Re-enable sidebar
    st.markdown("""
        <style>
            [data-testid="stSidebar"]  { display: flex !important; }
            [data-testid="stHeader"]   { background: transparent !important; }
        </style>
    """, unsafe_allow_html=True)

    role      = st.session_state.role
    user_name = st.session_state.user_name

    # ── Sidebar header + logout ───────────────
    with st.sidebar:
        # User badge
        role_icon  = "🧑‍🏫" if role == "mentor" else "👤"
        role_label = "Mentor" if role == "mentor" else "Intern"
        first_name = user_name.split()[0]

        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(32,127,141,0.25),
                 rgba(32,127,141,0.08));border-radius:16px;padding:16px 18px;
                 border:1px solid rgba(32,127,141,0.3);margin-bottom:4px'>
                <div style='font-size:1.6rem;margin-bottom:4px'>{role_icon}</div>
                <div style='font-weight:800;font-size:1rem;color:#0f3f47'>{user_name}</div>
                <div style='font-size:0.78rem;color:#5a7d82;margin-top:2px;
                     text-transform:uppercase;letter-spacing:0.6px'>{role_label}</div>
            </div>
        """, unsafe_allow_html=True)

        # Logout
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("⬅  Sign Out", key="logout_btn", use_container_width=True):
            for key in ["authenticated", "role", "user_name", "login_error"]:
                st.session_state[key] = False if key == "authenticated" else None if key != "login_error" else ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Route to correct dashboard ────────────
    if role == "mentor":
        from mentor_dashboard import show_mentor_dashboard

        # Pre-filter: lock mentor to their own name in the selectbox
        # We override dim_mentor to only show this mentor's name so their
        # sidebar selectbox starts on the right person
        dim_mentor_filtered = dim_mentor[dim_mentor['Mentor_Name'] == user_name]
        if dim_mentor_filtered.empty:
            dim_mentor_filtered = dim_mentor   # fallback if name mismatch

        show_mentor_dashboard(
            fact_lms, fact_activity, dim_intern,
            dim_course, dim_mentor_filtered
        )

    elif role == "intern":
        from intern_dashboard import show_intern_dashboard

        # Pre-filter dim_intern to only this intern so their
        # sidebar selectbox starts on the right person
        dim_intern_filtered = dim_intern[dim_intern['Intern_Name'] == user_name]
        if dim_intern_filtered.empty:
            dim_intern_filtered = dim_intern   # fallback

        show_intern_dashboard(
            fact_lms, fact_activity, dim_intern_filtered, dim_course
        )


# ─────────────────────────────────────────────
#  MAIN FLOW
# ─────────────────────────────────────────────
def main():
    # Load ETL data (cached)
    with st.spinner("🔄 Initialising data warehouse..."):
        fact_lms, fact_activity, dim_intern, dim_course, dim_mentor, quality_alerts = load_data()

    mentor_list = sorted(dim_mentor['Mentor_Name'].dropna().tolist())
    intern_list = sorted(dim_intern['Intern_Name'].dropna().tolist())

    if not st.session_state.authenticated:
        render_login(mentor_list, intern_list)
    else:
        render_dashboard(
            fact_lms, fact_activity, dim_intern,
            dim_course, dim_mentor, quality_alerts
        )


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file directly — call main at module level
    main()