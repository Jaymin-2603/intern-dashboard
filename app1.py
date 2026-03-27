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
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

/* ── Hide default Streamlit chrome on login ── */
[data-testid="stSidebar"]    { display: none; }
[data-testid="stHeader"]     { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"]    { display: none; }
footer                       { display: none; }

/* ── Dark full-page login backdrop ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 50%, rgba(32,127,141,0.25) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(32,127,141,0.14) 0%, transparent 55%),
        radial-gradient(ellipse at 60% 80%, rgba(15,63,71,0.20) 0%, transparent 50%),
        #0b2b30 !important;
    min-height: 100vh;
}

/* animated grid overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(32,127,141,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32,127,141,0.06) 1px, transparent 1px);
    background-size: 48px 48px;
    animation: gridShift 20s linear infinite;
}
@keyframes gridShift {
    0%   { background-position: 0 0,   0 0; }
    100% { background-position: 48px 48px, 48px 48px; }
}

/* floating orbs */
.orb {
    position: fixed; border-radius: 50%;
    filter: blur(80px); pointer-events: none; opacity: 0.20; z-index: 0;
    animation: floatOrb 12s ease-in-out infinite;
}
.orb-1 { width:500px; height:500px; background:#207F8D; top:-12%; left:-10%; animation-duration:14s; }
.orb-2 { width:380px; height:380px; background:#0f6e56; top:55%;  right:-6%; animation-duration:18s; animation-delay:-6s; }
.orb-3 { width:300px; height:300px; background:#2ECC8E; bottom:-10%; left:38%; animation-duration:16s; animation-delay:-3s; }
@keyframes floatOrb {
    0%,100% { transform: translate(0,0)   scale(1); }
    33%     { transform: translate(20px,-30px) scale(1.05); }
    66%     { transform: translate(-15px,20px) scale(0.95); }
}

/* ── Card wrapper — centres all Streamlit content in a glass card ── */
.login-wrapper {
    position: relative; z-index: 10;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 100vh;
    padding: 40px 16px;
}

/* logo */
.logo-mark {
    width: 68px; height: 68px;
    background: linear-gradient(135deg, #207F8D 0%, #2ECC8E 100%);
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 20px;
    font-size: 30px;
    box-shadow: 0 8px 28px rgba(32,127,141,0.45);
    animation: logoPulse 3s ease-in-out infinite;
}
@keyframes logoPulse {
    0%,100% { box-shadow: 0 8px 28px rgba(32,127,141,0.45); }
    50%     { box-shadow: 0 8px 45px rgba(32,127,141,0.75); }
}

.login-title {
    font-family: 'Nunito', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-align: center;
    margin: 0 0 6px !important;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.4);
}
.login-subtitle {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.45);
    text-align: center;
    margin: 0 0 32px;
    letter-spacing: 0.3px;
}
.field-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    display: block;
}
.login-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.10);
    margin: 24px 0 18px;
}
.login-footer {
    font-size: 0.74rem;
    color: rgba(255,255,255,0.22);
    text-align: center;
    margin-top: 20px;
}
.stat-row {
    display: flex; gap: 10px; justify-content: center; margin-top: 20px;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(32,127,141,0.18);
    border: 1px solid rgba(32,127,141,0.32);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    color: rgba(255,255,255,0.60);
    font-weight: 600;
}
.stat-pill b { color: #5dd9e8; }
.login-error {
    background: rgba(232,89,60,0.15);
    border: 1px solid rgba(232,89,60,0.40);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #ff9d84;
    margin-bottom: 14px;
    text-align: center;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }

/* ═══════════════════════════════════════════
   DARK OVERRIDES FOR STREAMLIT WIDGETS
   on the login page
═══════════════════════════════════════════ */

/* Main block background — transparent so the dark stApp shows */
.main .block-container {
    background: transparent !important;
    padding-top: 0 !important;
}

/* All text in the login area white */
.stApp p, .stApp label, .stApp div,
.stApp span, .stApp h1, .stApp h2, .stApp h3 {
    color: rgba(255,255,255,0.90) !important;
}

/* Radio buttons ─ role selector */
[data-testid="stRadio"] {
    background: rgba(0,0,0,0.28) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
}
[data-testid="stRadio"] label {
    flex: 1 !important;
    text-align: center !important;
    padding: 10px 0 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    color: rgba(255,255,255,0.55) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #207F8D, #0f5a63) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(32,127,141,0.45) !important;
}
/* Hide the actual radio circle */
[data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: inherit !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* Selectbox dark theme */
[data-testid="stSelectbox"] > label { display: none !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.90) !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: rgba(255,255,255,0.90) !important;
    background: transparent !important;
}
/* Dropdown menu */
[data-baseweb="popover"] ul {
    background: #0f3f47 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}
[data-baseweb="popover"] li {
    color: rgba(255,255,255,0.85) !important;
    background: transparent !important;
}
[data-baseweb="popover"] li:hover {
    background: rgba(32,127,141,0.30) !important;
}
[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(32,127,141,0.45) !important;
}
/* Selectbox arrow icon */
[data-testid="stSelectbox"] svg { fill: rgba(255,255,255,0.55) !important; }

/* ── Enter button ── */
[data-testid="stButton"] > button,
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
    margin-top: 6px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(32,127,141,0.45) !important;
    letter-spacing: 0.4px !important;
}
[data-testid="stButton"] > button:hover,
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(32,127,141,0.65) !important;
}
[data-testid="stButton"] > button:active,
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Column gaps (card centering) ── */
[data-testid="column"] { padding: 0 8px !important; }

/* ── After login: restore sidebar & light background ── */
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
    st.session_state.role = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "login_error" not in st.session_state:
    st.session_state.login_error = ""


# ─────────────────────────────────────────────
#  LOAD DATA (cached so sign-in is instant)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return run_etl_process()


# ─────────────────────────────────────────────
#  SIGN-IN PAGE
# ─────────────────────────────────────────────
def render_login(mentor_list, intern_list):

    # Floating orbs rendered as fixed overlay
    st.markdown("""
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    """, unsafe_allow_html=True)

    # ── Vertical spacer to push card down ──
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

    # ── Centre using 3 columns ──────────────
    _, card_col, _ = st.columns([1, 1.6, 1])

    with card_col:
        # ── Glass card wrapper (HTML only for styling, widgets flow inside naturally) ──
        st.markdown("""
            <div style='
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(28px);
                -webkit-backdrop-filter: blur(28px);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 28px;
                padding: 44px 48px 40px;
                box-shadow: 0 32px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(32,127,141,0.18) inset;
                animation: cardReveal 0.6s cubic-bezier(0.22,1,0.36,1) both;
                position: relative; z-index: 10;
            '>
            <style>
            @keyframes cardReveal {
                from { opacity:0; transform: translateY(28px) scale(0.97); }
                to   { opacity:1; transform: translateY(0) scale(1); }
            }
            </style>
        """, unsafe_allow_html=True)

        # ── Logo + Title ────────────────────
        st.markdown("""
            <div class='logo-mark'>🧠</div>
            <h1 class='login-title'>Intern Intelligence</h1>
            <p class='login-subtitle'>Powered by Star Schema · Gold Layer Analytics</p>
        """, unsafe_allow_html=True)

        # ── Role Selector ───────────────────
        st.markdown("<div style='margin-bottom:8px'>", unsafe_allow_html=True)
        role_tab = st.radio(
            "Role",
            ["🧑‍🏫  Mentor", "👤  Intern"],
            horizontal=True,
            key="role_radio",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        is_mentor = role_tab.startswith("🧑")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Error message ───────────────────
        if st.session_state.login_error:
            st.markdown(
                f'<div class="login-error">⚠️ {st.session_state.login_error}</div>',
                unsafe_allow_html=True,
            )

        # ── Name selector ───────────────────
        if is_mentor:
            st.markdown('<span class="field-label">Mentor Name</span>', unsafe_allow_html=True)
            selected = st.selectbox(
                "Mentor Name",
                ["— Select your name —"] + mentor_list,
                key="login_mentor_select",
                label_visibility="collapsed",
            )
        else:
            st.markdown('<span class="field-label">Intern Name</span>', unsafe_allow_html=True)
            selected = st.selectbox(
                "Intern Name",
                ["— Select your name —"] + intern_list,
                key="login_intern_select",
                label_visibility="collapsed",
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Enter button ────────────────────
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

        # ── Divider + stats ─────────────────
        st.markdown('<hr class="login-divider">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="stat-row">
                <span class="stat-pill"><b>{len(intern_list)}</b> interns</span>
                <span class="stat-pill"><b>{len(mentor_list)}</b> mentors</span>
                <span class="stat-pill"><b>4</b> courses</span>
            </div>
            <p class="login-footer">Kenexai Hackathon 2k26 · CHARUSAT · Data &amp; AI Challenge</p>
        """, unsafe_allow_html=True)

        # close the glass card div
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DASHBOARD SHELL (after login)
# ─────────────────────────────────────────────
def render_dashboard(fact_lms, fact_activity, dim_intern, dim_course, dim_mentor, quality_alerts):
    # Restore the app background to light theme and show sidebar
    st.markdown("""
        <style>
            /* Override dark login background for dashboard */
            .stApp {
                background:
                    radial-gradient(at 80% 0%, rgba(255,255,255,0.4) 0px, transparent 50%),
                    radial-gradient(at 0% 100%, rgba(32,127,141,0.1) 0px, transparent 50%),
                    #CADDE0 !important;
            }
            .stApp::before { display: none !important; }
            .orb { display: none !important; }
            [data-testid="stSidebar"]  { display: flex !important; }
            [data-testid="stHeader"]   { background: transparent !important; }
            /* Restore text colour for dashboard */
            .stApp p, .stApp label, .stApp div,
            .stApp span, .stApp h1, .stApp h2, .stApp h3 {
                color: #1a3c40 !important;
            }
            /* Selectbox back to light theme */
            [data-testid="stSelectbox"] [data-baseweb="select"] > div {
                background: rgba(255,255,255,0.4) !important;
                border: 1px solid rgba(255,255,255,0.6) !important;
                color: #0b2b30 !important;
            }
            [data-testid="stSelectbox"] [data-baseweb="select"] span,
            [data-testid="stSelectbox"] [data-baseweb="select"] div {
                color: #0b2b30 !important;
            }
            [data-baseweb="popover"] ul {
                background: #cadde0 !important;
            }
            [data-baseweb="popover"] li {
                color: #0b2b30 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    role      = st.session_state.role
    user_name = st.session_state.user_name

    # ── Sidebar header + logout ───────────────
    with st.sidebar:
        role_icon  = "🧑‍🏫" if role == "mentor" else "👤"
        role_label = "Mentor" if role == "mentor" else "Intern"

        st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(32,127,141,0.25),
                 rgba(32,127,141,0.08));border-radius:16px;padding:16px 18px;
                 border:1px solid rgba(32,127,141,0.3);margin-bottom:4px'>
                <div style='font-size:1.6rem;margin-bottom:4px'>{role_icon}</div>
                <div style='font-weight:800;font-size:1rem;color:#0f3f47 !important'>{user_name}</div>
                <div style='font-size:0.78rem;color:#5a7d82 !important;margin-top:2px;
                     text-transform:uppercase;letter-spacing:0.6px'>{role_label}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("⬅  Sign Out", key="logout_btn", use_container_width=True):
            for key in ["authenticated", "role", "user_name", "login_error"]:
                st.session_state[key] = False if key == "authenticated" else (
                    None if key != "login_error" else ""
                )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Route to correct dashboard ────────────
    if role == "mentor":
        from mentor_dashboard import show_mentor_dashboard

        dim_mentor_filtered = dim_mentor[dim_mentor['Mentor_Name'] == user_name]
        if dim_mentor_filtered.empty:
            dim_mentor_filtered = dim_mentor

        show_mentor_dashboard(
            fact_lms, fact_activity, dim_intern,
            dim_course, dim_mentor_filtered
        )

    elif role == "intern":
        from intern_dashboard import show_intern_dashboard

        dim_intern_filtered = dim_intern[dim_intern['Intern_Name'] == user_name]
        if dim_intern_filtered.empty:
            dim_intern_filtered = dim_intern

        show_intern_dashboard(
            fact_lms, fact_activity, dim_intern_filtered, dim_course
        )


# ─────────────────────────────────────────────
#  MAIN FLOW
# ─────────────────────────────────────────────
def main():
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
    main()