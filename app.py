import streamlit as st
import numpy as np
import joblib
import time
import os
import json
import hashlib
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# ENV LOAD
# ─────────────────────────────────────────────
load_dotenv()

ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL",    "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "")
USERS_DB_PATH  = os.getenv("USERS_DB_PATH",  "")
LOG_FILE_PATH  = os.getenv("LOG_FILE_PATH",  "")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SIM Swap Fraud Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');
:root {
    --bg:#0a0c12; --surface:#111520; --card:#161b28; --border:#1e2840;
    --accent:#00e5ff; --accent2:#ff3d71; --accent3:#36d399;
    --text:#c9d1e0; --muted:#5b6680;
    --mono:'Share Tech Mono',monospace; --body:'Barlow',sans-serif;
}
html,body,[class*="css"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--body)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem 3rem;max-width:1400px;margin:auto;}

/* HERO */
.hero{text-align:center;padding:2.5rem 1rem 1.5rem;}
.hero-badge{display:inline-block;font-family:var(--mono);font-size:.68rem;letter-spacing:.2em;color:var(--accent);border:1px solid var(--accent);padding:.25rem .9rem;border-radius:2px;text-transform:uppercase;margin-bottom:.9rem;background:rgba(0,229,255,.05);}
.hero h1{font-family:var(--body)!important;font-size:2.6rem!important;font-weight:700!important;color:#fff!important;letter-spacing:-.02em;line-height:1.1;margin-bottom:.4rem;}
.hero h1 span{color:var(--accent);}
.hero p{color:var(--muted);font-size:1rem;max-width:550px;margin:0 auto;}
.scanline{width:100%;height:1px;background:linear-gradient(90deg,transparent 0%,var(--accent) 50%,transparent 100%);margin:1.5rem auto 0;opacity:.4;}

/* AUTH */
.auth-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:2.4rem 2.2rem 2rem;position:relative;overflow:hidden;max-width:460px;margin:0 auto;}
.auth-card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),#0066ff);}
.auth-title{font-family:var(--mono);font-size:1rem;letter-spacing:.15em;color:var(--accent);text-transform:uppercase;margin-bottom:.3rem;}
.auth-sub{font-size:.82rem;color:var(--muted);margin-bottom:1.6rem;}
.auth-divider{text-align:center;color:var(--muted);font-size:.75rem;font-family:var(--mono);margin:1rem 0;position:relative;}
.auth-divider::before,.auth-divider::after{content:"";position:absolute;top:50%;width:42%;height:1px;background:var(--border);}
.auth-divider::before{left:0;}.auth-divider::after{right:0;}

/* SECTION LABELS */
.section-label{font-family:var(--mono);font-size:.65rem;letter-spacing:.18em;color:var(--accent);text-transform:uppercase;margin-bottom:.5rem;display:flex;align-items:center;gap:.5rem;}
.section-label::before{content:"";display:inline-block;width:18px;height:1px;background:var(--accent);}

/* CARDS */
.card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1.4rem 1.4rem 1rem;margin-bottom:1rem;position:relative;overflow:hidden;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent) 0%,transparent 100%);}

/* RESULT */
.result-panel{border-radius:8px;padding:1.8rem 2rem;margin-top:1.4rem;text-align:center;border:1px solid var(--border);}
.result-panel.safe{background:linear-gradient(135deg,#0a2a1a,#0d1f0f);border-color:var(--accent3);}
.result-panel.medium{background:linear-gradient(135deg,#2a1f0a,#1f1508);border-color:#f0a500;}
.result-panel.danger{background:linear-gradient(135deg,#2a0a0a,#1f0808);border-color:var(--accent2);}
.result-score{font-family:var(--mono);font-size:4rem;font-weight:bold;line-height:1;margin:.3rem 0;}
.result-label{font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;font-family:var(--mono);margin-bottom:.5rem;color:var(--muted);}
.result-verdict{font-size:1.3rem;font-weight:700;margin-top:.3rem;}
.risk-bar-wrap{margin:1rem 0 .4rem;}
.risk-bar-track{width:100%;height:6px;background:var(--border);border-radius:3px;overflow:hidden;}
.risk-bar-fill{height:100%;border-radius:3px;}

/* GAUGE */
.gauge-bar-track{width:100%;height:10px;border-radius:5px;background:linear-gradient(90deg,#36d399 0%,#f0a500 50%,#ff3d71 100%);position:relative;margin-bottom:.3rem;}
.gauge-needle{position:absolute;top:-4px;width:3px;height:18px;background:#fff;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 6px rgba(255,255,255,.6);}
.gauge-labels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;color:var(--muted);}
.factor-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.55rem;font-family:var(--mono);font-size:.72rem;}
.factor-name{width:200px;flex-shrink:0;color:var(--text);}
.factor-track{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden;}
.factor-fill{height:100%;border-radius:3px;}
.factor-val{width:36px;text-align:right;color:var(--muted);flex-shrink:0;}

/* INPUTS */
div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:4px!important;font-family:var(--mono)!important;}
div[data-testid="stNumberInput"] input:focus,div[data-testid="stTextInput"] input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(0,229,255,.15)!important;}
label{color:var(--text)!important;font-size:.88rem!important;}

/* BUTTONS */
div[data-testid="stFormSubmitButton"] button,.stButton>button{background:linear-gradient(135deg,#003d4d,#004a5c)!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:var(--mono)!important;letter-spacing:.12em!important;font-size:.85rem!important;padding:.65rem 2rem!important;border-radius:4px!important;width:100%!important;transition:all .2s ease!important;text-transform:uppercase;}
div[data-testid="stFormSubmitButton"] button:hover,.stButton>button:hover{background:rgba(0,229,255,.15)!important;box-shadow:0 0 20px rgba(0,229,255,.2)!important;}

/* FLAGS */
.flag-row{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.5rem;}
.flag-badge{font-family:var(--mono);font-size:.7rem;padding:.2rem .6rem;border-radius:2px;border:1px solid;}
.flag-on{background:rgba(255,61,113,.12);border-color:var(--accent2);color:var(--accent2);}
.flag-off{background:rgba(54,211,153,.1);border-color:var(--accent3);color:var(--accent3);}
.field-desc{font-size:.74rem;color:var(--muted);margin-top:.15rem;margin-bottom:.7rem;line-height:1.4;}

/* ALERTS */
.alert-banner{border-radius:6px;padding:.9rem 1.2rem;margin-bottom:1rem;font-family:var(--mono);font-size:.8rem;letter-spacing:.05em;display:flex;align-items:center;gap:.8rem;animation:fadeIn .4s ease;}
.alert-danger{background:rgba(255,61,113,.12);border:1px solid #ff3d71;color:#ff3d71;}
.alert-warning{background:rgba(240,165,0,.12);border:1px solid #f0a500;color:#f0a500;}
@keyframes fadeIn{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}

/* LOG TABLE */
.log-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.72rem;}
.log-table th{text-align:left;padding:.4rem .6rem;color:var(--accent);border-bottom:1px solid var(--border);letter-spacing:.08em;font-weight:normal;}
.log-table td{padding:.4rem .6rem;border-bottom:1px solid rgba(30,40,64,.5);color:var(--text);vertical-align:middle;}
.log-table tr:hover td{background:rgba(0,229,255,.03);}
.pill{display:inline-block;padding:.1rem .5rem;border-radius:10px;font-size:.68rem;}
.pill-danger{background:rgba(255,61,113,.18);color:#ff3d71;}
.pill-warning{background:rgba(240,165,0,.18);color:#f0a500;}
.pill-safe{background:rgba(54,211,153,.15);color:#36d399;}

/* BADGES */
.admin-badge{display:inline-flex;align-items:center;gap:.4rem;background:rgba(255,61,113,.1);border:1px solid #ff3d71;color:#ff3d71;font-family:var(--mono);font-size:.68rem;letter-spacing:.12em;padding:.2rem .7rem;border-radius:2px;text-transform:uppercase;}
.user-badge{display:inline-flex;align-items:center;gap:.4rem;background:rgba(0,229,255,.08);border:1px solid var(--accent);color:var(--accent);font-family:var(--mono);font-size:.68rem;letter-spacing:.12em;padding:.2rem .7rem;border-radius:2px;text-transform:uppercase;}

/* STAT CARDS */
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1rem;text-align:center;}
.stat-num{font-family:var(--mono);font-size:2rem;font-weight:bold;}
.stat-lbl{font-size:.68rem;color:var(--muted);font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;}
.divider{height:1px;background:var(--border);margin:1.2rem 0;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# UTILITIES — Users DB (flat JSON)
# ═══════════════════════════════════════════════
def _hash(password: str) -> str:
    return hashlib.sha256((password + APP_SECRET_KEY).encode()).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_DB_PATH):
        return {}
    with open(USERS_DB_PATH) as f:
        return json.load(f)

def save_users(db: dict):
    with open(USERS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def register_user(email: str, name: str, password: str):
    db = load_users()
    email = email.strip().lower()
    if email in db:
        return False, "An account with this email already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email address."
    db[email] = {"name": name.strip(), "password_hash": _hash(password),
                 "created_at": datetime.now().isoformat()}
    save_users(db)
    return True, "Account created successfully."

def verify_user(email: str, password: str):
    email = email.strip().lower()
    if email == ADMIN_EMAIL.lower() and password == ADMIN_PASSWORD:
        return True, "", {"name": "Administrator", "email": email, "role": "admin"}
    db = load_users()
    if email not in db:
        return False, "No account found with that email.", {}
    if db[email]["password_hash"] != _hash(password):
        return False, "Incorrect password.", {}
    return True, "", {**db[email], "email": email, "role": "user"}


# ═══════════════════════════════════════════════
# UTILITIES — Activity Log (persistent JSON)
# ═══════════════════════════════════════════════
def load_log() -> list:
    if not os.path.exists(LOG_FILE_PATH):
        return []
    with open(LOG_FILE_PATH) as f:
        return json.load(f)

def append_log(record: dict):
    log = load_log()
    log.append(record)
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(log, f, indent=2)

def add_log(source: str, risk_pct: float, tier: str, inputs: dict):
    user = st.session_state.get("user", {})
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performed_by": user.get("email", "unknown"),
        "source": source,
        "risk_pct": round(risk_pct, 2),
        "tier": tier,
        **inputs,
    }
    append_log(record)


# ═══════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = {}
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"


# ═══════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════
@st.cache_resource
def load_model():
    path = "sim_swap_model_v2.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()

FEATURE_COLS = [
    "sim_swap_time_gap_minutes","device_change_flag","sim_type_change_flag",
    "imsi_change_flag","iccid_change_flag","otp_and_sim_change_geo_hash_length",
    "recent_sim_activation_days","num_sim_changes_last_30d",
    "previous_sim_holder_tenure_days","account_age_days","ip_change_flag",
]


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════
def get_tier(risk_pct):
    if risk_pct < 30:   return "safe",   "✅ LOW RISK — NORMAL BEHAVIOUR",        "#36d399","#36d399"
    elif risk_pct < 70: return "medium", "⚠️ MEDIUM RISK — SUSPICIOUS ACTIVITY",  "#f0a500","#f0a500"
    else:               return "danger", "🚨 HIGH RISK — STRONG FRAUD INDICATORS", "#ff3d71","#ff3d71"

def render_alert(tier, risk_pct, source="Manual Input"):
    if tier not in ("danger","medium"): return
    css  = "alert-danger" if tier=="danger" else "alert-warning"
    icon = "🚨" if tier=="danger" else "⚠️"
    msg  = (f"HIGH-RISK ALERT: {source} scored {risk_pct:.1f}% — Immediate review required."
            if tier=="danger" else
            f"SUSPICIOUS ACTIVITY: {source} scored {risk_pct:.1f}% — Flag for secondary verification.")
    st.markdown(f'<div class="alert-banner {css}"><span style="font-size:1.2rem">{icon}</span><span>{msg}</span></div>',
                unsafe_allow_html=True)
    flash = "🚨 ALERT" if tier=="danger" else "⚠️ WARNING"
    st.markdown(f"""<script>(function(){{var o=document.title,c=0,iv=setInterval(function(){{
        document.title=c%2==0?"{flash} — SIM Swap Detector":o;c++;if(c>8){{clearInterval(iv);document.title=o;}}}},600);}})();</script>""",
                unsafe_allow_html=True)

def render_risk_breakdown(inputs, risk_pct):
    tg = inputs["sim_swap_time_gap_minutes"]
    factors = {
        "SIM Swap Time Gap":      max(0,min(100,(1-tg/500)*100)) if tg<500 else 0,
        "Device Change":          100 if inputs["device_change_flag"] else 0,
        "SIM Type Change":        100 if inputs["sim_type_change_flag"] else 0,
        "IMSI Change":            100 if inputs["imsi_change_flag"] else 0,
        "ICCID Change":           100 if inputs["iccid_change_flag"] else 0,
        "IP Change":              100 if inputs["ip_change_flag"] else 0,
        "Geo Distance (OTP–SIM)": max(0,min(100,(1-inputs["otp_and_sim_change_geo_hash_length"]/12)*100)),
        "Swap Frequency (30d)":   min(100,inputs["num_sim_changes_last_30d"]/5*100),
        "SIM Activation Recency": max(0,min(100,(1-inputs["recent_sim_activation_days"]/30)*100)),
        "Account Age":            max(0,min(100,(1-inputs["account_age_days"]/365)*100)),
        "Prev SIM Tenure":        max(0,min(100,(1-inputs["previous_sim_holder_tenure_days"]/365)*100)),
    }
    st.markdown('<div class="section-label">📊 Risk Factor Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"""<div style="margin-bottom:1rem;">
        <div style="font-family:var(--mono);font-size:.68rem;color:var(--muted);margin-bottom:.4rem;">OVERALL FRAUD PROBABILITY GAUGE</div>
        <div class="gauge-bar-track"><div class="gauge-needle" style="left:{risk_pct}%"></div></div>
        <div class="gauge-labels"><span>0% — Safe</span><span>50% — Suspicious</span><span>100% — Fraud</span></div>
    </div><div style="height:1px;background:var(--border);margin:.8rem 0;"></div>""", unsafe_allow_html=True)
    for fn, sc in sorted(factors.items(), key=lambda x:-x[1]):
        fc = "#ff3d71" if sc>=70 else ("#f0a500" if sc>=40 else "#36d399")
        st.markdown(f"""<div class="factor-row">
            <div class="factor-name">{fn}</div>
            <div class="factor-track"><div class="factor-fill" style="width:{sc:.0f}%;background:{fc};"></div></div>
            <div class="factor-val">{sc:.0f}</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE — LOGIN
# ═══════════════════════════════════════════════
def page_login():
    st.markdown("""<div class="hero">
        <div class="hero-badge">⬡ Telecom Security Intelligence</div>
        <h1>SIM Swap <span>Fraud</span> Detector</h1>
        <p>Sign in to access the ML-powered fraud detection platform.</p>
        <div class="scanline"></div></div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1,1.1,1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">🔐 Sign In</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Enter your credentials to continue.</div>', unsafe_allow_html=True)

        with st.form("login_form"):
            email    = st.text_input("Email Address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            login_btn = st.form_submit_button("→  SIGN IN")

        if login_btn:
            if not email or not password:
                st.error("Please fill in all fields.")
            else:
                ok, err, user = verify_user(email, password)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.success(f"Welcome back, {user['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(err)

        st.markdown('<div class="auth-divider">OR</div>', unsafe_allow_html=True)
        if st.button("Create a new account →"):
            st.session_state.auth_mode = "signup"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE — SIGNUP
# ═══════════════════════════════════════════════
def page_signup():
    st.markdown("""<div class="hero">
        <div class="hero-badge">⬡ Telecom Security Intelligence</div>
        <h1>SIM Swap <span>Fraud</span> Detector</h1>
        <p>Create an account to access the fraud detection platform.</p>
        <div class="scanline"></div></div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1,1.1,1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">🆕 Create Account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Fill in the details below to register.</div>', unsafe_allow_html=True)

        with st.form("signup_form"):
            name     = st.text_input("Full Name",        placeholder="Jane Doe")
            email    = st.text_input("Email Address",    placeholder="you@example.com")
            password = st.text_input("Password",         type="password", placeholder="Min 6 characters")
            confirm  = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            reg_btn  = st.form_submit_button("→  CREATE ACCOUNT")

        if reg_btn:
            if not all([name, email, password, confirm]):
                st.error("Please fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(email, name, password)
                if ok:
                    st.success(msg + " You can now sign in.")
                    time.sleep(1)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown('<div class="auth-divider">ALREADY HAVE AN ACCOUNT?</div>', unsafe_allow_html=True)
        if st.button("← Back to Sign In"):
            st.session_state.auth_mode = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE — ADMIN CONSOLE (embedded tab)
# ═══════════════════════════════════════════════
def render_admin_console():
    log       = load_log()
    df_log    = pd.DataFrame(log) if log else pd.DataFrame()
    users_db  = load_users()

    total      = len(log)
    high_n     = sum(1 for r in log if r.get("tier")=="danger")
    med_n      = sum(1 for r in log if r.get("tier")=="medium")
    safe_n     = sum(1 for r in log if r.get("tier")=="safe")
    user_count = len(users_db)

    # Stats
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, num, lbl, clr in [
        (c1,total,     "Total Events",   "#00e5ff"),
        (c2,high_n,    "High Risk",      "#ff3d71"),
        (c3,med_n,     "Medium Risk",    "#f0a500"),
        (c4,safe_n,    "Safe / Low",     "#36d399"),
        (c5,user_count,"Registered Users","#a78bfa"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:{clr};">{num}</div><div class="stat-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    at1, at2, at3 = st.tabs(["📋 Activity Log", "👥 Registered Users", "⚙️ System Info"])

    # ── Activity Log ──
    with at1:
        st.markdown('<div class="section-label">📋 Full Activity Log (Admin Only)</div>', unsafe_allow_html=True)
        if df_log.empty:
            st.info("No activity recorded yet.")
        else:
            fc1,fc2,fc3 = st.columns(3)
            with fc1: tier_f = st.selectbox("Filter by Tier", ["All","danger","medium","safe"])
            with fc2: user_f = st.text_input("Filter by Email", placeholder="Search by email…")
            with fc3: st.markdown("<br>",unsafe_allow_html=True); show_n = st.slider("Max rows",10,500,100)

            dv = df_log.copy()
            if tier_f != "All":   dv = dv[dv["tier"]==tier_f]
            if user_f.strip():    dv = dv[dv["performed_by"].str.contains(user_f.strip(),case=False,na=False)]
            dv = dv.sort_values("timestamp",ascending=False).head(show_n).reset_index(drop=True)

            def st_tier(v):
                if v=="danger":  return "background-color:rgba(255,61,113,.15);color:#ff3d71;font-weight:bold"
                elif v=="medium":return "background-color:rgba(240,165,0,.12);color:#f0a500"
                else:            return "background-color:rgba(54,211,153,.1);color:#36d399"

            show_cols = [c for c in ["timestamp","performed_by","source","risk_pct","tier"] if c in dv.columns]
            st.dataframe(dv[show_cols].style.applymap(st_tier,subset=["tier"]),
                         use_container_width=True, height=380)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">⬇ Export (Admin Only)</div>', unsafe_allow_html=True)
            dl1,dl2,dl3 = st.columns(3)
            with dl1:
                st.download_button("⬇ Full Log CSV", df_log.to_csv(index=False).encode(),
                                   f"full_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)
            with dl2:
                df_h = df_log[df_log["tier"]=="danger"] if "tier" in df_log.columns else pd.DataFrame()
                st.download_button("⬇ High-Risk Only CSV", df_h.to_csv(index=False).encode() if not df_h.empty else b"",
                                   f"high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)
            with dl3:
                st.download_button("⬇ Filtered View CSV", dv.to_csv(index=False).encode(),
                                   f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear Entire Activity Log"):
                with open(LOG_FILE_PATH,"w") as f: json.dump([],f)
                st.success("Log cleared."); st.rerun()

    # ── Users ──
    with at2:
        st.markdown('<div class="section-label">👥 Registered Users</div>', unsafe_allow_html=True)
        if not users_db:
            st.info("No registered users yet.")
        else:
            rows = [{"Email":e,"Name":d.get("name","—"),"Registered":d.get("created_at","—")[:10],"Role":"user"}
                    for e,d in users_db.items()]
            df_u = pd.DataFrame(rows)
            st.dataframe(df_u, use_container_width=True, height=300)
            st.download_button("⬇ Export Users CSV", df_u.to_csv(index=False).encode(),
                               "registered_users.csv","text/csv")

    # ── System ──
    with at3:
        st.markdown('<div class="section-label">⚙️ System Info</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        log_sz = os.path.getsize(LOG_FILE_PATH) if os.path.exists(LOG_FILE_PATH) else 0
        db_sz  = os.path.getsize(USERS_DB_PATH)  if os.path.exists(USERS_DB_PATH)  else 0
        st.markdown(f"""<table class="log-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model Status</td><td>{"✅ Loaded" if model else "❌ Not Found"}</td></tr>
            <tr><td>Admin Email</td><td>{ADMIN_EMAIL}</td></tr>
            <tr><td>Log File</td><td>{LOG_FILE_PATH} ({log_sz:,} bytes)</td></tr>
            <tr><td>Users DB</td><td>{USERS_DB_PATH} ({db_sz:,} bytes)</td></tr>
            <tr><td>Total Log Entries</td><td>{total}</td></tr>
            <tr><td>Server Time</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE — MAIN APP (authenticated)
# ═══════════════════════════════════════════════
def page_main():
    user     = st.session_state.user
    is_admin = user.get("role") == "admin"

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        role_html = '<span class="admin-badge">🔑 ADMIN</span>' if is_admin else '<span class="user-badge">👤 USER</span>'
        st.markdown(f"""<div style="margin-bottom:1rem;">{role_html}<br>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.8rem;color:#c9d1e0;margin-top:.5rem;">{user.get('name','')}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#5b6680;">{user.get('email','')}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:#1e2840;margin:.6rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:#00e5ff;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.7rem;">📋 My Session Log</div>', unsafe_allow_html=True)

        my_log = [r for r in load_log() if r.get("performed_by")==user.get("email")]

        if not my_log:
            st.markdown('<div style="color:#5b6680;font-size:.78rem;font-family:\'Share Tech Mono\',monospace;">No events yet.</div>', unsafe_allow_html=True)
        else:
            dn = sum(1 for r in my_log if r["tier"]=="danger")
            mn = sum(1 for r in my_log if r["tier"]=="medium")
            sn = sum(1 for r in my_log if r["tier"]=="safe")
            st.markdown(f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem;"><span class="pill pill-danger">🚨{dn}</span><span class="pill pill-warning">⚠️{mn}</span><span class="pill pill-safe">✅{sn}</span></div>',
                        unsafe_allow_html=True)
            rows_html = ""
            for r in reversed(my_log[-20:]):
                tp = {"danger":'<span class="pill pill-danger">HIGH</span>',
                      "medium":'<span class="pill pill-warning">MED</span>',
                      "safe":  '<span class="pill pill-safe">SAFE</span>'}[r["tier"]]
                rows_html += f"<tr><td>{r['timestamp'][11:]}</td><td style='max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{r.get('source','')[:14]}</td><td>{r['risk_pct']}%</td><td>{tp}</td></tr>"
            st.markdown(f'<table class="log-table"><thead><tr><th>Time</th><th>Src</th><th>Score</th><th>Tier</th></tr></thead><tbody>{rows_html}</tbody></table>',
                        unsafe_allow_html=True)

        # Log download — ADMIN ONLY
        st.markdown('<div style="height:1px;background:#1e2840;margin:.8rem 0;"></div>', unsafe_allow_html=True)
        if is_admin:
            full_log = load_log()
            if full_log:
                full_csv = pd.DataFrame(full_log).to_csv(index=False).encode()
                st.download_button("⬇ Download Full Log", data=full_csv,
                                   file_name=f"full_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv", use_container_width=True)
        else:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#5b6680;
                padding:.5rem;border:1px solid #1e2840;border-radius:4px;">
                🔒 Log export restricted to administrators.</div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:#1e2840;margin:.6rem 0;"></div>', unsafe_allow_html=True)
        if st.button("⏻  Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = {}
            st.session_state.auth_mode = "login"
            st.rerun()

    # ── Hero ─────────────────────────────────────
    st.markdown("""<div class="hero">
        <div class="hero-badge">⬡ Telecom Security Intelligence</div>
        <h1>SIM Swap <span>Fraud</span> Detector</h1>
        <p>ML-powered real-time detection of SIM swapping attacks using behavioral and network signals.</p>
        <div class="scanline"></div></div>""", unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ **Model not found.** Place `sim_swap_model_v2.pkl` in the same directory, then restart.")
        return

    # ── Tabs ─────────────────────────────────────
    tab_labels = ["🔍  Manual Detection", "📂  Batch CSV Upload"]
    if is_admin: tab_labels.append("🔑  Admin Console")
    tabs = st.tabs(tab_labels)
    t_manual = tabs[0]
    t_batch  = tabs[1]
    t_admin  = tabs[2] if is_admin else None

    # ══ TAB: MANUAL DETECTION ════════════════════
    with t_manual:
        with st.expander("📖 Feature Reference", expanded=False):
            for lbl, col, desc in [
                ("⏱ SIM Swap Time Gap",      "sim_swap_time_gap_minutes",           "Minutes since last swap. Short = higher risk."),
                ("📱 Device Change",          "device_change_flag",                  "New handset detected."),
                ("💳 SIM Type Change",        "sim_type_change_flag",                "SIM form factor changed (e.g. Standard→eSIM)."),
                ("🔢 IMSI Change",            "imsi_change_flag",                    "Mobile identity changed."),
                ("🔖 ICCID Change",           "iccid_change_flag",                   "Physical SIM replaced."),
                ("🌐 OTP+SIM Geo-hash",       "otp_and_sim_change_geo_hash_length",  "Lower = farther apart = riskier."),
                ("📅 SIM Activation Recency", "recent_sim_activation_days",          "Days since current SIM activated."),
                ("🔄 SIM Changes 30d",        "num_sim_changes_last_30d",            ">2 swaps in 30d is very suspicious."),
                ("📆 Prev SIM Tenure",        "previous_sim_holder_tenure_days",     "How long the replaced SIM was active."),
                ("🏦 Account Age",            "account_age_days",                    "New accounts with swap activity = riskier."),
                ("🌍 IP Change",              "ip_change_flag",                      "IP changed during swap."),
            ]:
                st.markdown(f"**{lbl}** — `{col}`"); st.caption(desc); st.markdown("---")

        st.markdown('<div class="section-label">Input Signals</div>', unsafe_allow_html=True)

        with st.form("fraud_form"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">⏱ Temporal Signals</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: sim_swap_time_gap_minutes = st.number_input("SIM Swap Time Gap (min)",0,100000,120,1); st.markdown('<div class="field-desc">Minutes since last swap.</div>',unsafe_allow_html=True)
            with c2: recent_sim_activation_days = st.number_input("SIM Activation (days)",0,3650,30,1); st.markdown('<div class="field-desc">Days since SIM activated.</div>',unsafe_allow_html=True)
            with c3: previous_sim_holder_tenure_days = st.number_input("Prev SIM Tenure (days)",0,3650,180,1); st.markdown('<div class="field-desc">How long replaced SIM was active.</div>',unsafe_allow_html=True)
            with c4: account_age_days = st.number_input("Account Age (days)",0,10000,365,1); st.markdown('<div class="field-desc">Total account age in days.</div>',unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🚩 Binary Flags (0=No, 1=Yes)</div>', unsafe_allow_html=True)
            c5,c6,c7,c8,c9 = st.columns(5)
            with c5: device_change_flag = st.number_input("Device Change",0,1,0,1); st.markdown('<div class="field-desc">New handset.</div>',unsafe_allow_html=True)
            with c6: sim_type_change_flag = st.number_input("SIM Type Change",0,1,0,1); st.markdown('<div class="field-desc">Form factor changed.</div>',unsafe_allow_html=True)
            with c7: imsi_change_flag = st.number_input("IMSI Change",0,1,0,1); st.markdown('<div class="field-desc">Identity changed.</div>',unsafe_allow_html=True)
            with c8: iccid_change_flag = st.number_input("ICCID Change",0,1,0,1); st.markdown('<div class="field-desc">Physical SIM replaced.</div>',unsafe_allow_html=True)
            with c9: ip_change_flag = st.number_input("IP Change",0,1,0,1); st.markdown('<div class="field-desc">IP changed.</div>',unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">📊 Behavioral & Geo Signals</div>', unsafe_allow_html=True)
            c10,c11 = st.columns(2)
            with c10: num_sim_changes_last_30d = st.number_input("SIM Changes (30 days)",0,50,1,1); st.markdown('<div class="field-desc">>2 is very suspicious.</div>',unsafe_allow_html=True)
            with c11: otp_and_sim_change_geo_hash_length = st.number_input("OTP–SIM Geo-hash",0.0,12.0,6.0,0.1,format="%.1f"); st.markdown('<div class="field-desc">Lower = farther apart = riskier.</div>',unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            submitted = st.form_submit_button("🔍  ANALYZE FRAUD RISK")

        if submitted:
            inputs = {
                "sim_swap_time_gap_minutes": sim_swap_time_gap_minutes,
                "device_change_flag": device_change_flag, "sim_type_change_flag": sim_type_change_flag,
                "imsi_change_flag": imsi_change_flag, "iccid_change_flag": iccid_change_flag,
                "otp_and_sim_change_geo_hash_length": otp_and_sim_change_geo_hash_length,
                "recent_sim_activation_days": recent_sim_activation_days,
                "num_sim_changes_last_30d": num_sim_changes_last_30d,
                "previous_sim_holder_tenure_days": previous_sim_holder_tenure_days,
                "account_age_days": account_age_days, "ip_change_flag": ip_change_flag,
            }
            arr = np.array([[inputs[c] for c in FEATURE_COLS]])
            with st.spinner("Running inference…"):
                time.sleep(0.6)
                prob = model.predict_proba(arr)[0][1]
            risk_pct = prob * 100
            tier, verdict, score_color, bar_color = get_tier(risk_pct)
            add_log("Manual Input", risk_pct, tier, inputs)
            render_alert(tier, risk_pct, "Manual Input")

            st.markdown(f"""<div class="result-panel {tier}">
                <div class="result-label">Fraud Probability Score</div>
                <div class="result-score" style="color:{score_color}">{risk_pct:.1f}%</div>
                <div class="result-verdict" style="color:{score_color}">{verdict}</div>
                <div class="risk-bar-wrap"><div class="risk-bar-track">
                    <div class="risk-bar-fill" style="width:{risk_pct}%;background:{bar_color};"></div>
                </div></div>
                <small style="color:#5b6680;font-family:'Share Tech Mono',monospace;">Raw confidence: {prob:.4f}</small>
            </div>""", unsafe_allow_html=True)

            render_risk_breakdown(inputs, risk_pct)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">🔎 Signal Breakdown</div>', unsafe_allow_html=True)
            cA, cB = st.columns(2)
            with cA:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Binary Flags Active**")
                flags = {"Device Change":device_change_flag,"SIM Type Change":sim_type_change_flag,
                         "IMSI Change":imsi_change_flag,"ICCID Change":iccid_change_flag,"IP Change":ip_change_flag}
                b = '<div class="flag-row">'
                for n,v in flags.items():
                    b += f'<span class="flag-badge {"flag-on" if v else "flag-off"}">{"●" if v else "○"} {n}</span>'
                st.markdown(b+"</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with cB:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Numeric Risk Signals**")
                warns = []
                if sim_swap_time_gap_minutes < 30:         warns.append(("🔴",f"Short gap: {sim_swap_time_gap_minutes} min"))
                if num_sim_changes_last_30d > 2:           warns.append(("🔴",f"High freq: {num_sim_changes_last_30d}x/30d"))
                if recent_sim_activation_days < 7:         warns.append(("🟡",f"SIM {recent_sim_activation_days}d old"))
                if account_age_days < 90:                  warns.append(("🟡",f"New acct: {account_age_days}d"))
                if otp_and_sim_change_geo_hash_length < 3: warns.append(("🔴",f"Geo gap: precision {otp_and_sim_change_geo_hash_length}"))
                if warns:
                    for ic,m in warns: st.markdown(f"{ic} `{m}`")
                else:
                    st.success("No strong numeric risk signals.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ══ TAB: BATCH CSV ═══════════════════════════
    with t_batch:
        st.markdown('<div class="section-label">📂 Batch Detection</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card"><div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;color:#c9d1e0;line-height:1.8;">
            Required columns:<br><br>
            <code>sim_swap_time_gap_minutes, device_change_flag, sim_type_change_flag,
            imsi_change_flag, iccid_change_flag, otp_and_sim_change_geo_hash_length,
            recent_sim_activation_days, num_sim_changes_last_30d,
            previous_sim_holder_tenure_days, account_age_days, ip_change_flag</code><br><br>
            Optional: <code>subscriber_id</code>
        </div></div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                miss = [c for c in FEATURE_COLS if c not in df_up.columns]
                if miss:
                    st.error(f"❌ Missing: {', '.join(miss)}")
                else:
                    X = df_up[FEATURE_COLS].values
                    with st.spinner(f"Running inference on {len(df_up)} records…"):
                        time.sleep(0.5)
                        probs = model.predict_proba(X)[:,1]
                    df_up["fraud_probability_%"] = (probs*100).round(2)
                    df_up["tier_label"] = df_up["fraud_probability_%"].apply(
                        lambda p: "HIGH" if p>=70 else ("MEDIUM" if p>=30 else "SAFE"))
                    df_res = df_up.sort_values("fraud_probability_%",ascending=False).reset_index(drop=True)

                    hn=(df_res["tier_label"]=="HIGH").sum(); mn=(df_res["tier_label"]=="MEDIUM").sum(); sn=(df_res["tier_label"]=="SAFE").sum()
                    if hn>0: render_alert("danger",df_res[df_res["tier_label"]=="HIGH"]["fraud_probability_%"].max(),f"Batch ({hn} HIGH-risk)")

                    st.markdown(f"""<div style="display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap;">
                        <div class="stat-card" style="flex:1"><div class="stat-num" style="color:#ff3d71;">{hn}</div><div class="stat-lbl">HIGH</div></div>
                        <div class="stat-card" style="flex:1"><div class="stat-num" style="color:#f0a500;">{mn}</div><div class="stat-lbl">MEDIUM</div></div>
                        <div class="stat-card" style="flex:1"><div class="stat-num" style="color:#36d399;">{sn}</div><div class="stat-lbl">SAFE</div></div>
                        <div class="stat-card" style="flex:1"><div class="stat-num" style="color:#00e5ff;">{len(df_res)}</div><div class="stat-lbl">TOTAL</div></div>
                    </div>""", unsafe_allow_html=True)

                    tier_map={"HIGH":"danger","MEDIUM":"medium","SAFE":"safe"}
                    for idx,row in df_res.iterrows():
                        src=str(row.get("subscriber_id",f"Row{idx+1}"))
                        add_log(f"CSV:{src}",row["fraud_probability_%"],tier_map[row["tier_label"]],{c:row[c] for c in FEATURE_COLS})

                    def st_batch(v):
                        if v=="HIGH":    return "background-color:rgba(255,61,113,.15);color:#ff3d71;font-weight:bold"
                        elif v=="MEDIUM":return "background-color:rgba(240,165,0,.12);color:#f0a500"
                        else:            return "background-color:rgba(54,211,153,.1);color:#36d399"

                    st.dataframe(df_res.style.applymap(st_batch,subset=["tier_label"]),use_container_width=True,height=400)
                    st.download_button("⬇ Download Results CSV", df_res.to_csv(index=False).encode(),
                                       f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv","text/csv",use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ══ TAB: ADMIN CONSOLE ═══════════════════════
    if is_admin and t_admin is not None:
        with t_admin:
            render_admin_console()

    # ── Footer ───────────────────────────────────
    st.markdown("""<div style="text-align:center;margin-top:3rem;color:#2a3550;
        font-family:'Share Tech Mono',monospace;font-size:.7rem;letter-spacing:.1em;">
        SIM SWAP FRAUD DETECTION SYSTEM &nbsp;|&nbsp; SOC-GRADE ML PIPELINE &nbsp;|&nbsp; v4.0
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════
if not st.session_state.authenticated:
    if st.session_state.auth_mode == "signup":
        page_signup()
    else:
        page_login()
else:
    page_main()