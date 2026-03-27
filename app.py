"""
app.py — FedFlood Dashboard
📡 Station Monitor  |  ⚡ Risk Predictor  |  🔗 FL Simulator
Built with Streamlit + TensorFlow, grounded in the FedFlood paper.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import atexit
_original_register = atexit.register
def _safe_register(func, *args, **kwargs):
    try:
        return _original_register(func, *args, **kwargs)
    except Exception:
        pass
atexit.register = _safe_register

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from stations import STATIONS, RIVER_COLORS, RISK_COLORS
from models import build_ffnn, predict_flood_risk, federated_round, _synthetic_data

# ────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedFlood Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# Premium CSS Design System
# ────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ───────────────────────── */
:root {
    --bg-primary: #060b18;
    --bg-secondary: #0c1225;
    --bg-card: rgba(15, 23, 42, 0.75);
    --bg-card-hover: rgba(22, 33, 56, 0.85);
    --border-subtle: rgba(99, 102, 241, 0.12);
    --border-glow: rgba(99, 102, 241, 0.35);
    --accent-indigo: #818cf8;
    --accent-violet: #a78bfa;
    --accent-blue: #60a5fa;
    --accent-emerald: #34d399;
    --accent-rose: #fb7185;
    --accent-amber: #fbbf24;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.08);
}

/* ── Global ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -20%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139,92,246,0.05) 0%, transparent 50%);
}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1225 0%, #111b33 40%, #0f1729 100%);
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 10px 16px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.08) !important;
    border-color: var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.4); }

/* ── Glass Cards ─────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-card);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-card), var(--shadow-glow);
    transform: translateY(-1px);
}

/* ── Station Cards ───────────────────────── */
.station-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow-card);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.station-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(99,102,241,0.03) 0%, transparent 50%);
    pointer-events: none;
    border-radius: var(--radius-lg);
}
.station-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-card), var(--shadow-glow);
    transform: translateY(-2px);
}

/* ── Risk Badges ─────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.risk-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.risk-low { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.risk-low::before { background: #34d399; box-shadow: 0 0 6px #34d399; }
.risk-moderate { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.risk-moderate::before { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }
.risk-high { background: rgba(251,146,60,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
.risk-high::before { background: #fb923c; box-shadow: 0 0 6px #fb923c; }
.risk-critical { background: rgba(251,113,133,0.15); color: #fb7185; border: 1px solid rgba(251,113,133,0.3); animation: pulse-critical 2s infinite; }
.risk-critical::before { background: #fb7185; box-shadow: 0 0 8px #fb7185; animation: dot-pulse 1s infinite alternate; }

@keyframes pulse-critical {
    0%, 100% { box-shadow: 0 0 0 0 rgba(251,113,133,0.3); }
    50% { box-shadow: 0 0 16px 4px rgba(251,113,133,0.12); }
}
@keyframes dot-pulse {
    from { opacity: 0.5; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1.2); }
}

/* ── Danger Alert ────────────────────────── */
.danger-alert {
    background: linear-gradient(90deg, rgba(251,113,133,0.1) 0%, rgba(251,113,133,0.03) 100%);
    border: 1px solid rgba(251,113,133,0.2);
    border-left: 4px solid #fb7185;
    border-radius: var(--radius-md);
    padding: 14px 20px;
    margin-bottom: 14px;
    color: var(--text-primary);
    font-size: 0.88rem;
    animation: alert-pulse 3s infinite;
}
@keyframes alert-pulse {
    0%, 100% { border-left-color: #fb7185; }
    50% { border-left-color: #f43f5e; }
}

/* ── Metric Boxes ────────────────────────── */
.metric-box {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 22px 16px;
    text-align: center;
    box-shadow: var(--shadow-card);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-indigo), var(--accent-violet));
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-box:hover::before { opacity: 1; }
.metric-box:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Page Headings ───────────────────────── */
.page-header {
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border-subtle);
}
.page-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c7d2fe 0%, #818cf8 30%, #6366f1 60%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── River Tags ──────────────────────────── */
.river-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Privacy Banner ──────────────────────── */
.privacy-banner {
    background: linear-gradient(90deg, rgba(52,211,153,0.08) 0%, rgba(52,211,153,0.02) 100%);
    border: 1px solid rgba(52,211,153,0.2);
    border-left: 4px solid var(--accent-emerald);
    border-radius: var(--radius-md);
    padding: 16px 24px;
    margin-bottom: 20px;
    color: var(--text-secondary);
    font-size: 0.88rem;
}

/* ── FL Dots ─────────────────────────────── */
.fl-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: fl-blink 1.2s ease-in-out infinite alternate;
}
@keyframes fl-blink {
    from { opacity: 0.3; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1.1); }
}

/* ── Section Headers ─────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-subtle), transparent);
}

/* ── Data Grid ───────────────────────────── */
.data-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.82rem;
}
.data-label { color: var(--text-muted); }
.data-value { color: var(--text-primary); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

/* ── Risk Bar ────────────────────────────── */
.risk-bar-bg {
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
    margin-top: 12px;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.risk-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 30px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2));
    border-radius: 0 6px 6px 0;
}

/* ── Prob gauge ───────────────────────────── */
.prob-gauge {
    position: relative;
    text-align: center;
    padding: 30px 20px;
}
.prob-number {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -2px;
    font-family: 'Inter', sans-serif;
}
.prob-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 12px;
}

/* ── Streamlit overrides ─────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stSelectbox > div > div { border-color: var(--border-subtle) !important; background: var(--bg-card) !important; }
.stSlider > div > div > div { color: var(--text-secondary) !important; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}
div.stCodeBlock { border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-md) !important; }

/* ── Animations ──────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
</style>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Plotly premium dark template
# ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=48, r=16, t=48, b=48),
    xaxis=dict(
        gridcolor="rgba(148,163,184,0.06)",
        zerolinecolor="rgba(148,163,184,0.06)",
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(148,163,184,0.06)",
        zerolinecolor="rgba(148,163,184,0.06)",
        tickfont=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="#334155",
        font=dict(family="Inter", color="#f1f5f9", size=13),
    ),
)


# ────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;margin-bottom:28px;padding-top:8px;">
<div style="font-size:3rem;margin-bottom:4px;filter:drop-shadow(0 0 12px rgba(99,102,241,0.3));">🌊</div>
<div style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#c7d2fe,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px;">FedFlood</div>
<div style="font-size:0.65rem;color:#64748b;letter-spacing:2.5px;text-transform:uppercase;margin-top:4px;">Flood Forecasting System</div>
</div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📡 Station Monitor", "⚡ Risk Predictor", "🔗 FL Simulator"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Paper metrics panel
    st.markdown("""<div style="padding:16px;background:rgba(15,23,42,0.6);border-radius:12px;border:1px solid rgba(99,102,241,0.1);">
<div style="font-size:0.68rem;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">📄 Paper Ground Truth</div>
<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">
<span style="color:#64748b;">Accuracy</span><span style="color:#34d399;font-weight:700;font-family:'JetBrains Mono',monospace;">84%</span></div>
<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">
<span style="color:#64748b;">R² Score</span><span style="color:#34d399;font-weight:700;font-family:'JetBrains Mono',monospace;">0.99</span></div>
<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">
<span style="color:#64748b;">RMSE</span><span style="color:#34d399;font-weight:700;font-family:'JetBrains Mono',monospace;">0.2 – 0.5</span></div>
<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.8rem;">
<span style="color:#64748b;">Lead Time</span><span style="color:#34d399;font-weight:700;font-family:'JetBrains Mono',monospace;">5 days</span></div>
<div style="display:flex;justify-content:space-between;padding:6px 0;font-size:0.8rem;">
<span style="color:#64748b;">Flood Danger</span><span style="color:#fb7185;font-weight:700;font-family:'JetBrains Mono',monospace;">1.6M cusec</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""<div style="text-align:center;font-size:0.62rem;color:#475569;letter-spacing:1px;">
POWERED BY TENSORFLOW + FEDERATED LEARNING
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  📡  STATION MONITOR
# ════════════════════════════════════════════════════════════════
if page == "📡 Station Monitor":

    st.markdown("""<div class="page-header">
<div class="page-title">📡 Station Monitor</div>
<div class="page-subtitle">Real-time monitoring of all 18 client stations across India's major river basins — Ganga, Brahmaputra, Yamuna, Godavari & Mahanadi</div>
</div>""", unsafe_allow_html=True)

    # ── Danger Alerts ─────────────────────────────────────────
    danger_stations = [s for s in STATIONS if s["danger_alert"]]
    for s in danger_stations:
        st.markdown(f"""<div class="danger-alert">
⚠️ <b style="color:#fb7185">DANGER ALERT</b> — <b style="color:#f1f5f9">{s['name']}</b> ({s['river']}) inflow at <b style="color:#fb7185">{s['inflow_cusec']:,} cusecs</b> · threshold: {s['danger_threshold_cusec']:,} cusecs
</div>""", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────
    total = len(STATIONS)
    crit = sum(1 for s in STATIONS if s["risk_level"] == "critical")
    high = sum(1 for s in STATIONS if s["risk_level"] == "high")
    avg_level = np.mean([s["water_level_m"] for s in STATIONS])

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl, icon in [
        (c1, str(total), "Total Stations", "🏢"),
        (c2, str(crit), "Critical", "🔴"),
        (c3, str(high), "High Risk", "🟠"),
        (c4, f"{avg_level:.1f}m", "Avg Water Level", "📊"),
    ]:
        col.markdown(f'<div class="metric-box"><div style="font-size:1.2rem;margin-bottom:6px;">{icon}</div><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── River filter ──────────────────────────────────────────
    rivers = sorted(set(s["river"] for s in STATIONS))
    selected_river = st.selectbox("🏞️ Filter by River", ["All Rivers"] + rivers)
    filtered = STATIONS if selected_river == "All Rivers" else [s for s in STATIONS if s["river"] == selected_river]

    # ── Station cards ─────────────────────────────────────────
    cols = st.columns(3)
    for i, s in enumerate(filtered):
        with cols[i % 3]:
            rc = RIVER_COLORS[s["river"]]
            risk_cls = f"risk-{s['risk_level']}"
            pct = min(s["inflow_cusec"] / s["danger_threshold_cusec"], 1.0)
            bar_color = RISK_COLORS[s["risk_level"]]
            r_int = int(rc[1:3], 16)
            g_int = int(rc[3:5], 16)
            b_int = int(rc[5:7], 16)

            st.markdown(f"""<div class="station-card" style="border-top:3px solid {rc};">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
<div>
<div style="font-weight:700;font-size:1.05rem;color:#f1f5f9;margin-bottom:6px;">{s['name']}</div>
<span class="river-tag" style="background:rgba({r_int},{g_int},{b_int},0.12);color:{rc};">{s['river']}</span>
</div>
<span class="risk-badge {risk_cls}">{s['risk_level']}</span>
</div>
<div class="data-row"><span class="data-label">Inflow</span><span class="data-value">{s['inflow_cusec']:,} cusec</span></div>
<div class="data-row"><span class="data-label">Outflow</span><span class="data-value">{s['outflow_cusec']:,} cusec</span></div>
<div class="data-row"><span class="data-label">Water Level</span><span class="data-value">{s['water_level_m']} m</span></div>
<div class="data-row" style="border:none;"><span class="data-label">Precipitation</span><span class="data-value">{s['precip_mm']} mm</span></div>
<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{pct*100:.0f}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});"></div></div>
<div style="display:flex;justify-content:space-between;margin-top:6px;font-size:0.65rem;color:#475569;">
<span>0</span>
<span style="font-family:'JetBrains Mono',monospace;">{s['inflow_cusec']:,} / {s['danger_threshold_cusec']:,}</span>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-header">📈 5-Day Water Level Forecast</div>', unsafe_allow_html=True)
        selected_station = st.selectbox("Select Station", [s["name"] for s in STATIONS], key="forecast_sel")
        sdata = next(s for s in STATIONS if s["name"] == selected_station)
        days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
        rc = RIVER_COLORS[sdata["river"]]
        r_int, g_int, b_int = int(rc[1:3], 16), int(rc[3:5], 16), int(rc[5:7], 16)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=sdata["forecast_5day"],
            mode="lines+markers",
            line=dict(color=rc, width=3, shape="spline"),
            marker=dict(size=10, color=rc, line=dict(width=3, color="#0f172a"), symbol="circle"),
            fill="tozeroy",
            fillcolor=f"rgba({r_int},{g_int},{b_int},0.08)",
            hovertemplate="<b>%{x}</b><br>Water Level: %{y:.1f} m<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                          yaxis_title="Water Level (m)",
                          title=dict(text=f"{sdata['name']} — {sdata['river']}", font=dict(size=13, color="#64748b")))
        st.plotly_chart(fig, use_container_width=True, key="forecast_line")

    with ch2:
        st.markdown('<div class="section-header">🌧️ Precipitation Efficiency</div>', unsafe_allow_html=True)
        eff_fig = go.Figure()
        eff_fig.add_trace(go.Bar(
            x=[s["name"] for s in filtered],
            y=[s["precip_efficiency"] * 100 for s in filtered],
            marker=dict(
                color=[RIVER_COLORS[s["river"]] for s in filtered],
                line=dict(width=0),
                opacity=0.85,
            ),
            text=[f"{s['precip_efficiency']*100:.0f}%" for s in filtered],
            textposition="outside",
            textfont=dict(size=9, color="#64748b"),
            hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.1f}%<extra></extra>",
        ))
        eff_fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Efficiency (%)",
                              yaxis_range=[0, 100], showlegend=False)
        eff_fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        st.plotly_chart(eff_fig, use_container_width=True, key="precip_eff")


# ════════════════════════════════════════════════════════════════
#  ⚡  RISK PREDICTOR
# ════════════════════════════════════════════════════════════════
elif page == "⚡ Risk Predictor":

    st.markdown("""<div class="page-header">
<div class="page-title">⚡ Risk Predictor</div>
<div class="page-subtitle">Simplified FFNN simulation — adjust hydrological inputs and observe real-time flood probability estimation</div>
</div>""", unsafe_allow_html=True)

    @st.cache_resource
    def get_model():
        m = build_ffnn()
        rng = np.random.RandomState(42)
        X = rng.rand(500, 6).astype(np.float32)
        y = ((X @ [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]) > 0.40).astype(np.float32)
        m.fit(X, y, epochs=15, batch_size=32, verbose=0)
        return m

    model = get_model()

    # ── Sliders ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🎛️ Input Parameters</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 300.0, 120.0, 1.0)
        water_level = st.slider("🌊 Water Level (m)", 0.0, 500.0, 200.0, 1.0)
    with sc2:
        snow_melt = st.slider("❄️ Snow Melt (mm)", 0.0, 150.0, 50.0, 1.0)
        upstream = st.slider("⬆️ Upstream Inflow (k cusec)", 0.0, 800.0, 350.0, 5.0)
    with sc3:
        duration = st.slider("⏱️ Duration (hours)", 1.0, 168.0, 48.0, 1.0)
        hydro_idx = st.slider("💧 Hydrodynamic Index", 0.0, 1.0, 0.5, 0.01)

    inputs = np.array([[
        rainfall / 300.0, water_level / 500.0, snow_melt / 150.0,
        upstream / 800.0, duration / 168.0, hydro_idx,
    ]], dtype=np.float32)

    result = predict_flood_risk(model, inputs)
    st.markdown("")

    # ── Probability + Forecast ────────────────────────────────
    prob = result["probability"]
    if prob < 25:
        prob_color, prob_glow = "#34d399", "rgba(52,211,153,0.3)"
    elif prob < 50:
        prob_color, prob_glow = "#fbbf24", "rgba(251,191,36,0.3)"
    elif prob < 75:
        prob_color, prob_glow = "#fb923c", "rgba(251,146,60,0.3)"
    else:
        prob_color, prob_glow = "#fb7185", "rgba(251,113,133,0.3)"

    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown(f"""<div class="glass-card prob-gauge">
<div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">Flood Probability</div>
<div class="prob-number" style="color:{prob_color};text-shadow:0 0 40px {prob_glow};">{prob:.1f}<span style="font-size:1.8rem;font-weight:600;">%</span></div>
<div style="margin-top:20px;background:rgba(255,255,255,0.04);border-radius:8px;height:8px;overflow:hidden;">
<div style="width:{prob}%;height:100%;background:linear-gradient(90deg,#34d399,#fbbf24,#fb923c,#fb7185);border-radius:8px;transition:width 0.5s;"></div>
</div>
<div style="display:flex;justify-content:space-between;margin-top:6px;font-size:0.6rem;color:#475569;">
<span>Safe</span><span>Moderate</span><span>High</span><span>Critical</span>
</div>
</div>""", unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="section-header">📊 5-Day Water Level Forecast</div>', unsafe_allow_html=True)
        fc_fig = go.Figure()
        colors = ["#818cf8", "#6366f1", "#4f46e5", "#4338ca", "#3730a3"]
        fc_fig.add_trace(go.Bar(
            x=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
            y=result["forecast_5day"],
            marker=dict(color=colors, line=dict(width=0), opacity=0.9),
            text=[f"{v:.1f}m" for v in result["forecast_5day"]],
            textposition="outside",
            textfont=dict(color="#818cf8", size=12, family="JetBrains Mono"),
            hovertemplate="<b>%{x}</b><br>Level: %{y:.2f} m<extra></extra>",
        ))
        fc_fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                             yaxis_title="Water Level (m)")
        st.plotly_chart(fc_fig, use_container_width=True, key="risk_forecast")

    st.markdown("")

    # ── Diagnostics ───────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Model Diagnostics</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    diag_data = [
        (d1, f"{result['rmse']:.3f}", "RMSE", "#60a5fa", "📉"),
        (d2, f"{result['r2']:.4f}", "R² Score", "#34d399", "📈"),
        (d3, f"{result['accuracy']:.1f}%", "Accuracy", "#a78bfa", "🎯"),
    ]
    for col, val, label, color, icon in diag_data:
        col.markdown(f'<div class="metric-box"><div style="font-size:1rem;margin-bottom:6px;">{icon}</div><div class="metric-value" style="background:linear-gradient(135deg,{color},{color}dd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Contributions ─────────────────────────────────────────
    st.markdown('<div class="section-header">🧩 Sub-Model Contributions</div>', unsafe_allow_html=True)
    contribs = result["contributions"]
    sub_colors = ["#60a5fa", "#34d399", "#fbbf24", "#fb7185"]
    sub_icons = ["❄️", "🌧️", "🌊", "💧"]

    contrib_cols = st.columns(4)
    for idx, (name, val) in enumerate(contribs.items()):
        contrib_cols[idx].markdown(f"""<div class="glass-card" style="text-align:center;padding:20px 12px;">
<div style="font-size:1.5rem;margin-bottom:8px;">{sub_icons[idx]}</div>
<div style="font-size:1.6rem;font-weight:800;color:{sub_colors[idx]};font-family:'JetBrains Mono',monospace;">{val*100:.1f}%</div>
<div style="font-size:0.68rem;color:#64748b;margin-top:6px;text-transform:uppercase;letter-spacing:0.8px;">{name}</div>
<div style="margin-top:10px;background:rgba(255,255,255,0.04);border-radius:4px;height:4px;overflow:hidden;">
<div style="width:{val*100}%;height:100%;background:{sub_colors[idx]};border-radius:4px;"></div>
</div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  🔗  FL SIMULATOR
# ════════════════════════════════════════════════════════════════
elif page == "🔗 FL Simulator":

    st.markdown("""<div class="page-header">
<div class="page-title">🔗 Federated Learning Simulator</div>
<div class="page-subtitle">FedFlood algorithm — privacy-preserving distributed training across 18 client stations</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="privacy-banner">
🔒 <b style="color:#34d399;">Privacy-Preserving Protocol</b> — Only model weights are exchanged between clients and the aggregation server. Raw sensor data never leaves the local station. This is the core principle of Federated Learning.
</div>""", unsafe_allow_html=True)

    # Controls
    st.markdown('<div class="section-header">⚙️ Training Configuration</div>', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        n_rounds = st.slider("Training Rounds", 3, 25, 10, 1)
    with ctrl2:
        n_clients = st.slider("Active Clients", 6, 18, 18, 1)
    with ctrl3:
        local_epochs = st.slider("Local Epochs per Round", 1, 5, 2, 1)

    run_btn = st.button("🚀 Start Federated Training", use_container_width=True, type="primary")

    # Placeholders
    status_ph = st.empty()
    m_cols = st.columns(4)
    m_phs = [c.empty() for c in m_cols]
    ch1, ch2 = st.columns(2)
    loss_ph = ch1.empty()
    acc_ph = ch2.empty()
    log_ph = st.empty()
    transfer_ph = st.empty()

    if "fl_results" not in st.session_state:
        st.session_state.fl_results = []
    if "fl_log" not in st.session_state:
        st.session_state.fl_log = []

    if run_btn:
        st.session_state.fl_results = []
        st.session_state.fl_log = []
        global_model = build_ffnn()
        client_ids = list(range(1, n_clients + 1))
        station_names = [s["name"] for s in STATIONS[:n_clients]]
        total_kb = 0.0

        for rnd in range(1, n_rounds + 1):
            pct_done = rnd / n_rounds * 100
            status_ph.markdown(f'<div class="glass-card" style="text-align:center;padding:16px;"><div class="fl-dot" style="background:#818cf8;"></div><span style="color:#f1f5f9;font-weight:600;">Round {rnd}/{n_rounds}</span><span style="color:#64748b;"> — Training {n_clients} clients · {local_epochs} local epochs</span><div style="margin-top:10px;background:rgba(255,255,255,0.04);border-radius:6px;height:4px;overflow:hidden;"><div style="width:{pct_done}%;height:100%;background:linear-gradient(90deg,#818cf8,#6366f1);border-radius:6px;transition:width 0.3s;"></div></div></div>', unsafe_allow_html=True)

            result = federated_round(global_model, client_ids, local_epochs=local_epochs)
            X_test, y_test = _synthetic_data(0, n=200)
            _, acc = global_model.evaluate(X_test, y_test, verbose=0)
            acc_pct = round(float(acc) * 100, 1)
            total_kb += result["weights_transferred_kb"] * n_clients

            rnd_data = {"round": rnd, "avg_loss": result["avg_loss"], "client_losses": result["client_losses"],
                        "accuracy": acc_pct, "weights_kb": result["weights_transferred_kb"], "total_kb": round(total_kb, 1)}
            st.session_state.fl_results.append(rnd_data)

            st.session_state.fl_log.append(f"[Round {rnd:02d}] Avg Loss: {result['avg_loss']:.4f} | Accuracy: {acc_pct:.1f}% | Weights: {result['weights_transferred_kb']:.1f} KB → server")
            for ci, cl in enumerate(result["client_losses"]):
                st.session_state.fl_log.append(f"   └─ Client {ci+1:02d} ({station_names[ci]}) loss: {cl:.4f}")

            # Metrics
            m_phs[0].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">🔄</div><div class="metric-value">{rnd}</div><div class="metric-label">Current Round</div></div>', unsafe_allow_html=True)
            m_phs[1].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">📉</div><div class="metric-value" style="font-size:1.6rem;">{result["avg_loss"]:.4f}</div><div class="metric-label">Avg Loss</div></div>', unsafe_allow_html=True)
            m_phs[2].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">🎯</div><div class="metric-value" style="background:linear-gradient(135deg,#34d399,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{acc_pct:.1f}%</div><div class="metric-label">Global Accuracy</div></div>', unsafe_allow_html=True)
            m_phs[3].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">📡</div><div class="metric-value" style="font-size:1.6rem;">{total_kb:.0f}<span style="font-size:0.8rem;"> KB</span></div><div class="metric-label">Total Transfer</div></div>', unsafe_allow_html=True)

            # Loss Chart
            rs = [r["round"] for r in st.session_state.fl_results]
            ls = [r["avg_loss"] for r in st.session_state.fl_results]
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=rs, y=ls, mode="lines+markers",
                line=dict(color="#818cf8", width=3, shape="spline"),
                marker=dict(size=8, color="#6366f1", line=dict(width=2, color="#0f172a")),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                hovertemplate="Round %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ))
            loss_fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                                   title=dict(text="Round-by-Round Loss", font=dict(size=13, color="#64748b")),
                                   xaxis_title="Round", yaxis_title="Loss")
            loss_ph.plotly_chart(loss_fig, use_container_width=True, key=f"fl_loss_{rnd}")

            # Accuracy Ring
            ring_fig = go.Figure()
            ring_fig.add_trace(go.Pie(
                values=[acc_pct, 100 - acc_pct], hole=0.8,
                marker=dict(colors=["#34d399", "rgba(255,255,255,0.03)"], line=dict(width=0)),
                textinfo="none", hoverinfo="skip", sort=False,
                direction="clockwise", rotation=90,
            ))
            ring_fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                                   title=dict(text="Global Accuracy", font=dict(size=13, color="#64748b")),
                                   annotations=[dict(text=f"<b>{acc_pct:.1f}%</b>",
                                                      font=dict(size=32, color="#34d399", family="JetBrains Mono"),
                                                      showarrow=False)])
            acc_ph.plotly_chart(ring_fig, use_container_width=True, key=f"fl_acc_{rnd}")

            # Log
            log_ph.code("\n".join(st.session_state.fl_log[-30:]), language="log")

            # Transfer Tracker
            items = ""
            for ci in range(min(n_clients, 18)):
                sn = station_names[ci] if ci < len(station_names) else f"Client {ci+1}"
                items += f'<div style="display:flex;align-items:center;gap:8px;padding:7px 12px;margin-bottom:4px;background:rgba(15,23,42,0.5);border-radius:8px;border:1px solid rgba(99,102,241,0.06);"><div class="fl-dot" style="background:#818cf8;"></div><span style="color:#94a3b8;font-size:0.76rem;min-width:130px;">{sn}</span><div style="flex:1;background:rgba(255,255,255,0.03);border-radius:3px;height:4px;"><div style="width:100%;height:100%;background:linear-gradient(90deg,#818cf8,#6366f1);border-radius:3px;"></div></div><span style="color:#818cf8;font-size:0.7rem;font-weight:600;font-family:JetBrains Mono,monospace;">{result["weights_transferred_kb"]:.1f} KB</span><span style="color:#34d399;font-size:0.62rem;">✓</span></div>'
            transfer_ph.markdown(f'<div class="glass-card" style="padding:20px;"><div class="section-header">🔄 Network Transfer — Round {rnd}</div>{items}</div>', unsafe_allow_html=True)

            time.sleep(0.3)

        status_ph.markdown(f'<div class="glass-card" style="text-align:center;padding:16px;border:1px solid rgba(52,211,153,0.25);">✅ <span style="color:#34d399;font-weight:700;">Training Complete</span> <span style="color:#94a3b8;">— {n_rounds} rounds · {n_clients} clients · final accuracy <b style="color:#34d399;">{st.session_state.fl_results[-1]["accuracy"]:.1f}%</b></span></div>', unsafe_allow_html=True)

    elif st.session_state.fl_results:
        r = st.session_state.fl_results[-1]
        status_ph.markdown(f'<div class="glass-card" style="text-align:center;padding:16px;border:1px solid rgba(52,211,153,0.25);">✅ <span style="color:#34d399;font-weight:700;">Training Complete</span> <span style="color:#94a3b8;">— {r["round"]} rounds · final accuracy <b style="color:#34d399;">{r["accuracy"]:.1f}%</b></span></div>', unsafe_allow_html=True)

        m_phs[0].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">🔄</div><div class="metric-value">{r["round"]}</div><div class="metric-label">Rounds</div></div>', unsafe_allow_html=True)
        m_phs[1].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">📉</div><div class="metric-value" style="font-size:1.6rem;">{r["avg_loss"]:.4f}</div><div class="metric-label">Final Loss</div></div>', unsafe_allow_html=True)
        m_phs[2].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">🎯</div><div class="metric-value" style="background:linear-gradient(135deg,#34d399,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{r["accuracy"]:.1f}%</div><div class="metric-label">Final Accuracy</div></div>', unsafe_allow_html=True)
        m_phs[3].markdown(f'<div class="metric-box"><div style="font-size:0.9rem;margin-bottom:4px;">📡</div><div class="metric-value" style="font-size:1.6rem;">{r["total_kb"]:.0f}<span style="font-size:0.8rem;"> KB</span></div><div class="metric-label">Total Transfer</div></div>', unsafe_allow_html=True)

        rs = [x["round"] for x in st.session_state.fl_results]
        ls = [x["avg_loss"] for x in st.session_state.fl_results]
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=rs, y=ls, mode="lines+markers",
                                      line=dict(color="#818cf8", width=3, shape="spline"),
                                      marker=dict(size=8, color="#6366f1", line=dict(width=2, color="#0f172a")),
                                      fill="tozeroy", fillcolor="rgba(99,102,241,0.06)"))
        loss_fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                               title=dict(text="Round-by-Round Loss", font=dict(size=13, color="#64748b")),
                               xaxis_title="Round", yaxis_title="Loss")
        loss_ph.plotly_chart(loss_fig, use_container_width=True, key="fl_loss_final")

        ring_fig = go.Figure()
        ring_fig.add_trace(go.Pie(values=[r["accuracy"], 100 - r["accuracy"]], hole=0.8,
                                  marker=dict(colors=["#34d399", "rgba(255,255,255,0.03)"], line=dict(width=0)),
                                  textinfo="none", hoverinfo="skip", sort=False, direction="clockwise", rotation=90))
        ring_fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                               title=dict(text="Global Accuracy", font=dict(size=13, color="#64748b")),
                               annotations=[dict(text=f"<b>{r['accuracy']:.1f}%</b>",
                                                  font=dict(size=32, color="#34d399", family="JetBrains Mono"),
                                                  showarrow=False)])
        acc_ph.plotly_chart(ring_fig, use_container_width=True, key="fl_acc_final")

        log_ph.code("\n".join(st.session_state.fl_log[-30:]), language="log")
