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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from stations import STATIONS, RIVER_COLORS, RISK_COLORS
from models import build_ffnn, predict_flood_risk, federated_round, _synthetic_data

# ────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedFlood Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }

    /* ── Glass cards ─────────────────────────── */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }

    /* ── Station card grid ───────────────────── */
    .station-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
    }
    .station-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }

    /* ── Risk badges ─────────────────────────── */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .risk-low      { background: rgba(76,175,80,0.2); color: #4CAF50; border: 1px solid rgba(76,175,80,0.3); }
    .risk-moderate { background: rgba(255,193,7,0.2); color: #FFC107; border: 1px solid rgba(255,193,7,0.3); }
    .risk-high     { background: rgba(255,152,0,0.2); color: #FF9800; border: 1px solid rgba(255,152,0,0.3); }
    .risk-critical { background: rgba(244,67,54,0.2); color: #F44336; border: 1px solid rgba(244,67,54,0.3); animation: pulse-critical 1.5s infinite; }

    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244,67,54,0.4); }
        50%      { box-shadow: 0 0 12px 4px rgba(244,67,54,0.15); }
    }

    /* ── Danger alert ────────────────────────── */
    .danger-alert {
        background: linear-gradient(90deg, rgba(244,67,54,0.15), rgba(244,67,54,0.05));
        border: 1px solid rgba(244,67,54,0.35);
        border-left: 4px solid #F44336;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
        animation: pulse-danger 2s infinite;
    }
    @keyframes pulse-danger {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.8; }
    }

    /* ── Metric boxes ────────────────────────── */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Headings ─────────────────────────────── */
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 28px;
    }

    /* ── FL status dots ──────────────────────── */
    .fl-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: fl-blink 1s infinite alternate;
    }
    @keyframes fl-blink {
        from { opacity: 0.4; } to { opacity: 1; }
    }

    /* ── Privacy banner ──────────────────────── */
    .privacy-banner {
        background: linear-gradient(90deg, rgba(34,197,94,0.12), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.3);
        border-left: 4px solid #22C55E;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 18px;
    }

    /* ── Hide Streamlit branding ─────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* River tags */
    .river-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Plotly dark template
# ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
)


# ────────────────────────────────────────────────────────────────
# Sidebar navigation
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:24px;">
        <div style="font-size:2.8rem;">🌊</div>
        <div style="font-size:1.4rem; font-weight:800;
             background:linear-gradient(135deg,#818cf8,#6366f1);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            FedFlood
        </div>
        <div style="font-size:0.72rem; color:#64748b; letter-spacing:1.5px; text-transform:uppercase;">
            Flood Forecasting Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📡 Station Monitor", "⚡ Risk Predictor", "🔗 FL Simulator"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding:12px; background:rgba(30,41,59,0.5); border-radius:10px;
                border:1px solid rgba(99,102,241,0.12); font-size:0.75rem; color:#94a3b8;">
        <b style="color:#818cf8;">Paper Metrics</b><br>
        Accuracy: <b style="color:#4ade80;">84%</b><br>
        R²: <b style="color:#4ade80;">0.99</b><br>
        RMSE: <b style="color:#4ade80;">0.2 – 0.5</b><br>
        Lead Time: <b style="color:#4ade80;">5 days</b><br>
        Tarbela Danger: <b style="color:#f87171;">650k cusec</b>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  📡  STATION MONITOR
# ════════════════════════════════════════════════════════════════
if page == "📡 Station Monitor":

    st.markdown('<div class="page-title">📡 Station Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time monitoring of all 18 client stations across the Indus Basin</div>', unsafe_allow_html=True)

    # ── Danger Alerts ─────────────────────────────────────────
    danger_stations = [s for s in STATIONS if s["danger_alert"]]
    if danger_stations:
        for s in danger_stations:
            st.markdown(f"""<div class="danger-alert">
⚠️ <b style="color:#F44336">DANGER ALERT</b> — <b>{s['name']}</b> ({s['river']}) inflow at <b>{s['inflow_cusec']:,} cusecs</b> (threshold: {s['danger_threshold_cusec']:,} cusecs)
</div>""", unsafe_allow_html=True)

    # ── Summary metrics row ───────────────────────────────────
    total = len(STATIONS)
    crit = sum(1 for s in STATIONS if s["risk_level"] == "critical")
    high = sum(1 for s in STATIONS if s["risk_level"] == "high")
    avg_level = np.mean([s["water_level_m"] for s in STATIONS])

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, str(total), "Total Stations"),
        (c2, str(crit), "Critical"),
        (c3, str(high), "High Risk"),
        (c4, f"{avg_level:.1f} m", "Avg Water Level"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── River filter ──────────────────────────────────────────
    rivers = sorted(set(s["river"] for s in STATIONS))
    selected_river = st.selectbox("Filter by River", ["All Rivers"] + rivers)
    filtered = STATIONS if selected_river == "All Rivers" else [s for s in STATIONS if s["river"] == selected_river]

    # ── Station cards (3-column grid) ─────────────────────────
    cols = st.columns(3)
    for i, s in enumerate(filtered):
        with cols[i % 3]:
            rc = RIVER_COLORS[s["river"]]
            risk_cls = f"risk-{s['risk_level']}"
            pct = min(s["inflow_cusec"] / s["danger_threshold_cusec"], 1.0)
            bar_color = RISK_COLORS[s["risk_level"]]

            card_html = f"""<div class="station-card" style="border-top: 3px solid {rc};">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<div>
<span style="font-weight:700; font-size:1.05rem; color:#f1f5f9;">{s['name']}</span><br>
<span class="river-tag" style="background:rgba({int(rc[1:3],16)},{int(rc[3:5],16)},{int(rc[5:7],16)},0.15); color:{rc};">{s['river']}</span>
</div>
<span class="risk-badge {risk_cls}">{s['risk_level']}</span>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.8rem; color:#cbd5e1; margin-bottom:10px;">
<div>Inflow: <b style="color:#f1f5f9">{s['inflow_cusec']:,}</b> cusec</div>
<div>Outflow: <b style="color:#f1f5f9">{s['outflow_cusec']:,}</b> cusec</div>
<div>Water Level: <b style="color:#f1f5f9">{s['water_level_m']}</b> m</div>
<div>Precip: <b style="color:#f1f5f9">{s['precip_mm']}</b> mm</div>
</div>
<div style="background:rgba(255,255,255,0.06); border-radius:6px; height:8px; overflow:hidden;">
<div style="width:{pct*100:.0f}%; height:100%; background:{bar_color}; border-radius:6px; transition: width 0.5s;"></div>
</div>
<div style="font-size:0.68rem; color:#64748b; margin-top:3px; text-align:right;">
{s['inflow_cusec']:,} / {s['danger_threshold_cusec']:,} cusec
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    # 5-Day Water Level Forecast Chart
    with ch1:
        st.markdown("#### 📈 5-Day Water Level Forecast")
        selected_station = st.selectbox("Select Station", [s["name"] for s in STATIONS], key="forecast_sel")
        sdata = next(s for s in STATIONS if s["name"] == selected_station)
        days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days, y=sdata["forecast_5day"],
            mode="lines+markers",
            line=dict(color=RIVER_COLORS[sdata["river"]], width=3),
            marker=dict(size=10, symbol="circle",
                        line=dict(width=2, color="#0f172a")),
            fill="tozeroy",
            fillcolor=f"rgba({int(RIVER_COLORS[sdata['river']][1:3],16)},{int(RIVER_COLORS[sdata['river']][3:5],16)},{int(RIVER_COLORS[sdata['river']][5:7],16)},0.12)",
            name="Forecast",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            yaxis_title="Water Level (m)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precipitation Efficiency Chart
    with ch2:
        st.markdown("#### 🌧️ Precipitation Efficiency by Station")
        display_stations = filtered
        eff_fig = go.Figure()
        eff_fig.add_trace(go.Bar(
            x=[s["name"] for s in display_stations],
            y=[s["precip_efficiency"] * 100 for s in display_stations],
            marker=dict(
                color=[RIVER_COLORS[s["river"]] for s in display_stations],
                line=dict(width=0),
                opacity=0.85,
            ),
            text=[f"{s['precip_efficiency']*100:.0f}%" for s in display_stations],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        eff_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            yaxis_title="Efficiency (%)",
            yaxis_range=[0, 100],
            showlegend=False,
        )
        eff_fig.update_xaxes(tickangle=-45)
        st.plotly_chart(eff_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  ⚡  RISK PREDICTOR
# ════════════════════════════════════════════════════════════════
elif page == "⚡ Risk Predictor":

    st.markdown('<div class="page-title">⚡ Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Simplified FFNN simulation — adjust inputs and observe flood probability</div>', unsafe_allow_html=True)

    # Build or cache the model
    @st.cache_resource
    def get_model():
        m = build_ffnn()
        # Quick pre-train so weights are meaningful
        rng = np.random.RandomState(42)
        X = rng.rand(500, 6).astype(np.float32)
        y = ((X @ [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]) > 0.40).astype(np.float32)
        m.fit(X, y, epochs=15, batch_size=32, verbose=0)
        return m

    model = get_model()

    # ── 6 Input Sliders ──────────────────────────────────────
    st.markdown("### 🎛️ Input Parameters")
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

    # Normalise to [0, 1]
    inputs = np.array([[
        rainfall / 300.0,
        water_level / 500.0,
        snow_melt / 150.0,
        upstream / 800.0,
        duration / 168.0,
        hydro_idx,
    ]], dtype=np.float32)

    result = predict_flood_risk(model, inputs)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flood Probability Gauge ───────────────────────────────
    prob = result["probability"]
    prob_color = "#4CAF50" if prob < 30 else "#FFC107" if prob < 60 else "#FF9800" if prob < 80 else "#F44336"

    p1, p2 = st.columns([1, 2])
    with p1:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
<div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;">Flood Probability</div>
<div style="font-size:4rem; font-weight:800; color:{prob_color}; text-shadow: 0 0 30px {prob_color}40;">{prob:.1f}%</div>
<div style="background:rgba(255,255,255,0.06); border-radius:6px; height:10px; overflow:hidden; margin-top:14px;">
<div style="width:{prob}%; height:100%; background:linear-gradient(90deg,#4CAF50,#FFC107,#F44336); border-radius:6px;"></div>
</div>
</div>""", unsafe_allow_html=True)

    # ── 5-Day Forecast Bars ───────────────────────────────────
    with p2:
        st.markdown("#### 📊 5-Day Water Level Forecast")
        fc_fig = go.Figure()
        fc_fig.add_trace(go.Bar(
            x=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
            y=result["forecast_5day"],
            marker=dict(
                color=["#818cf8", "#6366f1", "#4f46e5", "#4338ca", "#3730a3"],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in result["forecast_5day"]],
            textposition="outside",
            textfont=dict(color="#94a3b8"),
        ))
        fc_fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                             yaxis_title="Water Level (m)")
        st.plotly_chart(fc_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnostics ───────────────────────────────────────────
    st.markdown("### 📋 Model Diagnostics")
    d1, d2, d3 = st.columns(3)
    for col, val, label, fmt in [
        (d1, result["rmse"], "RMSE", f"{result['rmse']:.3f}"),
        (d2, result["r2"], "R² Score", f"{result['r2']:.4f}"),
        (d3, result["accuracy"], "Accuracy", f"{result['accuracy']:.1f}%"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-value">{fmt}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sub-Model Contributions ───────────────────────────────
    st.markdown("### 🧩 Sub-Model Contributions")
    contribs = result["contributions"]
    sub_colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"]
    contrib_fig = go.Figure()
    contrib_fig.add_trace(go.Bar(
        x=list(contribs.keys()),
        y=[v * 100 for v in contribs.values()],
        marker=dict(color=sub_colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in contribs.values()],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    contrib_fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              yaxis_title="Contribution (%)", yaxis_range=[0, 60])
    st.plotly_chart(contrib_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  🔗  FL SIMULATOR
# ════════════════════════════════════════════════════════════════
elif page == "🔗 FL Simulator":

    st.markdown('<div class="page-title">🔗 Federated Learning Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">FedFlood algorithm — privacy-preserving training across 18 clients</div>', unsafe_allow_html=True)

    # Privacy Banner
    st.markdown("""
    <div class="privacy-banner">
        🔒 <b style="color:#22C55E;">Privacy-Preserving</b>
        <span style="color:#94a3b8;"> — Only model weights are exchanged between clients and server.
        Raw sensor data never leaves the local station. This is the core principle of Federated Learning.</span>
    </div>
    """, unsafe_allow_html=True)

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        n_rounds = st.slider("Training Rounds", 3, 25, 10, 1)
    with ctrl2:
        n_clients = st.slider("Active Clients", 6, 18, 18, 1)
    with ctrl3:
        local_epochs = st.slider("Local Epochs per Round", 1, 5, 2, 1)

    run_btn = st.button("🚀 Start Federated Training", use_container_width=True, type="primary")

    # Placeholders for live updates
    status_placeholder = st.empty()
    metrics_cols = st.columns(4)
    metric_phs = [c.empty() for c in metrics_cols]
    chart_col1, chart_col2 = st.columns(2)
    loss_chart_ph = chart_col1.empty()
    acc_ring_ph = chart_col2.empty()
    log_ph = st.empty()
    transfer_ph = st.empty()

    # ── Initialise / retrieve session state ───────────────────
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
            status_placeholder.markdown(f'<div class="glass-card" style="text-align:center;"><div class="fl-dot" style="background:#818cf8;"></div><span style="color:#e2e8f0; font-weight:600;">Round {rnd}/{n_rounds} — Training {n_clients} clients (local epochs: {local_epochs})</span></div>', unsafe_allow_html=True)

            result = federated_round(global_model, client_ids, local_epochs=local_epochs)
            X_test, y_test = _synthetic_data(0, n=200)
            _, acc = global_model.evaluate(X_test, y_test, verbose=0)
            acc_pct = round(float(acc) * 100, 1)
            total_kb += result["weights_transferred_kb"] * n_clients

            rnd_data = {
                "round": rnd,
                "avg_loss": result["avg_loss"],
                "client_losses": result["client_losses"],
                "accuracy": acc_pct,
                "weights_kb": result["weights_transferred_kb"],
                "total_kb": round(total_kb, 1),
            }
            st.session_state.fl_results.append(rnd_data)

            # Log entries
            st.session_state.fl_log.append(
                f"[Round {rnd:02d}] Avg Loss: {result['avg_loss']:.4f} | "
                f"Accuracy: {acc_pct:.1f}% | "
                f"Weights: {result['weights_transferred_kb']:.1f} KB → server"
            )
            for ci, cl in enumerate(result["client_losses"]):
                st.session_state.fl_log.append(
                    f"   └─ Client {ci+1:02d} ({station_names[ci]}) loss: {cl:.4f}"
                )

            # ── Update Metrics ────────────────────────────────
            metric_phs[0].markdown(f'<div class="metric-box"><div class="metric-value">{rnd}</div><div class="metric-label">Current Round</div></div>', unsafe_allow_html=True)

            metric_phs[1].markdown(f'<div class="metric-box"><div class="metric-value">{result["avg_loss"]:.4f}</div><div class="metric-label">Avg Loss</div></div>', unsafe_allow_html=True)

            metric_phs[2].markdown(f'<div class="metric-box"><div class="metric-value" style="background:linear-gradient(135deg,#4ade80,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{acc_pct:.1f}%</div><div class="metric-label">Global Accuracy</div></div>', unsafe_allow_html=True)

            metric_phs[3].markdown(f'<div class="metric-box"><div class="metric-value">{total_kb:.0f} KB</div><div class="metric-label">Total Transfer</div></div>', unsafe_allow_html=True)

            # ── Loss Chart ────────────────────────────────────
            rounds_so_far = [r["round"] for r in st.session_state.fl_results]
            losses_so_far = [r["avg_loss"] for r in st.session_state.fl_results]

            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=rounds_so_far, y=losses_so_far,
                mode="lines+markers",
                line=dict(color="#818cf8", width=3),
                marker=dict(size=8, color="#6366f1",
                            line=dict(width=2, color="#0f172a")),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.1)",
                name="Avg Loss",
            ))
            loss_fig.update_layout(
                **PLOTLY_LAYOUT, height=320,
                title=dict(text="Round-by-Round Loss", font=dict(size=14)),
                xaxis_title="Round",
                yaxis_title="Loss",
                showlegend=False,
            )
            loss_chart_ph.plotly_chart(loss_fig, use_container_width=True)

            # ── Accuracy Ring ─────────────────────────────────
            ring_fig = go.Figure()
            ring_fig.add_trace(go.Pie(
                values=[acc_pct, 100 - acc_pct],
                hole=0.78,
                marker=dict(colors=["#4ade80", "rgba(255,255,255,0.04)"],
                            line=dict(width=0)),
                textinfo="none",
                hoverinfo="skip",
                sort=False,
            ))
            ring_fig.update_layout(
                **PLOTLY_LAYOUT, height=320,
                title=dict(text="Global Accuracy", font=dict(size=14)),
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{acc_pct:.1f}%</b>",
                    font=dict(size=36, color="#4ade80", family="Inter"),
                    showarrow=False,
                )],
            )
            acc_ring_ph.plotly_chart(ring_fig, use_container_width=True)

            # ── Training Log ──────────────────────────────────
            log_text = "\n".join(st.session_state.fl_log[-40:])
            log_ph.code(log_text, language="log")

            # ── Network Transfer Tracker ──────────────────────
            transfer_items = ""
            for ci in range(min(n_clients, 18)):
                sname = station_names[ci] if ci < len(station_names) else f"Client {ci+1}"
                transfer_items += f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;padding:6px 12px;background:rgba(30,41,59,0.5);border-radius:8px;border:1px solid rgba(99,102,241,0.08);"><div class="fl-dot" style="background:#818cf8;"></div><span style="color:#94a3b8;font-size:0.78rem;min-width:140px;">{sname}</span><div style="flex:1;background:rgba(255,255,255,0.04);border-radius:4px;height:6px;"><div style="width:100%;height:100%;background:linear-gradient(90deg,#818cf8,#6366f1);border-radius:4px;"></div></div><span style="color:#818cf8;font-size:0.72rem;font-weight:600;">{result["weights_transferred_kb"]:.1f} KB</span><span style="color:#22C55E;font-size:0.68rem;">✓ weights only</span></div>'
            transfer_ph.markdown(f'<div class="glass-card"><div style="font-weight:600;color:#e2e8f0;margin-bottom:12px;">🔄 Network Transfer Tracker — Round {rnd}</div>{transfer_items}</div>', unsafe_allow_html=True)

            time.sleep(0.3)  # brief pause for animation effect

        # Final status
        status_placeholder.markdown(f'<div class="glass-card" style="text-align:center;border:1px solid rgba(34,197,94,0.3);">✅ <span style="color:#22C55E;font-weight:700;">Training Complete</span> <span style="color:#94a3b8;"> — {n_rounds} rounds, {n_clients} clients, final accuracy <b style="color:#4ade80;">{st.session_state.fl_results[-1]["accuracy"]:.1f}%</b></span></div>', unsafe_allow_html=True)

    elif st.session_state.fl_results:
        # Re-render last state if results exist from a previous run
        r = st.session_state.fl_results[-1]
        status_placeholder.markdown(f'<div class="glass-card" style="text-align:center;border:1px solid rgba(34,197,94,0.3);">✅ <span style="color:#22C55E;font-weight:700;">Training Complete</span> <span style="color:#94a3b8;"> — {r["round"]} rounds, final accuracy <b style="color:#4ade80;">{r["accuracy"]:.1f}%</b></span></div>', unsafe_allow_html=True)

        metric_phs[0].markdown(f'<div class="metric-box"><div class="metric-value">{r["round"]}</div><div class="metric-label">Rounds</div></div>', unsafe_allow_html=True)
        metric_phs[1].markdown(f'<div class="metric-box"><div class="metric-value">{r["avg_loss"]:.4f}</div><div class="metric-label">Final Loss</div></div>', unsafe_allow_html=True)
        metric_phs[2].markdown(f'<div class="metric-box"><div class="metric-value" style="background:linear-gradient(135deg,#4ade80,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{r["accuracy"]:.1f}%</div><div class="metric-label">Final Accuracy</div></div>', unsafe_allow_html=True)
        metric_phs[3].markdown(f'<div class="metric-box"><div class="metric-value">{r["total_kb"]:.0f} KB</div><div class="metric-label">Total Transfer</div></div>', unsafe_allow_html=True)

        # Rebuild charts
        rounds_all = [x["round"] for x in st.session_state.fl_results]
        losses_all = [x["avg_loss"] for x in st.session_state.fl_results]
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=rounds_all, y=losses_all, mode="lines+markers",
                                      line=dict(color="#818cf8", width=3),
                                      marker=dict(size=8, color="#6366f1",
                                                  line=dict(width=2, color="#0f172a")),
                                      fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"))
        loss_fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Round-by-Round Loss", font=dict(size=14)),
                               xaxis_title="Round", yaxis_title="Loss", showlegend=False)
        loss_chart_ph.plotly_chart(loss_fig, use_container_width=True)

        ring_fig = go.Figure()
        ring_fig.add_trace(go.Pie(values=[r["accuracy"], 100 - r["accuracy"]], hole=0.78,
                                  marker=dict(colors=["#4ade80", "rgba(255,255,255,0.04)"],
                                              line=dict(width=0)),
                                  textinfo="none", hoverinfo="skip", sort=False))
        ring_fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Global Accuracy", font=dict(size=14)),
                               showlegend=False,
                               annotations=[dict(text=f"<b>{r['accuracy']:.1f}%</b>",
                                                  font=dict(size=36, color="#4ade80", family="Inter"),
                                                  showarrow=False)])
        acc_ring_ph.plotly_chart(ring_fig, use_container_width=True)

        log_text = "\n".join(st.session_state.fl_log[-40:])
        log_ph.code(log_text, language="log")
