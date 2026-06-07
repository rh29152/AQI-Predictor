"""
streamlit_app.py — Karachi AQI Predictor dashboard.

Pollutant-first forecast UI: current EPA AQI from the latest feature row,
24/48/72 h horizon cards, historical trends, and optional SHAP importances.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.database import get_latest_features, ensure_indexes
from src.predict import predict_all_horizons
from src.aqi_utils import calculate_final_aqi, aqi_category, EPA_AQI_CATEGORIES
from src.config import POLLUTANTS_TO_FORECAST

# ── Constants ──────────────────────────────────────────────────────────────────

POLL_LABEL  = {"pm2_5": "PM2.5", "pm10": "PM10", "o3": "O₃",  "no2": "NO₂"}
POLL_COLOR  = {"pm2_5": "#ff6b6b", "pm10": "#ffa94d", "o3": "#74c0fc", "no2": "#b197fc"}
POLL_UNIT   = {"pm2_5": "μg/m³",  "pm10": "μg/m³",  "o3": "μg/m³",  "no2": "μg/m³"}
POLL_DESC   = {
    "pm2_5": "Fine particles < 2.5 μm",
    "pm10":  "Coarse dust < 10 μm",
    "o3":    "Ground-level ozone",
    "no2":   "Nitrogen dioxide",
}

AQI_BG = {
    "Good":                          "#00c853",
    "Moderate":                      "#ffd600",
    "Unhealthy for Sensitive Groups": "#ff6d00",
    "Unhealthy":                     "#d50000",
    "Very Unhealthy":                "#6a1b9a",
    "Hazardous":                     "#4e0000",
}

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zeroline=False),
    margin=dict(l=10, r=10, t=45, b=10),
)


# ── Global CSS ─────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%) !important;
        min-height: 100vh;
    }
    section[data-testid="stSidebar"] { background: #0d1526 !important; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
    div[data-testid="stHorizontalBlock"] { gap: 1.2rem; }

    /* ── Plotly chart backgrounds ── */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Section divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.25), transparent);
        margin: 2rem 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #2d4a7a; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ── Reusable HTML components ───────────────────────────────────────────────────

def hero_header(now_str: str) -> None:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 60%, #162035 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.6rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top:-60px; right:-60px;
            width:220px; height:220px;
            background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:0.78rem; font-weight:600; letter-spacing:0.15em;
                            color:#63b3ed; text-transform:uppercase; margin-bottom:0.4rem;">
                    Karachi, Pakistan &nbsp;·&nbsp; Real-time Air Quality
                </div>
                <h1 style="margin:0; font-size:2rem; font-weight:800; color:#f0f4ff; line-height:1.2;">
                    AQI Predictor
                    <span style="font-size:1rem; font-weight:400; color:#8badc8; margin-left:0.6rem;">
                        Pollutant-First Forecasting
                    </span>
                </h1>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.72rem; color:#5a7fa0; margin-bottom:0.2rem;">Latest data</div>
                <div style="font-size:0.9rem; font-weight:600; color:#90cdf4;">{now_str}</div>
                <div style="font-size:0.68rem; color:#4a6a85; margin-top:0.2rem;">
                    EPA breakpoint interpolation · Not regulatory-grade
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def aqi_hero_card(aqi: int | None, category: str, color: str,
                  label: str, dominant: str, sub_label: str = "") -> str:
    """Large glassy AQI card."""
    aqi_text   = str(aqi) if aqi is not None else "—"
    dom_label  = POLL_LABEL.get(dominant, dominant) if dominant else "—"
    glow       = f"0 0 30px {color}44"
    bg         = f"linear-gradient(145deg, {color}18 0%, {color}08 100%)"
    ring_color = color if aqi is not None else "#2d4a7a"

    return f"""
    <div style="
        background: {bg};
        border: 1.5px solid {ring_color}55;
        border-radius: 18px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: {glow};
        transition: transform .2s;
        height: 100%;
    ">
        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em;
                    color:{ring_color}; text-transform:uppercase; margin-bottom:0.5rem;">
            {label}
        </div>
        <div style="font-size:3.4rem; font-weight:800; color:{ring_color};
                    line-height:1; margin-bottom:0.3rem; text-shadow:{glow};">
            {aqi_text}
        </div>
        <div style="font-size:0.8rem; font-weight:600; color:{ring_color};
                    margin-bottom:0.6rem; min-height:1.2rem;">
            {category}
        </div>
        <div style="border-top:1px solid {ring_color}22; padding-top:0.6rem; margin-top:0.2rem;">
            <div style="font-size:0.65rem; color:#6a8aaa; text-transform:uppercase;
                        letter-spacing:0.08em;">Dominant pollutant</div>
            <div style="font-size:0.85rem; font-weight:700; color:{ring_color}bb; margin-top:0.15rem;">
                {dom_label}
            </div>
        </div>
        {f'<div style="font-size:0.62rem; color:#4a6a85; margin-top:0.4rem;">{sub_label}</div>' if sub_label else ''}
    </div>"""


def pollutant_hz_header(hz_label: str, aqi: int | None, category: str, color: str) -> str:
    """Slim header row for a single forecast horizon."""
    return f"""
    <div style="
        background: linear-gradient(135deg, #111827 0%, #141e30 100%);
        border: 1px solid {color}30;
        border-top-left-radius: 14px;
        border-top-right-radius: 14px;
        padding: 0.6rem 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.8rem;
    ">
        <span style="font-size:0.72rem; font-weight:700; color:{color};
                     text-transform:uppercase; letter-spacing:0.1em;">+{hz_label} Forecast</span>
        <span style="font-size:1.4rem; font-weight:800; color:{color}; margin-right:0.4rem;">
            {aqi if aqi is not None else '—'}
        </span>
        <span style="font-size:0.72rem; font-weight:600; color:{color}aa;">{category}</span>
    </div>
    <div style="
        border-left: 1px solid {color}20;
        border-right: 1px solid {color}20;
        border-bottom: 1px solid {color}20;
        border-bottom-left-radius: 14px;
        border-bottom-right-radius: 14px;
        padding: 0.6rem 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
        background: rgba(10,14,26,0.4);
    ">"""


def pollutant_pill_html(poll: str, conc: float, sub: int) -> str:
    """A single pollutant pill meant for inside a Streamlit column."""
    pc = POLL_COLOR.get(poll, "#888")
    pl = POLL_LABEL.get(poll, poll)
    return f"""
    <div style="
        background: {pc}18;
        border: 1px solid {pc}45;
        border-radius: 10px;
        padding: 0.6rem 0.4rem;
        text-align: center;
    ">
        <div style="font-size:0.62rem; font-weight:700; color:{pc};
                    letter-spacing:0.07em; margin-bottom:0.1rem;">{pl}</div>
        <div style="font-size:1.1rem; font-weight:800; color:#e8f0f8; line-height:1.1;">{conc:.1f}</div>
        <div style="font-size:0.56rem; color:#4a6a85; margin-bottom:0.15rem;">ug/m3</div>
        <div style="font-size:0.58rem; color:{pc}bb;">sub {sub}</div>
    </div>"""


def alert_html(level: str, message: str) -> str:
    configs = {
        "hazardous":     ("#7e0023", "#ff1744", "🚨"),
        "very_unhealthy":("#4a0072", "#ce93d8", "⚠️"),
        "unhealthy":     ("#b71c1c", "#ff8a80", "⚠️"),
    }
    bg, fg, icon = configs.get(level, ("#1a2744", "#90cdf4", "ℹ️"))
    return f"""
    <div style="
        background: {bg}55;
        border-left: 4px solid {fg};
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    ">
        <span style="font-size:1.4rem;">{icon}</span>
        <span style="color:{fg}; font-size:0.88rem; font-weight:500;">{message}</span>
    </div>"""


def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin: 1.8rem 0 0.9rem;">
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; display:flex;
                    align-items:center; gap:0.5rem;">
            <span style="font-size:1.2rem;">{icon}</span> {title}
        </div>
        {f'<div style="font-size:0.75rem; color:#5a7fa0; margin-top:0.2rem;">{subtitle}</div>'
         if subtitle else ''}
    </div>""", unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_dashboard_data(hours: int = 168) -> tuple[pd.DataFrame, dict]:
    """
    Single cached load of features and live forecasts so both share the same
    latest MongoDB timestamp.
    """
    rows = get_latest_features(n=hours)
    if not rows:
        return pd.DataFrame(), {"error": "No feature data in MongoDB."}
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    try:
        forecasts = predict_all_horizons()
    except Exception as exc:
        forecasts = {"error": str(exc)}
    return df, forecasts


def forecast_target_str(latest_dt: pd.Timestamp, hz: str) -> str:
    """Horizon target timestamp anchored to the latest feature row."""
    hours = int(hz.replace("h", ""))
    return (latest_dt + pd.Timedelta(hours=hours)).strftime("%d %b %Y  %H:%M UTC")


# ── Plotly chart builders ──────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to Plotly-compatible rgba()."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def aqi_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Computed EPA AQI over time with coloured AQI band shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["computed_aqi"],
        mode="lines", name="AQI",
        line=dict(color="#63b3ed", width=2),
        fill="tozeroy", fillcolor="rgba(99,179,237,0.08)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y}<extra></extra>",
    ))
    # Rolling mean for trend smoothing
    roll = df.set_index("datetime")["computed_aqi"].rolling("7D").mean().reset_index()
    fig.add_trace(go.Scatter(
        x=roll["datetime"], y=roll["computed_aqi"],
        mode="lines", name="7-day avg",
        line=dict(color="#ffd700", width=1.5, dash="dot"),
        hovertemplate="7-day avg: %{y:.0f}<extra></extra>",
    ))
    for low, high, label, color in EPA_AQI_CATEGORIES:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.04, line_width=0)
    for _, high_val, label, color in EPA_AQI_CATEGORIES[1:]:
        fig.add_hline(y=high_val, line_dash="dot",
                      line_color=_rgba(color, 0.27), line_width=1)
    fig.update_layout(
        title=dict(text="Historical AQI Trend (EPA computed)", font=dict(size=13, color="#8badc8")),
        xaxis_title="", yaxis_title="EPA AQI",
        height=320, legend=dict(orientation="h", y=1.12, x=0),
        **CHART_TEMPLATE,
    )
    return fig


def pollutant_trend_chart(df: pd.DataFrame, polls: list[str]) -> go.Figure:
    """Stacked pollutant concentration time series."""
    rows = len(polls)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        subplot_titles=[f"{POLL_LABEL.get(p, p)} (μg/m³)" for p in polls],
        vertical_spacing=0.06,
    )
    for i, poll in enumerate(polls, 1):
        color = POLL_COLOR.get(poll, "#888")
        fig.add_trace(
            go.Scatter(
                x=df["datetime"], y=df[poll],
                mode="lines", name=POLL_LABEL.get(poll, poll),
                line=dict(color=color, width=1.4),
                fill="tozeroy", fillcolor=_rgba(color, 0.06),
                hovertemplate=f"{POLL_LABEL.get(poll, poll)}: %{{y:.1f}} μg/m³<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.update_yaxes(row=i, col=1, gridcolor="rgba(255,255,255,0.06)",
                         tickfont=dict(size=10, color="#8badc8"))
    fig.update_layout(
        height=max(180, 170 * rows),
        showlegend=False,
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#8badc8"))
    return fig


def sub_aqi_chart(forecasts: dict) -> go.Figure:
    """Grouped bar chart: sub-AQI per pollutant per horizon."""
    hz_colors = {"24h": "#63b3ed", "48h": "#ffa94d", "72h": "#ff6b6b"}
    fig = go.Figure()
    for hz_label, clr in hz_colors.items():
        info = forecasts.get(hz_label, {})
        sub  = info.get("sub_indices", {})
        fig.add_trace(go.Bar(
            name=f"+{hz_label}",
            x=[POLL_LABEL.get(p, p) for p in sub],
            y=list(sub.values()),
            marker_color=clr,
            marker_line_width=0,
            opacity=0.85,
            hovertemplate=f"+{hz_label}: %{{y}}<extra></extra>",
        ))
    fig.add_hline(y=150, line_dash="dash", line_color="#ffd700",
                  annotation_text="Unhealthy  ", annotation_font_color="#ffd700",
                  annotation_position="top left")
    fig.update_layout(
        barmode="group",
        title=dict(text="Sub-AQI Comparison — highest value drives final AQI",
                   font=dict(size=13, color="#8badc8")),
        yaxis_title="Sub-AQI",
        height=320,
        legend=dict(orientation="h", y=1.12, x=0),
        **CHART_TEMPLATE,
    )
    return fig


def forecast_timeline_chart(current_result: dict | None, forecasts: dict) -> go.Figure:
    """Horizontal-style AQI forecast bar (Now → +72h)."""
    labels, aqi_vals, colors = [], [], []
    if current_result and current_result.get("aqi") is not None:
        labels.append("Now")
        aqi_vals.append(current_result["aqi"])
        colors.append(current_result.get("color", "#63b3ed"))
    for hz in ["24h", "48h", "72h"]:
        info = forecasts.get(hz, {})
        if info.get("predicted_aqi") is not None:
            labels.append(f"+{hz}")
            aqi_vals.append(info["predicted_aqi"])
            colors.append(info.get("color", "#63b3ed"))

    fig = go.Figure(go.Bar(
        x=labels, y=aqi_vals,
        marker=dict(color=colors, line_width=0),
        text=[str(v) for v in aqi_vals],
        textposition="outside",
        textfont=dict(size=14, color="#e2e8f0"),
        hovertemplate="%{x}: AQI %{y}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color=_rgba("#ffd700", 0.33))
    fig.add_hline(y=150, line_dash="dash", line_color=_rgba("#ffa500", 0.4),
                  annotation_text="Unhealthy  ", annotation_font_color="#ffa500",
                  annotation_position="top left")
    max_y = max(aqi_vals + [200]) * 1.25
    fig.update_layout(
        title=dict(text="AQI Forecast: Now -> +72 hours",
                   font=dict(size=13, color="#8badc8")),
        height=300,
        **CHART_TEMPLATE,
    )
    fig.update_yaxes(range=[0, max_y], title="EPA AQI")
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    hours_window = st.slider("Historical window (hours)", 24, 720, 168, 24)
    st.caption(f"Displaying last **{hours_window}h** of data.")
    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("""
    **Data pipeline**
    - 🌐 OpenWeather API (hourly)
    - 🗄️ MongoDB Atlas (feature store)
    - 🤗 Hugging Face Hub (models)
    - ⚡ GitHub Actions (automation)
    """)
    st.divider()
    st.caption(
        "AQI uses EPA breakpoint interpolation on hourly forecasted PM2.5, "
        "PM10, O₃ & NO₂.  Educational approximation — not regulatory-grade."
    )


# ── Main dashboard ─────────────────────────────────────────────────────────────

inject_css()

now_str = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

try:
    ensure_indexes()
except Exception:
    pass

# Shared data load — features and forecasts from one cache key
df, forecasts = load_dashboard_data(hours=hours_window)

# Current AQI derived from the newest feature row pollutants
current_aqi_result: dict | None = None
latest_data_dt: pd.Timestamp | None = None
if not df.empty:
    latest_row = df.iloc[-1]
    latest_data_dt = pd.to_datetime(latest_row["datetime"], utc=True)
    current_pollutants = {p: float(latest_row.get(p, 0) or 0) for p in POLLUTANTS_TO_FORECAST}
    current_aqi_result = calculate_final_aqi(current_pollutants)
    # Add computed AQI column for trend chart
    computed = []
    for _, row in df.iterrows():
        concs = {p: float(row.get(p, 0) or 0) for p in POLLUTANTS_TO_FORECAST}
        computed.append(calculate_final_aqi(concs)["aqi"])
    df["computed_aqi"] = computed

data_as_of_str = (
    latest_data_dt.strftime("%d %b %Y  %H:%M UTC") if latest_data_dt is not None else now_str
)
hero_header(data_as_of_str)

# ── Alert banners ──────────────────────────────────────────────────────────────
if "error" in forecasts:
    st.markdown(
        alert_html("hazardous", f"Forecast pipeline error: {forecasts['error']}"),
        unsafe_allow_html=True,
    )
elif forecasts and all(
    isinstance(v, dict) and v.get("predicted_aqi") is None
    for v in forecasts.values()
):
    st.markdown(
        alert_html(
            "moderate",
            "Forecasts unavailable — models could not be loaded from Hugging Face. "
            "Confirm HF_TOKEN and HF_REPO_ID in Streamlit secrets (Manage app → Settings → Secrets), "
            "then reboot the app.",
        ),
        unsafe_allow_html=True,
    )
else:
    alert_shown = False
    for hz, info in forecasts.items():
        aqi_v = info.get("predicted_aqi")
        cat_v = info.get("aqi_category", "")
        if aqi_v is None:
            continue
        if aqi_v >= 301:
            st.markdown(alert_html("hazardous",
                f"HAZARDOUS air quality predicted at +{hz} (AQI {aqi_v}). "
                "Avoid all outdoor activity. N95 mask required outdoors."), unsafe_allow_html=True)
            alert_shown = True
        elif aqi_v >= 201:
            st.markdown(alert_html("very_unhealthy",
                f"Very Unhealthy air predicted at +{hz} (AQI {aqi_v}). "
                "Avoid prolonged outdoor exertion."), unsafe_allow_html=True)
            alert_shown = True
        elif aqi_v >= 151:
            st.markdown(alert_html("unhealthy",
                f"Unhealthy air predicted at +{hz} (AQI {aqi_v}). "
                "Sensitive groups should limit outdoor activity."), unsafe_allow_html=True)
            alert_shown = True

# ── AQI Summary Cards ──────────────────────────────────────────────────────────
section_header("🌡️", "Current & Forecasted Air Quality")

col_now, col_24, col_48, col_72 = st.columns(4, gap="medium")

def _get_now_card() -> str:
    if current_aqi_result and current_aqi_result.get("aqi") is not None:
        cat, clr = aqi_category(current_aqi_result["aqi"])
        return aqi_hero_card(
            current_aqi_result["aqi"], cat, clr,
            "Current AQI",
            current_aqi_result.get("dominant_pollutant", ""),
            f"As of {data_as_of_str}",
        )
    return aqi_hero_card(None, "No data", "#2d4a7a", "Current AQI", "", "")

with col_now:
    st.markdown(_get_now_card(), unsafe_allow_html=True)

for col, hz in [(col_24, "24h"), (col_48, "48h"), (col_72, "72h")]:
    with col:
        info = forecasts.get(hz, {}) if "error" not in forecasts else {}
        aqi  = info.get("predicted_aqi")
        cat  = info.get("aqi_category", "—")
        clr  = info.get("color", "#2d4a7a")
        dom  = info.get("dominant_pollutant", "")
        tgt_str = forecast_target_str(latest_data_dt, hz) if latest_data_dt is not None else ""
        st.markdown(
            aqi_hero_card(aqi, cat, clr, f"+{hz} Forecast", dom, tgt_str),
            unsafe_allow_html=True,
        )

divider()

# ── Pollutant Breakdown per Horizon ───────────────────────────────────────────
section_header("🧪", "Predicted Pollutant Concentrations",
               "Each pollutant is predicted independently; final AQI = max sub-index")

if "error" not in forecasts and forecasts:
    for hz in ["24h", "48h", "72h"]:
        info  = forecasts.get(hz, {})
        polls = info.get("predicted_pollutants", {})
        subs  = info.get("sub_indices", {})
        aqi_v = info.get("predicted_aqi")
        cat_v = info.get("aqi_category", "—")
        clr_v = info.get("color", "#2d4a7a")
        if not polls:
            continue
        # Header strip
        st.markdown(pollutant_hz_header(hz, aqi_v, cat_v, clr_v), unsafe_allow_html=True)
        # One Streamlit column per pollutant pill (avoids nested HTML rendering)
        pill_cols = st.columns(len(polls))
        for col, (poll, conc) in zip(pill_cols, polls.items()):
            with col:
                sub = subs.get(poll, 0)
                st.markdown(pollutant_pill_html(poll, conc, sub), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

divider()

# ── Forecast Timeline ─────────────────────────────────────────────────────────
if current_aqi_result and "error" not in forecasts:
    section_header("🔮", "AQI Forecast Timeline  (Now → +72 h)")
    fig_tl = forecast_timeline_chart(current_aqi_result, forecasts)
    st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})
    divider()

# ── Sub-AQI Comparison ────────────────────────────────────────────────────────
if "error" not in forecasts and forecasts:
    section_header("🎯", "Sub-AQI Breakdown",
                   "Which pollutant is driving the forecast AQI at each horizon?")
    col_sub, col_info = st.columns([3, 1], gap="large")
    with col_sub:
        st.plotly_chart(sub_aqi_chart(forecasts), use_container_width=True,
                        config={"displayModeBar": False})
    with col_info:
        st.markdown("""
        <div style="padding:1rem; background:rgba(255,255,255,0.03);
                    border-radius:12px; border:1px solid rgba(99,179,237,0.12);">
        <div style="font-size:0.72rem; font-weight:700; color:#63b3ed;
                    letter-spacing:0.1em; margin-bottom:0.8rem; text-transform:uppercase;">
            EPA AQI Scale
        </div>
        """, unsafe_allow_html=True)
        for low, high, label, color in EPA_AQI_CATEGORIES:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.5rem;
                        margin-bottom:0.45rem; font-size:0.76rem;">
                <div style="width:10px; height:10px; border-radius:50%;
                            background:{color}; flex-shrink:0;"></div>
                <span style="color:#8badc8;">{low}–{high}</span>
                <span style="color:#5a7fa0; font-size:0.68rem;">{label}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    divider()

# ── Historical AQI Trend ──────────────────────────────────────────────────────
if not df.empty and "computed_aqi" in df.columns:
    section_header("📈", "Historical AQI Trend",
                   f"Last {hours_window}h — computed from real pollutant measurements")
    st.plotly_chart(aqi_trend_chart(df), use_container_width=True,
                    config={"displayModeBar": False})
    divider()

# ── Pollutant Concentration Trends ────────────────────────────────────────────
if not df.empty:
    section_header("📊", "Pollutant Concentration History")
    avail = [p for p in ["pm2_5", "pm10", "o3", "no2"] if p in df.columns]
    sel = st.multiselect(
        "Select pollutants:",
        options=avail,
        default=avail,
        format_func=lambda x: POLL_LABEL.get(x, x),
        label_visibility="collapsed",
    )
    if sel:
        st.plotly_chart(pollutant_trend_chart(df, sel), use_container_width=True,
                        config={"displayModeBar": False})
    divider()

# ── SHAP Feature Importance ───────────────────────────────────────────────────
section_header("🔍", "Key Drivers of PM2.5 Forecast",
               "SHAP values — which features push the +24h PM2.5 prediction higher or lower")

@st.cache_data(ttl=1800)
def compute_shap() -> pd.DataFrame | None:
    try:
        import shap
        from src.model_registry import load_model
        model, meta = load_model(target="target_pm2_5_24h")
        fcols = meta.get("feature_columns", [])
        rows  = get_latest_features(n=300)
        dfs   = pd.DataFrame(rows)[fcols].dropna()
        if dfs.empty:
            return None
        inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
        try:
            sv = shap.TreeExplainer(inner).shap_values(dfs)
        except Exception:
            sv = shap.LinearExplainer(inner, dfs).shap_values(dfs)
        mean_shap = np.abs(sv).mean(axis=0)
        return (pd.DataFrame({"feature": fcols, "importance": mean_shap})
                .sort_values("importance", ascending=False)
                .head(15))
    except Exception:
        return None

with st.spinner("Computing SHAP values…"):
    shap_df = compute_shap()

if shap_df is not None and not shap_df.empty:
    fig_shap = go.Figure(go.Bar(
        x=shap_df["importance"],
        y=shap_df["feature"],
        orientation="h",
        marker=dict(
            color=shap_df["importance"],
            colorscale=[[0, "#1a2744"], [0.5, "#3b82f6"], [1, "#ff6b6b"]],
            showscale=False,
            line_width=0,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        height=420,
        title=dict(text="Top 15 Feature Importances (PM2.5 +24h)",
                   font=dict(size=13, color="#8badc8")),
        **CHART_TEMPLATE,
    )
    fig_shap.update_xaxes(title="Mean |SHAP|")
    fig_shap.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#8badc8"))
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("SHAP values not available. Make sure train.py has been run.", icon="ℹ️")

# ── Footer ────────────────────────────────────────────────────────────────────
divider()
st.markdown("""
<div style="text-align:center; padding:0.6rem 0; color:#3a5a7a; font-size:0.72rem;">
    Data: <strong style="color:#4a7aaa;">OpenWeather API</strong> &nbsp;·&nbsp;
    Store: <strong style="color:#4a7aaa;">MongoDB Atlas</strong> &nbsp;·&nbsp;
    Models: <strong style="color:#4a7aaa;">Hugging Face Hub</strong> &nbsp;·&nbsp;
    Automation: <strong style="color:#4a7aaa;">GitHub Actions</strong> &nbsp;·&nbsp;
    Built with <strong style="color:#4a7aaa;">Streamlit</strong>
</div>
""", unsafe_allow_html=True)
