"""
streamlit_app.py — Karachi AQI Predictor Dashboard.

Displays computed current AQI and 24h/48h/72h pollutant-first AQI forecasts.
AQI is computed via EPA-style breakpoint interpolation, not the OpenWeather 1-5
category scale.

Launch:
    streamlit run app/streamlit_app.py
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
    initial_sidebar_state="expanded",
)

from src.database import get_latest_features, get_collection, ensure_indexes
from src.predict import predict_all_horizons
from src.model_registry import list_models
from src.aqi_utils import calculate_final_aqi, aqi_category, EPA_AQI_CATEGORIES
from src.config import FEATURES_COLLECTION, MODELS_COLLECTION, POLLUTANTS_TO_FORECAST

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "o3": "O3", "no2": "NO2",
    "co": "CO", "so2": "SO2", "nh3": "NH3",
}
POLLUTANT_COLORS = {
    "pm2_5": "#e74c3c", "pm10": "#e67e22",
    "o3": "#3498db",   "no2": "#9b59b6",
}

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_recent_features(hours: int = 168) -> pd.DataFrame:
    rows = get_latest_features(n=hours)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(ttl=600)
def load_forecasts() -> dict:
    try:
        return predict_all_horizons()
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=600)
def load_model_registry() -> list[dict]:
    try:
        return list_models()
    except Exception:
        return []


# ── UI helpers ─────────────────────────────────────────────────────────────────

def aqi_card_html(aqi: int | None, category: str, color: str, title: str) -> str:
    if aqi is None:
        return f"""
        <div style='background:#1e1e1e;border-radius:12px;padding:18px;text-align:center;'>
            <div style='font-size:13px;color:#aaa;'>{title}</div>
            <div style='font-size:36px;font-weight:bold;color:#666;'>—</div>
            <div style='font-size:14px;color:#666;'>No data</div>
        </div>"""
    return f"""
    <div style='background:{color}22;border:2px solid {color};border-radius:12px;
                padding:18px;text-align:center;'>
        <div style='font-size:13px;color:#ccc;margin-bottom:4px;'>{title}</div>
        <div style='font-size:44px;font-weight:bold;color:{color};line-height:1.1;'>{aqi}</div>
        <div style='font-size:14px;font-weight:600;color:{color};margin-top:4px;'>{category}</div>
    </div>"""


def alert_banners(forecasts: dict) -> None:
    for hz, info in forecasts.items():
        aqi = info.get("predicted_aqi")
        cat = info.get("aqi_category", "")
        if aqi is None:
            continue
        if aqi >= 301:
            st.error(
                f"🚨 **HAZARDOUS** alert for +{hz}: AQI={aqi} — {cat}. "
                "Avoid all outdoor activity. Wear N95 mask if going outside."
            )
        elif aqi >= 201:
            st.error(
                f"⚠️ **Very Unhealthy** alert for +{hz}: AQI={aqi} — {cat}. "
                "Avoid prolonged outdoor exertion."
            )
        elif aqi >= 151:
            st.warning(
                f"⚠️ **Unhealthy** alert for +{hz}: AQI={aqi} — {cat}. "
                "Sensitive groups should limit outdoor activity."
            )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌫️ AQI Predictor")
    st.caption("Karachi, Pakistan")
    st.divider()
    st.markdown("**Approach:** Pollutant-first forecasting")
    st.markdown("**Data source:** OpenWeather API")
    st.markdown("**Feature store:** MongoDB Atlas")
    st.markdown("**Model registry:** Hugging Face Hub")
    st.markdown("**Models:** Ridge · RF · GBR · XGB")
    st.divider()
    hours_window = st.slider("Historical window (hours)", 24, 720, 168, 24)
    st.caption(f"Showing last {hours_window} hours.")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(
        "AQI computed via EPA breakpoint interpolation from predicted "
        "PM2.5, PM10, O3, NO2 concentrations.  Educational approximation "
        "— not regulatory-grade."
    )


# ── Main page ──────────────────────────────────────────────────────────────────

st.title("🌫️ Karachi Air Quality Index Predictor")
st.caption(
    f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
    "Pollutant-first forecast → EPA AQI"
)

try:
    ensure_indexes()
except Exception:
    pass

df = load_recent_features(hours=hours_window)
forecasts = load_forecasts()

# ── Alerts ─────────────────────────────────────────────────────────────────────
if "error" not in forecasts:
    alert_banners(forecasts)
elif "error" in forecasts:
    st.error(f"Forecast error: {forecasts['error']}")

# ── Current + forecast AQI cards ───────────────────────────────────────────────
st.subheader("Current & 3-Day Forecast AQI")

current_aqi_result = None
if not df.empty:
    latest = df.iloc[-1]
    current_pollutants = {
        p: float(latest.get(p, 0) or 0) for p in POLLUTANTS_TO_FORECAST
    }
    current_aqi_result = calculate_final_aqi(current_pollutants)

col_now, col_24, col_48, col_72 = st.columns(4)

with col_now:
    if current_aqi_result and current_aqi_result["aqi"] is not None:
        cat, color = aqi_category(current_aqi_result["aqi"])
        st.markdown(
            aqi_card_html(current_aqi_result["aqi"], cat, color, "Current AQI"),
            unsafe_allow_html=True,
        )
        st.caption(f"Dominant: **{POLLUTANT_LABELS.get(current_aqi_result.get('dominant_pollutant',''), '')}**")
    else:
        st.info("No current data")

for col, hz_label in [(col_24, "24h"), (col_48, "48h"), (col_72, "72h")]:
    with col:
        info = forecasts.get(hz_label, {})
        aqi = info.get("predicted_aqi")
        cat_label = info.get("aqi_category", "—")
        color = info.get("color", "#888")
        dom = info.get("dominant_pollutant", "—")
        st.markdown(
            aqi_card_html(aqi, cat_label, color, f"+{hz_label} Forecast"),
            unsafe_allow_html=True,
        )
        if aqi is not None:
            st.caption(f"Dominant: **{POLLUTANT_LABELS.get(dom, dom)}**")

st.divider()

# ── Pollutant breakdown cards per horizon ──────────────────────────────────────
st.subheader("📊 Pollutant Concentrations by Horizon (μg/m³)")
if forecasts and "error" not in forecasts:
    hz_cols = st.columns(3)
    for col, hz_label in zip(hz_cols, ["24h", "48h", "72h"]):
        with col:
            st.markdown(f"**+{hz_label}**")
            info = forecasts.get(hz_label, {})
            pollutants = info.get("predicted_pollutants", {})
            sub_idxs = info.get("sub_indices", {})
            if pollutants:
                poll_df = pd.DataFrame([
                    {
                        "Pollutant": POLLUTANT_LABELS.get(p, p),
                        "Conc. (μg/m³)": f"{v:.1f}",
                        "Sub-AQI": sub_idxs.get(p, "—"),
                    }
                    for p, v in pollutants.items()
                ])
                st.dataframe(poll_df, hide_index=True, use_container_width=True)

st.divider()

# ── Sub-AQI comparison bar chart ───────────────────────────────────────────────
st.subheader("🎯 Sub-AQI Comparison (all horizons)")
if forecasts and "error" not in forecasts:
    rows = []
    for hz_label in ["24h", "48h", "72h"]:
        info = forecasts.get(hz_label, {})
        for poll, val in info.get("sub_indices", {}).items():
            rows.append({"Horizon": f"+{hz_label}", "Pollutant": POLLUTANT_LABELS.get(poll, poll), "Sub-AQI": val})
    if rows:
        sub_df = pd.DataFrame(rows)
        fig_sub = px.bar(
            sub_df, x="Pollutant", y="Sub-AQI", color="Horizon",
            barmode="group",
            title="Sub-AQI per Pollutant — Dominant pollutant drives final AQI",
            color_discrete_sequence=["#3498db", "#e67e22", "#e74c3c"],
            template="plotly_dark",
        )
        fig_sub.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy threshold (150)")
        fig_sub.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sub, use_container_width=True)

st.divider()

# ── Current AQI trend (computed from recent pollutant data) ────────────────────
st.subheader("📈 Recent Computed AQI Trend")
if not df.empty:
    aqi_vals = []
    for _, row in df.iterrows():
        concs = {p: float(row.get(p, 0) or 0) for p in POLLUTANTS_TO_FORECAST}
        res = calculate_final_aqi(concs)
        aqi_vals.append(res["aqi"])
    df["computed_aqi"] = aqi_vals

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["datetime"], y=df["computed_aqi"],
        mode="lines", name="Computed AQI (EPA)",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ))
    for low, high, label, color in EPA_AQI_CATEGORIES:
        fig_trend.add_hrect(
            y0=low, y1=high,
            fillcolor=color, opacity=0.04, line_width=0,
            annotation_text=label, annotation_position="right",
            annotation=dict(font_size=10, font_color=color),
        )
    fig_trend.update_layout(
        title="EPA AQI computed from current pollutant concentrations",
        xaxis_title="Date (UTC)", yaxis_title="AQI (0-500)",
        height=380, template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Pollutant trends ──────────────────────────────────────────────────────────
st.subheader("🧪 Pollutant Concentration Trends")
if not df.empty:
    avail_polls = [p for p in ["pm2_5", "pm10", "o3", "no2", "co", "so2"] if p in df.columns]
    selected = st.multiselect(
        "Pollutants to display:",
        options=avail_polls,
        default=["pm2_5", "pm10", "o3", "no2"],
        format_func=lambda x: POLLUTANT_LABELS.get(x, x),
    )
    if selected:
        fig_polls = make_subplots(
            rows=len(selected), cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{POLLUTANT_LABELS.get(p, p)} (μg/m³)" for p in selected],
            vertical_spacing=0.04,
        )
        colors = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#2ecc71", "#1abc9c"]
        for i, (poll, color) in enumerate(zip(selected, colors), 1):
            fig_polls.add_trace(
                go.Scatter(
                    x=df["datetime"], y=df[poll],
                    mode="lines", name=POLLUTANT_LABELS.get(poll, poll),
                    line=dict(color=color, width=1.2),
                ),
                row=i, col=1,
            )
        fig_polls.update_layout(
            height=200 * len(selected), template="plotly_dark",
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_polls, use_container_width=True)

st.divider()

# ── Predicted AQI over 3 days ──────────────────────────────────────────────────
st.subheader("🔮 Forecast AQI Timeline")
if forecasts and "error" not in forecasts and current_aqi_result:
    timeline_rows = []
    if current_aqi_result["aqi"] is not None:
        timeline_rows.append({
            "Label": "Now",
            "AQI": current_aqi_result["aqi"],
            "Category": current_aqi_result["category"],
            "Color": current_aqi_result["color"],
        })
    for hz_label in ["24h", "48h", "72h"]:
        info = forecasts.get(hz_label, {})
        if info.get("predicted_aqi") is not None:
            timeline_rows.append({
                "Label": f"+{hz_label}",
                "AQI": info["predicted_aqi"],
                "Category": info["aqi_category"],
                "Color": info["color"],
            })
    if timeline_rows:
        tl_df = pd.DataFrame(timeline_rows)
        fig_tl = go.Figure(go.Bar(
            x=tl_df["Label"], y=tl_df["AQI"],
            marker_color=tl_df["Color"],
            text=tl_df.apply(lambda r: f"AQI {r['AQI']}<br>{r['Category']}", axis=1),
            textposition="outside",
        ))
        fig_tl.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy (150)")
        fig_tl.update_layout(
            title="AQI Forecast: Now → +72h",
            yaxis_title="EPA AQI (0-500)", yaxis_range=[0, max(tl_df["AQI"].max() * 1.3, 200)],
            height=400, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ── SHAP feature importance ────────────────────────────────────────────────────
st.subheader("🔍 SHAP Feature Importance (PM2.5 +24h Model)")

@st.cache_data(ttl=1800)
def compute_shap_pm25() -> pd.DataFrame | None:
    try:
        import shap
        from src.model_registry import load_model

        model, meta = load_model(target="target_pm2_5_24h")
        feature_columns = meta.get("feature_columns", [])
        rows = get_latest_features(n=200)
        df_shap = pd.DataFrame(rows)[feature_columns].dropna()
        if df_shap.empty:
            return None
        inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
        try:
            explainer = shap.TreeExplainer(inner)
            shap_vals = explainer.shap_values(df_shap)
        except Exception:
            explainer = shap.LinearExplainer(inner, df_shap)
            shap_vals = explainer.shap_values(df_shap)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame({"feature": feature_columns, "mean_shap": mean_shap}).sort_values("mean_shap", ascending=False)
    except Exception as exc:
        return None

with st.spinner("Computing SHAP…"):
    shap_df = compute_shap_pm25()

if shap_df is not None and not shap_df.empty:
    top_n = min(15, len(shap_df))
    fig_shap = px.bar(
        shap_df.head(top_n), x="mean_shap", y="feature",
        orientation="h", title="Mean |SHAP| — PM2.5 +24h Forecast",
        color="mean_shap", color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_shap.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("SHAP values not available. Run train.py first.")

st.divider()

# ── Model performance table ────────────────────────────────────────────────────
st.subheader("🏆 Model Registry (latest per target)")
registry = load_model_registry()
if registry:
    rows_table = []
    seen_targets: set[str] = set()
    for rec in sorted(registry, key=lambda r: str(r.get("trained_at", "")), reverse=True):
        tgt = rec.get("target", "—")
        if tgt in seen_targets:
            continue
        seen_targets.add(tgt)
        m = rec.get("metrics", {})
        rows_table.append({
            "Target": tgt,
            "Pollutant": rec.get("pollutant", "—"),
            "+Horizon": f"{rec.get('horizon_hours', '?')}h",
            "Model": rec.get("model_name", "—"),
            "Test RMSE": round(m.get("test_rmse", m.get("rmse", 0)), 4),
            "Test R²": round(m.get("test_r2", m.get("r2", 0)), 4),
            "Overfit Ratio": m.get("overfit_ratio", "—"),
            "Overfitting": m.get("is_overfitting", "—"),
            "Trained At": str(rec.get("trained_at", "—"))[:19],
        })
    perf_df = pd.DataFrame(rows_table)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
else:
    st.info("No models in registry. Run train.py first.")

st.divider()

# ── AQI scale legend ───────────────────────────────────────────────────────────
st.subheader("📘 EPA AQI Scale Reference")
legend_cols = st.columns(len(EPA_AQI_CATEGORIES))
for (low, high, label, color), col in zip(EPA_AQI_CATEGORIES, legend_cols):
    with col:
        st.markdown(
            f"<div style='background:{color}33;border:1px solid {color};padding:10px;"
            f"border-radius:8px;text-align:center;'>"
            f"<b style='color:{color};'>{low}–{high}</b><br>"
            f"<span style='font-size:12px;'>{label}</span></div>",
            unsafe_allow_html=True,
        )

st.caption(
    "Data: OpenWeather API · Feature store: MongoDB Atlas · "
    "Models: Hugging Face Hub · Built with Streamlit  \n"
    "⚠️ AQI computed from hourly pollutant concentrations as approximations. "
    "Not regulatory-grade."
)
