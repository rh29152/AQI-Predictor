"""
streamlit_app.py — Interactive AQI Predictor Dashboard for Karachi.

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
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (only after path fix) ────────────────────────────────────────
from src.config import AQI_CATEGORIES
from src.database import get_latest_features, get_collection, ensure_indexes
from src.predict import predict_all_horizons
from src.model_registry import list_models
from src.utils import aqi_label, aqi_color
from src.config import FEATURES_COLLECTION, MODELS_COLLECTION


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_recent_data(hours: int = 168) -> pd.DataFrame:
    """Load the last `hours` feature rows from MongoDB."""
    rows = get_latest_features(n=hours)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(ttl=600)
def load_predictions() -> dict:
    """Run prediction pipeline (cached 10 min)."""
    try:
        return predict_all_horizons()
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=600)
def load_model_registry() -> list[dict]:
    """Load all model metadata from MongoDB."""
    try:
        return list_models()
    except Exception:
        return []


def aqi_badge(aqi_val: float | None) -> str:
    """Return a coloured HTML badge for an AQI value."""
    if aqi_val is None:
        return "<span style='color:gray'>N/A</span>"
    label = aqi_label(aqi_val)
    color = aqi_color(aqi_val)
    return f"<span style='background:{color};padding:3px 10px;border-radius:6px;color:#000;font-weight:bold'>{label}</span>"


def alert_if_hazardous(preds: dict) -> None:
    """Show a Streamlit warning/error if any prediction is Poor or Very Poor."""
    for horizon, info in preds.items():
        aqi = info.get("aqi")
        if aqi is not None and aqi >= 4:
            hours = horizon.replace("target_aqi_", "").replace("h", "")
            st.error(
                f"⚠️ **Air Quality Alert**: Predicted AQI in +{hours}h is "
                f"**{aqi_label(aqi)}** (AQI={aqi:.0f}). "
                "Take precautions — limit outdoor activity."
            )


# ── SHAP feature importance ────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def compute_shap(target: str = "target_aqi_24h") -> pd.DataFrame | None:
    """Compute SHAP mean absolute values for the best 24h model."""
    try:
        import shap
        from src.model_registry import load_model

        model, meta = load_model(target=target)
        feature_columns = meta.get("feature_columns", [])
        rows = get_latest_features(n=200)
        df = pd.DataFrame(rows)[feature_columns].dropna()
        if df.empty:
            return None

        # Use TreeExplainer for tree-based models; LinearExplainer otherwise
        inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
        try:
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(df)
        except Exception:
            explainer = shap.LinearExplainer(inner, df)
            shap_values = explainer.shap_values(df)

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {"feature": feature_columns, "mean_shap": mean_shap}
        ).sort_values("mean_shap", ascending=False)
        return shap_df
    except Exception as exc:
        st.warning(f"SHAP computation skipped: {exc}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://openweathermap.org/themes/openweathermap/assets/img/logo_white_cropped.png",
        width=160,
    )
    st.title("🌫️ AQI Predictor")
    st.caption("Karachi, Pakistan")
    st.divider()
    st.markdown("**Data source:** OpenWeather API")
    st.markdown("**Feature store:** MongoDB Atlas")
    st.markdown("**Models:** LR · RF · GB · XGB")
    st.divider()
    hours_window = st.slider("Historical window (hours)", 24, 720, 168, 24)
    st.caption(f"Showing last {hours_window} hours of data.")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Main layout ────────────────────────────────────────────────────────────────

st.title("🌫️ Karachi Air Quality Index Predictor")
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# Initialise MongoDB indexes silently
try:
    ensure_indexes()
except Exception:
    pass

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_recent_data(hours=hours_window)
preds = load_predictions()

# ── Alert box ─────────────────────────────────────────────────────────────────
if "error" not in preds:
    alert_if_hazardous(preds)

# ── Current AQI card + 3-day forecast ─────────────────────────────────────────
st.subheader("Current & Forecast AQI")

current_aqi = df["aqi"].iloc[-1] if not df.empty else None
col_now, col_24, col_48, col_72 = st.columns(4)

with col_now:
    st.metric("Current AQI", current_aqi if current_aqi else "—")
    if current_aqi:
        st.markdown(aqi_badge(current_aqi), unsafe_allow_html=True)

for col, horizon, label in [
    (col_24, "target_aqi_24h", "+24 h"),
    (col_48, "target_aqi_48h", "+48 h"),
    (col_72, "target_aqi_72h", "+72 h"),
]:
    with col:
        info = preds.get(horizon, {})
        aqi_val = info.get("aqi")
        st.metric(f"Predicted {label}", f"{aqi_val:.1f}" if aqi_val else "—")
        st.markdown(aqi_badge(aqi_val), unsafe_allow_html=True)

st.divider()

# ── AQI trend chart ────────────────────────────────────────────────────────────
st.subheader("📈 Recent AQI Trend")
if not df.empty:
    fig_aqi = px.line(
        df,
        x="datetime",
        y="aqi",
        title="AQI over Time",
        labels={"aqi": "AQI (1-5)", "datetime": "Date (UTC)"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig_aqi.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_aqi, use_container_width=True)
else:
    st.info("No historical data available. Run the feature pipeline first.")

# ── Pollutant trend chart ──────────────────────────────────────────────────────
st.subheader("🧪 Pollutant Trends")
if not df.empty:
    pollutants = [c for c in ["pm2_5", "pm10", "no2", "o3", "so2", "co"] if c in df.columns]
    selected_pollutants = st.multiselect(
        "Select pollutants:", pollutants, default=["pm2_5", "pm10", "no2"]
    )
    if selected_pollutants:
        fig_poll = px.line(
            df,
            x="datetime",
            y=selected_pollutants,
            title="Pollutant Concentrations (μg/m³)",
            labels={"value": "Concentration", "datetime": "Date (UTC)", "variable": "Pollutant"},
        )
        fig_poll.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_poll, use_container_width=True)

st.divider()

# ── Model performance table ────────────────────────────────────────────────────
st.subheader("🏆 Model Performance Registry")
registry = load_model_registry()
if registry:
    rows = []
    for rec in registry[:20]:
        metrics = rec.get("metrics", {})
        rows.append(
            {
                "Model": rec.get("model_name", "—"),
                "Target": rec.get("target", "—"),
                "MAE": metrics.get("mae", "—"),
                "RMSE": metrics.get("rmse", "—"),
                "R²": metrics.get("r2", "—"),
                "Trained At": rec.get("trained_at", "—"),
            }
        )
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
else:
    st.info("No models in registry. Run train.py first.")

st.divider()

# ── SHAP feature importance ────────────────────────────────────────────────────
st.subheader("🔍 SHAP Feature Importance (24h Forecast)")
with st.spinner("Computing SHAP values…"):
    shap_df = compute_shap(target="target_aqi_24h")

if shap_df is not None and not shap_df.empty:
    top_n = min(15, len(shap_df))
    fig_shap = px.bar(
        shap_df.head(top_n),
        x="mean_shap",
        y="feature",
        orientation="h",
        title="Mean |SHAP| — Top Feature Importances",
        labels={"mean_shap": "Mean |SHAP value|", "feature": "Feature"},
        color="mean_shap",
        color_continuous_scale="Blues",
    )
    fig_shap.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("SHAP values not available. Ensure a trained model exists.")

st.divider()

# ── AQI scale legend ───────────────────────────────────────────────────────────
st.subheader("📘 AQI Scale Reference")
legend_cols = st.columns(5)
for (idx, label), col in zip(AQI_CATEGORIES.items(), legend_cols):
    color = aqi_color(idx)
    with col:
        st.markdown(
            f"<div style='background:{color};padding:12px;border-radius:8px;"
            f"text-align:center;color:#000;font-weight:bold'>"
            f"{idx}<br>{label}</div>",
            unsafe_allow_html=True,
        )

st.caption("Data from OpenWeather API · Built with Streamlit · Powered by MongoDB Atlas")
