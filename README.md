# Karachi AQI Predictor

An automated air-quality forecasting system for Karachi, Pakistan. The pipeline ingests hourly pollution and weather data, engineers temporal features in MongoDB Atlas, trains twelve horizon-specific regressors, and serves EPA-style AQI forecasts through a Streamlit dashboard.

The design follows a **pollutant-first** approach: models predict future concentrations of PM2.5, PM10, O₃, and NO₂ at 24 h, 48 h, and 72 h lead times. Each concentration maps to an EPA sub-index via breakpoint interpolation; the reported AQI is the maximum sub-index (dominant pollutant), not OpenWeather’s 1–5 category scale.

> **Scope note:** Regulatory AQI relies on pollutant-specific averaging windows (e.g. 24 h PM, 8 h O₃). This system uses hourly forecast values as approximations for educational forecasting and is not intended as regulatory-grade AQI.

---

## Architecture

```
OpenWeather API (air pollution + weather)
        │
        ▼  hourly — GitHub Actions
  Feature pipeline (hourly_pipeline.py)
  ├── fetch_openweather.py      — live and historical raw ingestion
  ├── feature_engineering.py    — lag, rolling, calendar features
  └── optional catch-up          — fills gaps up to 48 h since last feature row
        │
        ▼
  MongoDB Atlas
  ├── raw_data         — hourly API records (upsert by datetime + city)
  ├── features         — engineered rows; batch rows include 12 target columns
  ├── model_registry   — active model metadata (metrics, HF paths, feature schema)
  └── predictions      — forecast documents from predict.py
        │
        ▼  daily — GitHub Actions
  Training pipeline (train.py)
  ├── 12 targets       — 4 pollutants × 24/48/72 h horizons
  ├── 4 algorithms     — Ridge, Random Forest, Gradient Boosting, XGBoost
  └── selection        — lowest test RMSE among non-overfitting candidates
        │
        ▼
  Hugging Face Hub (optional)
  └── models/<target>/<timestamp>/{model.pkl, metadata.json}
        │
        ▼
  Inference (predict.py)
  ├── twelve regressors → pollutant concentrations (clipped ≥ 0)
  └── aqi_utils.calculate_final_aqi() → EPA AQI + dominant pollutant
        │
        ▼
  Streamlit dashboard (app/streamlit_app.py)
  └── current AQI, horizon cards, trends, optional SHAP importances
```

Historical bootstrap uses `backfill.py` (~90 days of OpenWeather pollution plus Open-Meteo weather) before the hourly pipeline maintains the live window.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data sources | OpenWeather Air Pollution API; Open-Meteo archive (weather backfill) |
| Feature store | MongoDB Atlas |
| ML | scikit-learn (Ridge, RF, GB), XGBoost |
| Model registry | Hugging Face Hub + local `models/` cache |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Automation | GitHub Actions |
| Serialisation | joblib |

---

## Project structure

```
├── .github/workflows/
│   ├── feature_pipeline.yml    # Hourly ingestion + incremental features
│   └── training_pipeline.yml   # Daily training + HF upload + artefact backup
├── app/
│   └── streamlit_app.py        # Forecast dashboard
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory analysis
├── scripts/
│   ├── verify_latest_feature.py
│   └── flush_and_reset.py      # One-time migration reset utility
├── src/
│   ├── config.py               # Environment and schema constants
│   ├── database.py             # MongoDB access layer
│   ├── fetch_openweather.py    # OpenWeather client
│   ├── feature_engineering.py  # Batch and incremental feature builders
│   ├── backfill.py             # Historical raw fetch + batch features
│   ├── hourly_pipeline.py      # Live hourly orchestration + catch-up
│   ├── train.py                # Multi-target training pipeline
│   ├── predict.py              # Multi-horizon inference
│   ├── model_registry.py       # Save / load with HF + MongoDB
│   ├── hf_model_registry.py    # Hugging Face Hub I/O
│   ├── aqi_utils.py            # EPA breakpoint AQI
│   ├── cleanup_old_models.py   # Legacy registry housekeeping
│   └── utils.py                # Logging and presentation helpers
├── models/                     # Local model cache (gitignored)
├── data/                       # Local exports (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Data and features

**Sources:** [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution) for pollution; Open-Meteo archive for historical weather during backfill.

**Raw hourly fields** include OpenWeather AQI category (1–5), pollutant concentrations (PM2.5, PM10, O₃, NO₂, CO, SO₂, NH₃), and meteorology (temperature, humidity, pressure, wind speed, cloud cover).

**Engineered inputs** (per pollutant where applicable: PM2.5, PM10, O₃, NO₂):

| Category | Examples |
|---|---|
| Calendar | `hour`, `day`, `month`, `weekday`, `is_weekend` |
| Input signal | `aqi_category` (OpenWeather 1–5, not the forecast target) |
| Lags | `{pollutant}_lag_1`, `_lag_24`, `_lag_48` |
| Rolling means | `{pollutant}_rolling_6_mean`, `_rolling_12_mean`, `_rolling_24_mean` |
| Momentum | `{pollutant}_change_rate` |

**Supervised targets** (batch / backfill rows only — twelve columns):

`target_{pollutant}_{24|48|72}h` for `pm2_5`, `pm10`, `o3`, `no2`.

Incremental hourly rows omit targets because future concentrations are unknown at ingest time.

---

## Model training

Training evaluates four regressors per target (48 candidate fits per daily run). Features use a temporal 80/20 split (no shuffle) to respect lag structure. Overfitting is flagged when both test/train RMSE ratio and train–test R² gap exceed configured thresholds; selection prefers non-overfitting models, then lowest test RMSE.

Winning models are serialised locally, uploaded to Hugging Face when credentials are present, and registered in MongoDB with `active=True` for their target.

| Algorithm | Role |
|---|---|
| Ridge (+ StandardScaler) | Linear baseline with correlated lag features |
| RandomForestRegressor | Shallow ensemble, subsampled rows/features |
| GradientBoostingRegressor | Boosted trees with regularisation |
| XGBRegressor | Gradient boosting with L1/L2 penalties |

**Metrics:** MAE, RMSE, and R² on train and holdout sets; overfit ratio and R² gap for diagnostics.

---

## Automation

| Workflow | Schedule | Responsibility |
|---|---|---|
| Feature pipeline | Every hour (`:00` UTC) | Catch-up (≤48 h), live OpenWeather fetch, single-row feature upsert |
| Training pipeline | Daily 02:00 UTC | Full retrain of 12 targets, HF upload, 30-day Actions artefact backup |

Both workflows support manual dispatch from the GitHub Actions UI.

---

## Configuration

Runtime settings load from environment variables. Local development typically uses a `.env` file; CI injects the same keys as repository secrets.

| Variable | Purpose |
|---|---|
| `OPENWEATHER_API_KEY` | OpenWeather API access (required) |
| `MONGODB_URI` | MongoDB Atlas connection string (required) |
| `DB_NAME` | Database name (default: `aqi_predictor`) |
| `HF_TOKEN` | Hugging Face write token (optional; enables Hub upload) |
| `HF_REPO_ID` | Hugging Face model repo id (optional; pairs with `HF_TOKEN`) |

Without Hugging Face credentials, models persist locally and metadata still registers in MongoDB; Hub download on fresh runners requires prior upload or a local cache.

---

## Local development

A Python 3.11 virtual environment with dependencies from `requirements.txt` provides the runtime. Environment variables supply API and database credentials.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**Initial data:** `backfill.py` populates ~90 days of raw history and batch-engineered features with supervised targets. `--rebuild-features` regenerates the features collection from existing raw rows after schema changes.

**Training and inference:** `train.py` fits all twelve targets; `predict.py` loads active models and writes optional forecast documents to MongoDB.

**Dashboard:** `streamlit run app/streamlit_app.py` binds the UI to the latest feature row and live predictions.

**Diagnostics:** `scripts/verify_latest_feature.py` compares MongoDB state against `predict.py` output.

---

## Deployment

**Streamlit Community Cloud** connects to the repository with main file `app/streamlit_app.py`. Secrets mirror the environment table above (TOML format under Advanced settings).

**Container platforms** (Cloud Run, Railway, etc.) use a standard Python 3.11 image, install `requirements.txt`, expose port 8501, and launch Streamlit bound to `0.0.0.0`.

---

## EDA notebook

`notebooks/01_eda.ipynb` explores historical trends, diurnal and weekly patterns, pollutant correlations, lag/rolling feature behaviour, and target distributions after the feature store contains backfilled data.

---

## Future directions

- Sequence models (LSTM / temporal transformers) for multi-step forecasting
- Secondary validation sources (e.g. AQICN)
- Threshold-based alert channels (email / SMS)
- Multi-city expansion
- FastAPI inference endpoint
- Feature drift monitoring and automated retrain triggers

---

## License

MIT License — free to use, modify, and distribute.
